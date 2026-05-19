from __future__ import annotations

"""taili_quad 云端同步与证据收集工具。

这些工具负责确定性工作：
- 同步本地生成物到云端 robot_lab 固定路径
- 扫描远端 logs/checkpoints/videos/tensorboard
- 供 LLM Agent 作为工具调用，不承担最终判断
"""

from pathlib import Path
import posixpath
import shlex
import shutil
import tempfile
import time
import uuid

import paramiko

from robot_agent.tools.ssh_client import execute_ssh_command


class TailiCloudToolError(RuntimeError):
    """Taili 云端工具失败时抛出。"""


def upload_files_via_sftp(host: str, port: int, user: str, password: str, files: list[tuple[str, str]], timeout_seconds: int) -> list[dict[str, str]]:
    """通过 SFTP 上传文件到远端固定路径。"""

    transport = paramiko.Transport((host, port))
    transport.banner_timeout = timeout_seconds
    transport.auth_timeout = timeout_seconds
    transport.connect(username=user, password=password)
    sftp = paramiko.SFTPClient.from_transport(transport)
    uploaded: list[dict[str, str]] = []
    try:
        for src_rel, dst_rel in files:
            src = Path(src_rel)
            dst = posixpath.normpath(dst_rel)
            parent = posixpath.dirname(dst)
            if parent:
                _mkdir_p_sftp(sftp, parent)
            sftp.put(str(src), dst)
            uploaded.append({"src": str(src), "dst": dst, "status": "uploaded"})
    finally:
        sftp.close()
        transport.close()
    return uploaded


def remote_upload_taili_workspace(
    host: str, port: int, user: str, password: str, 
    local_root: str, 
    cloud_root: str,
    cloud_asset_path: str, cloud_task_cfg_root: str, 
    timeout_seconds: int
) -> list[dict[str, str]]:
    """把 taili 本地 workspace 关键产物（包括 urdf/meshes 目录和 6 个生成的 Python 文件）上传到远端固定路径。"""

    files = []
    
    # 1. 递归扫描本地模型文件夹 (urdf/, meshes/ 等)
    local_base = Path(local_root)
    cloud_model_dir = posixpath.join(cloud_root, "source/robot_lab/data/Robots/taili_quad")
    
    if local_base.exists():
        for p in local_base.rglob("*"):
            if p.is_file() and ".taili_generated" not in p.parts:
                rel_path = p.relative_to(local_base)
                dst = posixpath.join(cloud_model_dir, rel_path.as_posix())
                files.append((str(p), dst))
                
    # 2. 生成的 6 个 Python 文件
    gen_dir = Path(local_root) / ".taili_generated"
    config_files = {
        "taili_quad.py": posixpath.join(cloud_root, cloud_asset_path),
        "agents/__init__.py": posixpath.join(cloud_root, cloud_task_cfg_root, "agents/__init__.py"),
        "agents/rsl_rl_ppo_cfg.py": posixpath.join(cloud_root, cloud_task_cfg_root, "agents/rsl_rl_ppo_cfg.py"),
        "__init__.py": posixpath.join(cloud_root, cloud_task_cfg_root, "__init__.py"),
        "flat_env_cfg.py": posixpath.join(cloud_root, cloud_task_cfg_root, "flat_env_cfg.py"),
        "rough_env_cfg.py": posixpath.join(cloud_root, cloud_task_cfg_root, "rough_env_cfg.py"),
    }
    
    for rel_src, dst in config_files.items():
        src = gen_dir / rel_src
        if src.exists():
            files.append((str(src), dst))
            
    return upload_files_via_sftp(host, port, user, password, files, timeout_seconds)


def _mkdir_p_sftp(sftp: paramiko.SFTPClient, remote_directory: str) -> None:
    parts = []
    current = remote_directory
    while current not in {"", "/"}:
        parts.append(current)
        current = posixpath.dirname(current)
    for directory in reversed(parts):
        try:
            sftp.stat(directory)
        except OSError:
            try:
                sftp.mkdir(directory)
            except OSError:
                pass


def remote_list_latest_run(host: str, port: int, user: str, password: str, root: str, timeout_seconds: int) -> str:
    cmd = (
        "python - <<'PY'\n"
        "from pathlib import Path\n"
        f"root = Path(r'''{root}''')\n"
        "runs = [p for p in root.rglob('*') if p.is_dir() and (p / 'checkpoints').exists()]\n"
        "runs = sorted(runs, key=lambda p: p.stat().st_mtime)\n"
        "print(runs[-1] if runs else '')\n"
        "PY"
    )
    out, err, code = execute_ssh_command(host, port, user, password, cmd, timeout_seconds)
    if code != 0:
        raise TailiCloudToolError(err or out)
    return out.strip()


def start_remote_training(host: str, port: int, user: str, password: str, command: str, timeout_seconds: int) -> dict[str, str]:
    """在远端异步启动训练命令。

    会为本次训练生成唯一 run_id，训练命令退出后将 exit code 写入文件，
    便于后续通过 remote_check_training_status 判断训练是否正常结束。

    Returns:
        {"pid": str, "log_path": str, "exit_code_path": str, "run_id": str}
    """
    run_id = uuid.uuid4().hex[:12]
    log_path = f"/tmp/taili_train_{run_id}.log"
    exit_code_path = f"/tmp/taili_train_{run_id}.exit_code"

    # 包装命令：训练结束后写入 exit code 文件
    wrapped = (
        f"( bash -lc {shlex.quote(command)}; "
        f"echo $? > {shlex.quote(exit_code_path)} "
        f") > {shlex.quote(log_path)} 2>&1 < /dev/null & echo $!"
    )
    out, err, code = execute_ssh_command(host, port, user, password, wrapped, timeout_seconds)
    if code != 0:
        raise TailiCloudToolError(err or out)
    pid = out.strip()
    if not pid:
        raise TailiCloudToolError("远端训练启动失败：未返回 PID")
    return {"pid": pid, "log_path": log_path, "exit_code_path": exit_code_path, "run_id": run_id}


def remote_check_training_status(
    host: str, port: int, user: str, password: str,
    pid: str, exit_code_path: str, timeout_seconds: int,
) -> dict:
    """检查远端训练进程的运行状态。

    通过一次 SSH 命令同时检查：
    1. pid 是否仍然存活（kill -0）
    2. exit_code_path 文件是否存在，若存在则读取 exit code

    Returns:
        {
            "is_running": bool,
            "has_exit_code": bool,
            "exit_code": int | None,
            "status": "running" | "completed" | "failed" | "unknown_failed"
        }
    """
    cmd = (
        f"ALIVE=0; kill -0 {pid} 2>/dev/null && ALIVE=1; "
        f"EC=''; "
        f"if [ -f {shlex.quote(exit_code_path)} ]; then "
        f"  EC=$(cat {shlex.quote(exit_code_path)} 2>/dev/null | tr -d '[:space:]'); "
        f"fi; "
        f"echo \"___ALIVE___$ALIVE\"; "
        f"echo \"___EC___$EC\""
    )
    out, err, code = execute_ssh_command(host, port, user, password, cmd, timeout_seconds)
    if code != 0:
        raise TailiCloudToolError(err or out)

    is_running = False
    exit_code_val: int | None = None
    has_exit_code = False

    for line in out.splitlines():
        line = line.strip()
        if line.startswith("___ALIVE___"):
            is_running = line.replace("___ALIVE___", "") == "1"
        elif line.startswith("___EC___"):
            ec_str = line.replace("___EC___", "")
            if ec_str:
                has_exit_code = True
                try:
                    exit_code_val = int(ec_str)
                except ValueError:
                    exit_code_val = -1

    # 状态推断
    if has_exit_code:
        status = "completed" if exit_code_val == 0 else "failed"
    elif is_running:
        status = "running"
    else:
        # pid 不存活、且没有 exit_code 文件 → 异常退出
        status = "unknown_failed"

    return {
        "is_running": is_running,
        "has_exit_code": has_exit_code,
        "exit_code": exit_code_val,
        "status": status,
    }


def fetch_remote_file(host: str, port: int, user: str, password: str, remote_path: str, local_path: str, timeout_seconds: int) -> dict[str, str]:
    transport = paramiko.Transport((host, port))
    transport.banner_timeout = timeout_seconds
    transport.auth_timeout = timeout_seconds
    transport.connect(username=user, password=password)
    sftp = paramiko.SFTPClient.from_transport(transport)
    try:
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        sftp.get(remote_path, local_path)
    finally:
        sftp.close()
        transport.close()
    return {"remote_path": remote_path, "local_path": local_path, "status": "downloaded"}


def remote_tail_log(host: str, port: int, user: str, password: str, log_path: str, timeout_seconds: int, byte_offset: int = 0) -> tuple[str, int]:
    """从远端日志文件增量读取内容。

    Args:
        byte_offset: 上一次读取结束时的字节位置。0 表示从头读取。

    Returns:
        (new_text, new_offset): 本次读到的新增文本和新的字节偏移量。
        如果文件不存在或无新增，返回 ("", byte_offset)。
    """
    # 用 shell 命令获取文件大小并增量读取，避免嵌入式 Python 被 .bashrc 干扰。
    # `wc -c < file` 获取字节数；`tail -c +{offset+1}` 从指定偏移开始读取。
    cmd = (
        f"test -f {shlex.quote(log_path)} || {{ echo '___NOFILE___'; exit 0; }}; "
        f"SIZE=$(wc -c < {shlex.quote(log_path)}); "
        f"echo \"___SIZE___$SIZE\"; "
        f"if [ \"$SIZE\" -gt {byte_offset} ]; then "
        f"  tail -c +{byte_offset + 1} {shlex.quote(log_path)}; "
        f"fi"
    )
    out, err, code = execute_ssh_command(host, port, user, password, cmd, timeout_seconds)
    if code != 0:
        raise TailiCloudToolError(err or out)

    if "___NOFILE___" in out:
        return ("", byte_offset)

    # 解析文件大小
    lines = out.split("\n", 1)
    size_line = lines[0].strip()
    new_text = lines[1] if len(lines) > 1 else ""
    try:
        new_offset = int(size_line.replace("___SIZE___", ""))
    except ValueError:
        new_offset = byte_offset + len(new_text.encode("utf-8", errors="replace"))
    return (new_text, new_offset)


def remote_kill_process(host: str, port: int, user: str, password: str, pid: str, timeout_seconds: int) -> None:
    cmd = f"kill -TERM {pid} || true"
    out, err, code = execute_ssh_command(host, port, user, password, cmd, timeout_seconds)
    if code != 0:
        raise TailiCloudToolError(err or out)


def remote_find_latest_video_file(host: str, port: int, user: str, password: str, run_root: str, timeout_seconds: int) -> str:
    cmd = (
        "python - <<'PY'\n"
        "from pathlib import Path\n"
        f"root = Path(r'''{run_root}''') / 'video'\n"
        "candidates = sorted([p for p in root.rglob('*.mp4') if p.is_file()], key=lambda p: p.stat().st_mtime)\n"
        "print(candidates[-1] if candidates else '')\n"
        "PY"
    )
    out, err, code = execute_ssh_command(host, port, user, password, cmd, timeout_seconds)
    if code != 0:
        raise TailiCloudToolError(err or out)
    return out.strip()


def download_remote_file_to_temp(host: str, port: int, user: str, password: str, remote_path: str, timeout_seconds: int, suffix: str = ".mp4") -> str:
    local_dir = Path(tempfile.mkdtemp(prefix="taili-video-"))
    local_path = local_dir / (Path(remote_path).name or f"video{suffix}")
    fetch_remote_file(host, port, user, password, remote_path, str(local_path), timeout_seconds)
    return str(local_path)


def wait_for_remote_file_stable(host: str, port: int, user: str, password: str, remote_path: str, timeout_seconds: int, polls: int = 3, interval_seconds: int = 2) -> bool:
    last_size: int | None = None
    stable_count = 0
    for _ in range(max(1, polls)):
        cmd = (
            "python - <<'PY'\n"
            "from pathlib import Path\n"
            f"p = Path(r'''{remote_path}''')\n"
            "print(p.stat().st_size if p.exists() else -1)\n"
            "PY"
        )
        out, err, code = execute_ssh_command(host, port, user, password, cmd, timeout_seconds)
        if code != 0:
            raise TailiCloudToolError(err or out)
        try:
            size = int(out.strip())
        except ValueError:
            size = -1
        if size > 0 and size == last_size:
            stable_count += 1
        else:
            stable_count = 0
        last_size = size
        if stable_count >= 1:
            return True
        time.sleep(interval_seconds)
    return False


