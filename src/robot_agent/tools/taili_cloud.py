from __future__ import annotations

"""taili_quad 云端同步与证据收集工具。

这些工具负责确定性工作：
- 同步本地生成物到云端 robot_lab 固定路径
- 扫描远端 logs/videos
- 供 LLM Agent 作为工具调用，不承担最终判断
"""

from pathlib import Path
import json
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


def start_remote_training(host: str, port: int, user: str, password: str, command: str, tmp_dir: str, timeout_seconds: int) -> dict[str, str]:
    """在远端通过 tmux 会话异步启动训练命令。

    为规避超长命令行和重定向管道在多层 shell/tmux 间转义解析错位的问题，
    本方法在远端动态生成一个确定性的 bash 启动脚本并执行。
    训练结束后会自动写入 exit_code 并销毁会话。

    Returns:
        {"session_name": str, "log_path": str, "exit_code_path": str, "run_id": str}
    """
    run_id = uuid.uuid4().hex[:12]
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    session_name = f"taili_{run_id}"

    # 在项目内 tmp 目录下以 train 目录划分干净独立的归档子文件夹
    run_tmp_dir = f"{tmp_dir}/train/{timestamp}"
    log_path = f"{run_tmp_dir}/taili_train_{run_id}.log"
    exit_code_path = f"{run_tmp_dir}/taili_train_{run_id}.exit_code"
    script_path = f"{run_tmp_dir}/taili_run_{run_id}.sh"

    # 构建启动脚本内容，先强行 mkdir -p 创建该时间戳子目录，确保重定向 100% 成功
    script_content = (
        f"#!/bin/bash\n"
        f"mkdir -p {shlex.quote(run_tmp_dir)}\n"
        f"{command} 2>&1 | tee {shlex.quote(log_path)}\n"
        f"echo ${{PIPESTATUS[0]}} > {shlex.quote(exit_code_path)}\n"
        f"rm -f \"$0\"\n"
        f"exit\n"
    )

    # 写入脚本前，优先强行通过 SSH 执行一次 mkdir -p 确保时间戳父目录存在，否则 heredoc 写入会因目录缺失报错
    mkdir_cmd = f"mkdir -p {shlex.quote(run_tmp_dir)}"
    execute_ssh_command(host, port, user, password, mkdir_cmd, timeout_seconds)

    # 使用 heredoc 写入远端脚本并赋予执行权限
    write_cmd = (
        f"cat << 'EOF' > {shlex.quote(script_path)}\n"
        f"{script_content}"
        f"EOF\n"
        f"chmod +x {shlex.quote(script_path)}"
    )
    out, err, code = execute_ssh_command(host, port, user, password, write_cmd, timeout_seconds)
    if code != 0:
        raise TailiCloudToolError(f"远端训练启动脚本创建失败: {err or out}")

    # 清理同名旧会话 → 创建新会话 → 发送简洁明了的脚本执行命令
    setup_cmd = (
        f"tmux kill-session -t {shlex.quote(session_name)} 2>/dev/null || true; "
        f"tmux new-session -d -s {shlex.quote(session_name)}; "
        f"tmux send-keys -t {shlex.quote(session_name)} 'bash {script_path}' Enter"
    )
    out, err, code = execute_ssh_command(host, port, user, password, setup_cmd, timeout_seconds)
    if code != 0:
        raise TailiCloudToolError(f"tmux 训练会话创建失败: {err or out}")
    return {"session_name": session_name, "log_path": log_path, "exit_code_path": exit_code_path, "run_id": run_id}


def remote_check_training_status(
    host: str, port: int, user: str, password: str,
    session_name: str, exit_code_path: str, timeout_seconds: int,
) -> dict:
    """检查远端训练状态（基于 tmux 会话）。

    一次 SSH 同时检查：
    1. tmux session 是否仍然存在
    2. exit_code_path 是否已写入

    Returns:
        {"session_alive": bool, "has_exit_code": bool,
         "exit_code": int | None, "status": str}
    """
    cmd = (
        f"SA=0; tmux has-session -t {shlex.quote(session_name)} 2>/dev/null && SA=1; "
        f"EC=''; "
        f"if [ -f {shlex.quote(exit_code_path)} ]; then "
        f"  EC=$(cat {shlex.quote(exit_code_path)} 2>/dev/null | tr -d '[:space:]'); "
        f"fi; "
        f"echo \"___SA___$SA\"; echo \"___EC___$EC\""
    )
    out, err, code = execute_ssh_command(host, port, user, password, cmd, timeout_seconds)
    if code != 0:
        raise TailiCloudToolError(err or out)

    session_alive = False
    exit_code_val: int | None = None
    has_exit_code = False
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("___SA___"):
            session_alive = line.replace("___SA___", "") == "1"
        elif line.startswith("___EC___"):
            ec_str = line.replace("___EC___", "")
            if ec_str:
                has_exit_code = True
                try:
                    exit_code_val = int(ec_str)
                except ValueError:
                    exit_code_val = -1

    if has_exit_code:
        status = "completed" if exit_code_val == 0 else "failed"
    elif session_alive:
        status = "running"
    else:
        status = "unknown_failed"

    return {"session_alive": session_alive, "has_exit_code": has_exit_code, "exit_code": exit_code_val, "status": status}


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


def remote_kill_training(host: str, port: int, user: str, password: str, session_name: str, timeout_seconds: int) -> None:
    """终止远端训练 tmux 会话（会杀死会话内所有进程）。"""
    cmd = f"tmux kill-session -t {shlex.quote(session_name)} 2>/dev/null || true"
    execute_ssh_command(host, port, user, password, cmd, timeout_seconds)


def remote_execute_play_in_tmux(
    host: str, port: int, user: str, password: str,
    session_name: str, play_command: str, tmp_dir: str, timeout_seconds: int,
    poll_interval: int = 10,
) -> tuple[str, int]:
    """在 tmux 会话中执行 play.py 并轮询等待完成。

    如果训练会话仍存在则复用，否则新建同名会话。
    play 的输出同样通过 tee 双写到终端和日志文件。
    执行结束后自动 kill 会话，防止下一轮 Revise 时名称冲突。

    Returns:
        (play_stdout, play_exit_code)
    """
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_tag = session_name.replace("taili_", "")
    
    # 隔离归档子目录，使用 play 划分子目录分隔
    run_tmp_dir = f"{tmp_dir}/play/{timestamp}"
    play_log = f"{run_tmp_dir}/taili_play_{run_tag}.log"
    play_ec = f"{run_tmp_dir}/taili_play_{run_tag}.exit_code"
    script_path = f"{run_tmp_dir}/taili_play_{run_tag}.sh"

    # 构建并写入确定性的 play.py 执行脚本以规避复杂命令行嵌套转义隐患，先强行创建时间戳目录并在完成后自毁
    script_content = (
        f"#!/bin/bash\n"
        f"mkdir -p {shlex.quote(run_tmp_dir)}\n"
        f"{play_command} 2>&1 | tee {shlex.quote(play_log)}\n"
        f"echo ${{PIPESTATUS[0]}} > {shlex.quote(play_ec)}\n"
        f"rm -f \"$0\"\n"
        f"exit\n"
    )

    # 写入脚本前，优先强行通过 SSH 执行一次 mkdir -p 确保时间戳目录存在
    mkdir_cmd = f"mkdir -p {shlex.quote(run_tmp_dir)}"
    execute_ssh_command(host, port, user, password, mkdir_cmd, timeout_seconds)

    # 写入并赋予执行权限
    write_cmd = (
        f"cat << 'EOF' > {shlex.quote(script_path)}\n"
        f"{script_content}"
        f"EOF\n"
        f"chmod +x {shlex.quote(script_path)}"
    )
    execute_ssh_command(host, port, user, password, write_cmd, timeout_seconds)

    # 确保会话存在 + 清理旧 exit_code
    ensure = (
        f"tmux has-session -t {shlex.quote(session_name)} 2>/dev/null || "
        f"tmux new-session -d -s {shlex.quote(session_name)}; "
        f"rm -f {shlex.quote(play_ec)}"
    )
    execute_ssh_command(host, port, user, password, ensure, timeout_seconds)

    # 发送脚本执行指令
    send = f"tmux send-keys -t {shlex.quote(session_name)} 'bash {script_path}' Enter"
    execute_ssh_command(host, port, user, password, send, timeout_seconds)

    # 轮询等待 play 完成
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        time.sleep(poll_interval)
        chk = f"cat {shlex.quote(play_ec)} 2>/dev/null || echo '___NOTDONE___'"
        out, _, _ = execute_ssh_command(host, port, user, password, chk, 30)
        out = out.strip()
        if out and out != "___NOTDONE___":
            try:
                exit_code = int(out)
            except ValueError:
                exit_code = -1
            # 读取 play 输出日志
            log_out, _, _ = execute_ssh_command(
                host, port, user, password,
                f"cat {shlex.quote(play_log)} 2>/dev/null", 60,
            )
            # 清理会话，临时生成的脚本由其自身运行的 rm -f "$0" 机制在 tmux 退出时自动清扫
            remote_kill_training(host, port, user, password, session_name, timeout_seconds)
            return (log_out, exit_code)

    # 超时：kill 会话并报错，临时生成的脚本由其内部自毁处理
    remote_kill_training(host, port, user, password, session_name, timeout_seconds)
    raise TailiCloudToolError(f"play.py 在 {timeout_seconds}s 内未完成")


def remote_list_latest_run(host: str, port: int, user: str, password: str, root: str, timeout_seconds: int) -> str:
    # 纯 shell 匹配格式为 YYYY-MM-DD_HH-MM-SS 的目录并倒序取第一个，避免依赖远端 python 环境
    cmd = (
        f"ls -1d {shlex.quote(root)}/[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]_[0-9][0-9]-[0-9][0-9]-[0-9][0-9]/ 2>/dev/null "
        "| sort | tail -n 1"
    )
    out, err, code = execute_ssh_command(host, port, user, password, cmd, timeout_seconds)
    if code != 0:
        return ""
    return out.strip().rstrip('/')


def remote_find_play_eval_artifacts(
    host: str,
    port: int,
    user: str,
    password: str,
    run_root: str,
    timeout_seconds: int,
    expected_terrains: list[str] | tuple[str, ...] | None = None,
) -> dict[str, dict]:
    """查找 play_eval.py 生成的一组评估产物。

    扫描 ``<run_root>/videos/play_eval/**/eval_meta.json``，读取每个 meta 中的 terrain 字段，
    并在同目录下找到对应的最新 mp4。返回按 terrain 归类后的字典。

    这个函数故意依赖 eval_meta.json，而不是依赖目录名。这样未来目录命名变化时，
    只要 meta 的 terrain 字段稳定，评估链路就不会断。

    Returns:
        {
          "flat": {
            "terrain": "flat",
            "video_remote_path": ".../rl-video-step-0.mp4",
            "meta_remote_path": ".../eval_meta.json",
            "video_folder": "...",
            "meta": {...}
          },
          ...
        }
    """
    expected = set(expected_terrains or [])
    play_eval_dir = posixpath.join(run_root, "videos", "play_eval")
    cmd = (
        f"find {shlex.quote(play_eval_dir)} -mindepth 2 -maxdepth 2 -name 'eval_meta.json' -type f "
        "-printf '%T@ %p\\n' 2>/dev/null | sort -n"
    )
    out, err, code = execute_ssh_command(host, port, user, password, cmd, timeout_seconds)
    if code != 0 or not out.strip():
        return {}

    artifacts: dict[str, dict] = {}
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        # format: <mtime> <path>
        try:
            _, meta_path = line.split(" ", 1)
        except ValueError:
            continue
        meta_path = meta_path.strip()
        meta_out, _, meta_code = execute_ssh_command(
            host, port, user, password, f"cat {shlex.quote(meta_path)} 2>/dev/null", timeout_seconds
        )
        if meta_code != 0 or not meta_out.strip():
            continue
        try:
            meta = json.loads(meta_out)
        except json.JSONDecodeError:
            meta = {"raw_meta_text": meta_out}

        terrain = str(meta.get("terrain") or "").strip()
        if not terrain:
            # 兜底：从上级目录名猜，但优先级低于 meta 字段
            terrain = posixpath.basename(posixpath.dirname(meta_path)).split("_")[0]
        if expected and terrain not in expected:
            continue

        video_folder = posixpath.dirname(meta_path)
        video_cmd = (
            f"find {shlex.quote(video_folder)} -maxdepth 1 -name '*.mp4' -type f "
            "-printf '%T@ %p\\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2-"
        )
        video_out, _, video_code = execute_ssh_command(host, port, user, password, video_cmd, timeout_seconds)
        video_path = video_out.strip() if video_code == 0 else ""
        if not video_path:
            continue

        # 如果同一 run_root 下重复生成过同一 terrain，按 find 的 mtime 排序，后出现的会覆盖旧项。
        artifacts[terrain] = {
            "terrain": terrain,
            "video_remote_path": video_path,
            "meta_remote_path": meta_path,
            "video_folder": video_folder,
            "meta": meta,
        }

    return artifacts


def wait_for_remote_file_stable(host: str, port: int, user: str, password: str, remote_path: str, timeout_seconds: int, polls: int = 3, interval_seconds: int = 2) -> bool:
    last_size: int | None = None
    stable_count = 0
    for _ in range(max(1, polls)):
        # 纯 shell 获取文件大小，如果文件不存在则返回 -1
        cmd = f"stat -c %s {shlex.quote(remote_path)} 2>/dev/null || echo -1"
        out, err, code = execute_ssh_command(host, port, user, password, cmd, timeout_seconds)
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


def download_remote_file_to_temp(host: str, port: int, user: str, password: str, remote_path: str, timeout_seconds: int, suffix: str = ".mp4") -> str:
    local_dir = Path(tempfile.mkdtemp(prefix="taili-video-"))
    local_path = local_dir / (Path(remote_path).name or f"video{suffix}")
    fetch_remote_file(host, port, user, password, remote_path, str(local_path), timeout_seconds)
    return str(local_path)


def remote_file_exists(host: str, port: int, user: str, password: str, remote_path: str, timeout_seconds: int) -> bool:
    """判断远端是否存在某个文件（续训前确认最优 checkpoint 仍在，避免 get_checkpoint_path 直接报错浪费一轮）。"""
    if not remote_path:
        return False
    cmd = f"test -f {shlex.quote(remote_path)} && echo __YES__ || echo __NO__"
    out, err, code = execute_ssh_command(host, port, user, password, cmd, timeout_seconds)
    return "__YES__" in out


def remote_log_contains(host: str, port: int, user: str, password: str, log_path: str, marker: str, timeout_seconds: int) -> bool:
    """服务端整文件 grep 某个标志串（不受增量 byte-offset 窗口影响，用于确定性校验续训是否真的加载了 checkpoint）。"""
    cmd = f"grep -F -q -- {shlex.quote(marker)} {shlex.quote(log_path)} && echo __YES__ || echo __NO__"
    out, err, code = execute_ssh_command(host, port, user, password, cmd, timeout_seconds)
    return "__YES__" in out


def remote_find_latest_checkpoint(host: str, port: int, user: str, password: str, run_dir: str, timeout_seconds: int) -> str:
    """在远端 run_dir 下找到迭代号最大的 model_*.pt（兜底用，正常应直接取 eval_meta.checkpoint）。

    rsl_rl 的 checkpoint 命名为 ``model_<iter>.pt``。这里抽取出迭代号做数值排序，
    取最大的一个并重组成完整路径。找不到时返回空串。
    """
    cmd = (
        f"ls -1 {shlex.quote(run_dir)}/model_*.pt 2>/dev/null "
        f"| sed 's#.*/model_##; s#\\.pt$##' | grep -E '^[0-9]+$' | sort -n | tail -n 1"
    )
    out, err, code = execute_ssh_command(host, port, user, password, cmd, timeout_seconds)
    if code != 0 or not out.strip():
        return ""
    return posixpath.join(run_dir, f"model_{out.strip()}.pt")


def download_checkpoint_bundle(
    host: str, port: int, user: str, password: str,
    checkpoint_remote: str, run_dir: str, local_dir: str, timeout_seconds: int,
) -> dict:
    """把一轮评估对应的"最优"checkpoint 物料下载到本地 local_dir。

    会先清空 local_dir 以保证"唯一一份最优"，再按需下载以下内容（缺失项静默跳过，不报错）：
      - 原始 RL checkpoint：checkpoint_remote（model_*.pt，可续训/即最优参数）
      - 导出策略：run_dir/exported/policy.pt、policy.onnx（可直接部署推理）
      - 配置快照：run_dir/params/env.yaml、agent.yaml（复现训练所需）

    Returns:
        {"local_dir": str, "files": {logical_name: local_path}, "missing": [logical_name, ...]}
    """
    local_dir_p = Path(local_dir)
    # 清空旧的最优目录，避免上一份最优的残留文件与本份混淆
    if local_dir_p.exists():
        shutil.rmtree(local_dir_p, ignore_errors=True)
    local_dir_p.mkdir(parents=True, exist_ok=True)

    targets: list[tuple[str, str, Path]] = []
    if checkpoint_remote:
        targets.append(("checkpoint", checkpoint_remote, local_dir_p / posixpath.basename(checkpoint_remote)))
    if run_dir:
        targets.append(("exported_policy", posixpath.join(run_dir, "exported", "policy.pt"), local_dir_p / "policy.pt"))
        targets.append(("exported_onnx", posixpath.join(run_dir, "exported", "policy.onnx"), local_dir_p / "policy.onnx"))
        targets.append(("env_yaml", posixpath.join(run_dir, "params", "env.yaml"), local_dir_p / "env.yaml"))
        targets.append(("agent_yaml", posixpath.join(run_dir, "params", "agent.yaml"), local_dir_p / "agent.yaml"))

    got: dict[str, str] = {}
    missing: list[str] = []
    for name, remote, local in targets:
        try:
            fetch_remote_file(host, port, user, password, remote, str(local), timeout_seconds)
            got[name] = str(local)
        except Exception:
            missing.append(name)
    return {"local_dir": str(local_dir_p), "files": got, "missing": missing}
