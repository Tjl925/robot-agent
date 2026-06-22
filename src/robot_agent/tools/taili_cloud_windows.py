from __future__ import annotations

"""taili_quad 云端同步与证据收集工具 —— Windows / PowerShell 7 后端。

这是 taili_cloud.py（Linux: bash + tmux + posix 命令）的平行实现，面向一台
远端 shell 为 PowerShell 7（pwsh）的 Windows 主机。函数签名与 taili_cloud 逐一对齐，
使得 taili_steps 可以按 remote_platform 无缝切换后端（self._remote.xxx(...)）。

设计要点：
- 复杂 PowerShell 一律 UTF-16LE base64 编码后用 `pwsh -NoProfile -EncodedCommand` 执行，
  彻底规避命令行引号/换行转义（等价于 Linux 端的 bash heredoc 技巧）。
- 训练/评估用 Start-Process 派生独立后台进程（替代 tmux 会话），脚本自身落 pid 与 exit_code，
  轮询凭 pid 存活 + exit_code 文件判定状态；终止用 `taskkill /T /F` 杀进程树。
- 启动训练/评估的 launcher 进程不加 -NoProfile，让 pwsh profile 里的 conda 初始化生效，
  使 `conda activate` 可用；编排类短命令（探活/查日志/找文件）才用 -NoProfile 提速。
- 路径统一用正斜杠返回（posixpath 可直接复用）；SFTP 落到 OpenSSH 的 `/<盘符>:/...` 形式。
"""

import base64
import json
import posixpath
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path

import paramiko

from robot_agent.tools.ssh_client import execute_ssh_command
from robot_agent.tools.taili_cloud import TailiCloudToolError, _mkdir_p_sftp


# ============================ 基础工具 ============================

def _ps_encode(script: str) -> str:
    """把 PowerShell 脚本编码成 -EncodedCommand 所需的 UTF-16LE base64。"""
    return base64.b64encode(script.encode("utf-16-le")).decode("ascii")


def _run_ps(host: str, port: int, user: str, password: str, script: str, timeout_seconds: int) -> tuple[str, str, int]:
    """以 `pwsh -NoProfile -NonInteractive -EncodedCommand` 执行一段 PowerShell，返回 (stdout, stderr, code)。"""
    cmd = "pwsh -NoProfile -NonInteractive -EncodedCommand " + _ps_encode(script)
    return execute_ssh_command(host, port, user, password, cmd, timeout_seconds)


def _q(value: str) -> str:
    """转义供 PowerShell 单引号字符串内嵌的值（单引号翻倍）。"""
    return str(value).replace("'", "''")


def _fill(template: str, mapping: dict[str, str]) -> str:
    """用 @@TOKEN@@ 占位替换构建脚本，避免与 PowerShell 的 $ 和 {} 冲突。"""
    out = template
    for key, val in mapping.items():
        out = out.replace(key, val)
    return out


def _fwd(path: str) -> str:
    """把远端返回的路径统一成正斜杠（供 posixpath 下游消费）。"""
    return str(path).replace("\\", "/")


def _sftp_path(path: str) -> str:
    """把 Windows 路径转成 OpenSSH SFTP 接受的绝对形式：e:\\a\\b 或 e:/a/b -> /e:/a/b。"""
    p = str(path).replace("\\", "/")
    if re.match(r"^[A-Za-z]:", p):
        p = "/" + p
    return p


# ============================ SFTP（与平台无关，仅路径形式不同）============================

def upload_files_via_sftp(host: str, port: int, user: str, password: str, files: list[tuple[str, str]], timeout_seconds: int) -> list[dict[str, str]]:
    """通过 SFTP 上传文件到 Windows 远端固定路径。"""
    transport = paramiko.Transport((host, port))
    transport.banner_timeout = timeout_seconds
    transport.auth_timeout = timeout_seconds
    transport.connect(username=user, password=password)
    sftp = paramiko.SFTPClient.from_transport(transport)
    uploaded: list[dict[str, str]] = []
    try:
        for src_rel, dst_rel in files:
            src = Path(src_rel)
            dst = _sftp_path(dst_rel)
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
    timeout_seconds: int,
) -> list[dict[str, str]]:
    """把 taili 本地 workspace 关键产物上传到 Windows 远端固定路径（路径形式不同，逻辑同 Linux）。"""
    files = []

    local_base = Path(local_root)
    cloud_model_dir = posixpath.join(cloud_root, "source/robot_lab/data/Robots/taili_quad")

    if local_base.exists():
        for p in local_base.rglob("*"):
            if p.is_file() and ".taili_generated" not in p.parts:
                rel_path = p.relative_to(local_base)
                dst = posixpath.join(cloud_model_dir, rel_path.as_posix())
                files.append((str(p), dst))

    gen_dir = Path(local_root) / ".taili_generated"
    # cloud_asset_path 和 cloud_task_cfg_root 在 Windows 模式下已是完整绝对路径（e:/tjl/...），
    # 不能用 posixpath.join 再拼 cloud_root（posixpath 只认 "/" 开头为绝对路径，"e:/" 不认）。
    config_files = {
        "taili_quad.py": cloud_asset_path,
        "agents/__init__.py": f"{cloud_task_cfg_root}/agents/__init__.py",
        "agents/rsl_rl_ppo_cfg.py": f"{cloud_task_cfg_root}/agents/rsl_rl_ppo_cfg.py",
        "__init__.py": f"{cloud_task_cfg_root}/__init__.py",
        "flat_env_cfg.py": f"{cloud_task_cfg_root}/flat_env_cfg.py",
        "rough_env_cfg.py": f"{cloud_task_cfg_root}/rough_env_cfg.py",
    }
    for rel_src, dst in config_files.items():
        src = gen_dir / rel_src
        if src.exists():
            files.append((str(src), dst))

    return upload_files_via_sftp(host, port, user, password, files, timeout_seconds)


def fetch_remote_file(host: str, port: int, user: str, password: str, remote_path: str, local_path: str, timeout_seconds: int) -> dict[str, str]:
    transport = paramiko.Transport((host, port))
    transport.banner_timeout = timeout_seconds
    transport.auth_timeout = timeout_seconds
    transport.connect(username=user, password=password)
    sftp = paramiko.SFTPClient.from_transport(transport)
    try:
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        sftp.get(_sftp_path(remote_path), local_path)
    finally:
        sftp.close()
        transport.close()
    return {"remote_path": remote_path, "local_path": local_path, "status": "downloaded"}


def download_remote_file_to_temp(host: str, port: int, user: str, password: str, remote_path: str, timeout_seconds: int, suffix: str = ".mp4") -> str:
    local_dir = Path(tempfile.mkdtemp(prefix="taili-video-"))
    local_path = local_dir / (Path(remote_path).name or f"video{suffix}")
    fetch_remote_file(host, port, user, password, remote_path, str(local_path), timeout_seconds)
    return str(local_path)


def download_checkpoint_bundle(
    host: str, port: int, user: str, password: str,
    checkpoint_remote: str, run_dir: str, local_dir: str, timeout_seconds: int,
) -> dict:
    """把一轮评估对应的“最优”checkpoint 物料下载到本地 local_dir（逻辑同 Linux，路径用 SFTP 转换）。"""
    local_dir_p = Path(local_dir)
    if local_dir_p.exists():
        shutil.rmtree(local_dir_p, ignore_errors=True)
    local_dir_p.mkdir(parents=True, exist_ok=True)

    # posixpath.basename 只认 "/" 为分隔符，对含反斜杠的 Windows 路径会返回整串。
    # 用跨平台方式：先把 "\" 转为 "/"，再取最后一个 "/" 之后的部分。
    def _win_basename(p: str) -> str:
        return p.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1] or p

    targets: list[tuple[str, str, Path]] = []
    if checkpoint_remote:
        targets.append(("checkpoint", checkpoint_remote, local_dir_p / _win_basename(checkpoint_remote)))
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


# ============================ PowerShell 脚本模板 ============================

# 训练 launcher（被 Start-Process 以独立后台进程执行；不加 -NoProfile 以加载 conda 初始化）。
_LAUNCHER_TEMPLATE = r"""$ErrorActionPreference = 'Continue'
# 强制 PowerShell 进程及其子进程使用 UTF-8，避免 Python 输出中文时走 GBK 再被 UTF-8 读乱码。
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = '1'
New-Item -ItemType Directory -Force -Path '@@RUNDIR@@' | Out-Null
$PID | Out-File -FilePath '@@PIDPATH@@' -Encoding ascii -Force
& { @@COMMAND@@ } *>&1 | Tee-Object -FilePath '@@LOGPATH@@' -Encoding utf8
$code = $LASTEXITCODE
if ($null -eq $code) { $code = 0 }
"$code" | Out-File -FilePath '@@EXITPATH@@' -Encoding ascii -Force
"""

# 外层：建目录 -> 落 launcher 脚本 -> Start-Process 后台拉起。
_OUTER_START_TEMPLATE = r"""New-Item -ItemType Directory -Force -Path '@@RUNDIR@@' | Out-Null
Set-Content -Path '@@SCRIPTPATH@@' -Encoding utf8 -Value @'
@@LAUNCHER@@
'@
Start-Process -FilePath 'pwsh' -ArgumentList @('-File','@@SCRIPTPATH@@') -WindowStyle Hidden
Write-Output 'STARTED'
"""


def _build_launcher(run_tmp_dir: str, pid_path: str, log_path: str, exit_code_path: str, command: str) -> str:
    return _fill(_LAUNCHER_TEMPLATE, {
        "@@RUNDIR@@": _q(run_tmp_dir),
        "@@PIDPATH@@": _q(pid_path),
        "@@LOGPATH@@": _q(log_path),
        "@@EXITPATH@@": _q(exit_code_path),
        "@@COMMAND@@": command,
    })


def _start_detached(host: str, port: int, user: str, password: str, run_tmp_dir: str, script_path: str, launcher: str, timeout_seconds: int) -> None:
    outer = _fill(_OUTER_START_TEMPLATE, {
        "@@RUNDIR@@": _q(run_tmp_dir),
        "@@SCRIPTPATH@@": _q(script_path),
        "@@LAUNCHER@@": launcher,
    })
    out, err, code = _run_ps(host, port, user, password, outer, timeout_seconds)
    if code != 0 or "STARTED" not in out:
        raise TailiCloudToolError(f"Windows 后台进程启动失败: {err or out}")


def _pid_path_from_exit(exit_code_path: str) -> str:
    if exit_code_path.endswith(".exit_code"):
        return exit_code_path[: -len(".exit_code")] + ".pid"
    return exit_code_path + ".pid"


# ============================ 训练 ============================

def start_remote_training(host: str, port: int, user: str, password: str, command: str, tmp_dir: str, timeout_seconds: int) -> dict[str, str]:
    """在 Windows 远端以独立后台进程异步启动训练命令。

    返回 {"session_name", "log_path", "exit_code_path", "run_id"}，语义对齐 Linux 版。
    session_name=taili_<run_id>，仅作逻辑句柄；终止时由 run_id 反查进程树。
    """
    run_id = uuid.uuid4().hex[:12]
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    session_name = f"taili_{run_id}"

    run_tmp_dir = f"{tmp_dir}/train/{timestamp}"
    log_path = f"{run_tmp_dir}/taili_train_{run_id}.log"
    exit_code_path = f"{run_tmp_dir}/taili_train_{run_id}.exit_code"
    pid_path = f"{run_tmp_dir}/taili_train_{run_id}.pid"
    # 脚本名内嵌 run_id（taili_run_<run_id>），是 remote_kill_training 反查进程树的锚点。
    script_path = f"{run_tmp_dir}/taili_run_{run_id}.ps1"

    launcher = _build_launcher(run_tmp_dir, pid_path, log_path, exit_code_path, command)
    _start_detached(host, port, user, password, run_tmp_dir, script_path, launcher, timeout_seconds)
    return {"session_name": session_name, "log_path": log_path, "exit_code_path": exit_code_path, "run_id": run_id}


_STATUS_TEMPLATE = r"""$sa = 0
if (Test-Path '@@PIDPATH@@') {
  $procId = (Get-Content '@@PIDPATH@@' -Raw).Trim()
  $pidNum = 0
  if ([int]::TryParse($procId, [ref]$pidNum)) {
    if (Get-Process -Id $pidNum -ErrorAction SilentlyContinue) { $sa = 1 }
  }
}
$ec = ''
if (Test-Path '@@EXITPATH@@') { $ec = (Get-Content '@@EXITPATH@@' -Raw).Trim() }
Write-Output "___SA___$sa"
Write-Output "___EC___$ec"
"""


def remote_check_training_status(
    host: str, port: int, user: str, password: str,
    session_name: str, exit_code_path: str, timeout_seconds: int,
) -> dict:
    """检查 Windows 远端训练状态：pid 进程是否存活 + exit_code 文件是否写入。语义对齐 Linux 版。"""
    pid_path = _pid_path_from_exit(exit_code_path)
    script = _fill(_STATUS_TEMPLATE, {"@@PIDPATH@@": _q(pid_path), "@@EXITPATH@@": _q(exit_code_path)})
    out, err, code = _run_ps(host, port, user, password, script, timeout_seconds)
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
            ec_str = line.replace("___EC___", "").strip()
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


_KILL_TEMPLATE = r"""$rid = '@@RUNID@@'
$procs = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -and ($_.CommandLine -like "*taili_run_$rid*" -or $_.CommandLine -like "*taili_play_$rid*") }
foreach ($p in $procs) { taskkill /T /F /PID $p.ProcessId 2>$null | Out-Null }
Write-Output 'KILLED'
"""


def remote_kill_training(host: str, port: int, user: str, password: str, session_name: str, timeout_seconds: int) -> None:
    """终止 Windows 远端训练进程树（按 run_id 匹配 launcher 命令行，taskkill /T 杀全树）。"""
    run_id = session_name.replace("taili_", "", 1) if session_name.startswith("taili_") else session_name
    script = _fill(_KILL_TEMPLATE, {"@@RUNID@@": _q(run_id)})
    try:
        _run_ps(host, port, user, password, script, timeout_seconds)
    except Exception:
        pass


# ============================ 日志增量读取 ============================

_TAIL_TEMPLATE = r"""$p = '@@LOGPATH@@'
if (-not (Test-Path $p)) { Write-Output '___NOFILE___'; exit 0 }
$size = (Get-Item $p).Length
Write-Output "___SIZE___$size"
$offset = @@OFFSET@@
if ($size -gt $offset) {
  $fs = [System.IO.File]::Open($p, 'Open', 'Read', 'ReadWrite')
  try {
    [void]$fs.Seek([long]$offset, 'Begin')
    $sr = New-Object System.IO.StreamReader($fs, [System.Text.Encoding]::UTF8, $true)
    $sr.ReadToEnd()
    $sr.Close()
  } finally { $fs.Close() }
}
"""


def remote_tail_log(host: str, port: int, user: str, password: str, log_path: str, timeout_seconds: int, byte_offset: int = 0) -> tuple[str, int]:
    """从 Windows 远端日志文件按字节偏移增量读取。返回 (new_text, new_offset)。语义对齐 Linux 版。"""
    script = _fill(_TAIL_TEMPLATE, {"@@LOGPATH@@": _q(log_path), "@@OFFSET@@": str(int(byte_offset))})
    out, err, code = _run_ps(host, port, user, password, script, timeout_seconds)
    if code != 0:
        raise TailiCloudToolError(err or out)

    if "___NOFILE___" in out:
        return ("", byte_offset)

    marker = "___SIZE___"
    pos = out.find(marker)
    if pos < 0:
        return ("", byte_offset)
    after = out[pos + len(marker):]
    nl = after.find("\n")
    if nl < 0:
        size_str, new_text = after.strip(), ""
    else:
        size_str, new_text = after[:nl].strip(), after[nl + 1:]
    try:
        new_offset = int(size_str)
    except ValueError:
        new_offset = byte_offset + len(new_text.encode("utf-8", errors="replace"))
    return (new_text, new_offset)


_LOG_CONTAINS_TEMPLATE = r"""if ((Test-Path '@@LOG@@') -and (Select-String -Path '@@LOG@@' -SimpleMatch -Pattern '@@MARKER@@' -Quiet)) { Write-Output '__YES__' } else { Write-Output '__NO__' }
"""


def remote_log_contains(host: str, port: int, user: str, password: str, log_path: str, marker: str, timeout_seconds: int) -> bool:
    script = _fill(_LOG_CONTAINS_TEMPLATE, {"@@LOG@@": _q(log_path), "@@MARKER@@": _q(marker)})
    out, err, code = _run_ps(host, port, user, password, script, timeout_seconds)
    return "__YES__" in out


# ============================ play_eval 视频渲染 ============================

def remote_execute_play_in_tmux(
    host: str, port: int, user: str, password: str,
    session_name: str, play_command: str, tmp_dir: str, timeout_seconds: int,
    poll_interval: int = 10,
) -> tuple[str, int]:
    """在 Windows 远端以独立后台进程执行 play_eval 并轮询等待完成。

    名称沿用 Linux 版（remote_execute_play_in_tmux）以对齐调用方；内部不依赖 tmux，
    用 Start-Process 后台跑 + 轮询 exit_code 文件，结束后读日志并杀进程树。
    Returns: (play_stdout, play_exit_code)
    """
    play_id = uuid.uuid4().hex[:12]
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_tmp_dir = f"{tmp_dir}/play/{timestamp}"
    play_log = f"{run_tmp_dir}/taili_play_{play_id}.log"
    play_ec = f"{run_tmp_dir}/taili_play_{play_id}.exit_code"
    pid_path = f"{run_tmp_dir}/taili_play_{play_id}.pid"
    # 脚本名内嵌 taili_play_<play_id>，供超时/收尾时反查进程树终止。
    script_path = f"{run_tmp_dir}/taili_play_{play_id}.ps1"

    launcher = _build_launcher(run_tmp_dir, pid_path, play_log, play_ec, play_command)
    _start_detached(host, port, user, password, run_tmp_dir, script_path, launcher, timeout_seconds)

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        time.sleep(poll_interval)
        chk = _fill(r"""if (Test-Path '@@EC@@') { (Get-Content '@@EC@@' -Raw).Trim() } else { Write-Output '___NOTDONE___' }
""", {"@@EC@@": _q(play_ec)})
        out, _, _ = _run_ps(host, port, user, password, chk, 30)
        out = out.strip()
        if out and out != "___NOTDONE___":
            try:
                exit_code = int(out)
            except ValueError:
                exit_code = -1
            log_out = _ps_read_file(host, port, user, password, play_log, 60)
            remote_kill_training(host, port, user, password, f"taili_{play_id}", timeout_seconds)
            return (log_out, exit_code)

    remote_kill_training(host, port, user, password, f"taili_{play_id}", timeout_seconds)
    raise TailiCloudToolError(f"play_eval 在 {timeout_seconds}s 内未完成")


# ============================ 远端文件/目录查询 ============================

def _ps_read_file(host: str, port: int, user: str, password: str, path: str, timeout_seconds: int) -> str:
    script = _fill(r"""if (Test-Path '@@P@@') { Get-Content -Raw -Encoding utf8 -Path '@@P@@' }
""", {"@@P@@": _q(path)})
    out, _, code = _run_ps(host, port, user, password, script, timeout_seconds)
    return out if code == 0 else ""


_LIST_RUN_TEMPLATE = r"""$root = '@@ROOT@@'
if (Test-Path $root) {
  Get-ChildItem -Path $root -Directory -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match '^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$' } |
    Sort-Object Name | Select-Object -Last 1 | ForEach-Object { $_.FullName }
}
"""


def remote_list_latest_run(host: str, port: int, user: str, password: str, root: str, timeout_seconds: int) -> str:
    script = _fill(_LIST_RUN_TEMPLATE, {"@@ROOT@@": _q(root)})
    out, err, code = _run_ps(host, port, user, password, script, timeout_seconds)
    if code != 0 or not out.strip():
        return ""
    return _fwd(out.strip()).rstrip("/")


_FIND_CKPT_TEMPLATE = r"""$rd = '@@RD@@'
if (Test-Path $rd) {
  Get-ChildItem -Path $rd -Filter 'model_*.pt' -File -ErrorAction SilentlyContinue |
    ForEach-Object { if ($_.BaseName -match '^model_(\d+)$') { [pscustomobject]@{ N = [int]$Matches[1]; P = $_.FullName } } } |
    Sort-Object N | Select-Object -Last 1 | ForEach-Object { $_.P }
}
"""


def remote_find_latest_checkpoint(host: str, port: int, user: str, password: str, run_dir: str, timeout_seconds: int) -> str:
    """在远端 run_dir 下找迭代号最大的 model_*.pt（兜底用）。找不到返回空串。"""
    script = _fill(_FIND_CKPT_TEMPLATE, {"@@RD@@": _q(run_dir)})
    out, err, code = _run_ps(host, port, user, password, script, timeout_seconds)
    if code != 0 or not out.strip():
        return ""
    return _fwd(out.strip())


def remote_file_exists(host: str, port: int, user: str, password: str, remote_path: str, timeout_seconds: int) -> bool:
    if not remote_path:
        return False
    script = _fill(r"""if (Test-Path -PathType Leaf '@@P@@') { Write-Output '__YES__' } else { Write-Output '__NO__' }
""", {"@@P@@": _q(remote_path)})
    out, err, code = _run_ps(host, port, user, password, script, timeout_seconds)
    return "__YES__" in out


def wait_for_remote_file_stable(host: str, port: int, user: str, password: str, remote_path: str, timeout_seconds: int, polls: int = 3, interval_seconds: int = 2) -> bool:
    last_size: int | None = None
    stable_count = 0
    script = _fill(r"""if (Test-Path '@@P@@') { (Get-Item '@@P@@').Length } else { -1 }
""", {"@@P@@": _q(remote_path)})
    for _ in range(max(1, polls)):
        out, err, code = _run_ps(host, port, user, password, script, timeout_seconds)
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


_LIST_META_TEMPLATE = r"""$base = '@@BASE@@'
if (Test-Path $base) {
  Get-ChildItem -Path $base -Directory -ErrorAction SilentlyContinue | ForEach-Object {
    $m = Join-Path $_.FullName 'eval_meta.json'
    if (Test-Path $m) { Write-Output (((Get-Item $m).LastWriteTime.Ticks).ToString() + '|' + $m) }
  }
}
"""

_LATEST_MP4_TEMPLATE = r"""$f = '@@FOLDER@@'
if (Test-Path $f) {
  Get-ChildItem -Path $f -Filter '*.mp4' -File -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime | Select-Object -Last 1 | ForEach-Object { $_.FullName }
}
"""


def remote_find_play_eval_artifacts(
    host: str,
    port: int,
    user: str,
    password: str,
    run_root: str,
    timeout_seconds: int,
    expected_terrains: list[str] | tuple[str, ...] | None = None,
) -> dict[str, dict]:
    """查找 play_eval.py 生成的一组评估产物（Windows 版，逻辑/返回结构对齐 Linux）。"""
    expected = set(expected_terrains or [])
    play_eval_dir = posixpath.join(run_root, "videos", "play_eval")

    list_script = _fill(_LIST_META_TEMPLATE, {"@@BASE@@": _q(play_eval_dir)})
    out, err, code = _run_ps(host, port, user, password, list_script, timeout_seconds)
    if code != 0 or not out.strip():
        return {}

    # 解析 "<ticks>|<meta_path>" 并按 ticks 数值升序（后出现的同 terrain 覆盖旧项）。
    rows: list[tuple[int, str]] = []
    for line in out.splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        ticks_str, meta_path = line.split("|", 1)
        try:
            ticks = int(ticks_str.strip())
        except ValueError:
            ticks = 0
        rows.append((ticks, _fwd(meta_path.strip())))
    rows.sort(key=lambda r: r[0])

    artifacts: dict[str, dict] = {}
    for _ticks, meta_path in rows:
        meta_out = _ps_read_file(host, port, user, password, meta_path, timeout_seconds)
        if not meta_out.strip():
            continue
        try:
            meta = json.loads(meta_out)
        except json.JSONDecodeError:
            meta = {"raw_meta_text": meta_out}

        terrain = str(meta.get("terrain") or "").strip()
        if not terrain:
            terrain = posixpath.basename(posixpath.dirname(meta_path)).split("_")[0]
        if expected and terrain not in expected:
            continue

        video_folder = posixpath.dirname(meta_path)
        mp4_script = _fill(_LATEST_MP4_TEMPLATE, {"@@FOLDER@@": _q(video_folder)})
        video_out, _, video_code = _run_ps(host, port, user, password, mp4_script, timeout_seconds)
        video_path = _fwd(video_out.strip()) if video_code == 0 else ""
        if not video_path:
            continue

        artifacts[terrain] = {
            "terrain": terrain,
            "video_remote_path": video_path,
            "meta_remote_path": meta_path,
            "video_folder": video_folder,
            "meta": meta,
        }

    return artifacts
