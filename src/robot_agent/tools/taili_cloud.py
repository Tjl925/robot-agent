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

import paramiko

from robot_agent.tools.ssh_client import execute_ssh_command


class TailiCloudToolError(RuntimeError):
    """Taili 云端工具失败时抛出。"""


def sync_local_tree_to_cloud(local_root: str, cloud_root: str, files: list[tuple[str, str]]) -> list[dict[str, str]]:
    """把本地生成文件复制到云端镜像目录（本地模拟同步结果）。

    注意：当 cloud_root 是远端路径（如 /root/robot_lab）且本地不存在时，
    此函数会跳过复制并返回空列表。真正的文件同步由 SFTP 上传完成。
    """

    cloud_root_path = Path(cloud_root)
    if not cloud_root_path.is_absolute() or not cloud_root_path.anchor:
        return []
    # 远端路径（如 /root/...）在 Windows 上无法映射，安全跳过。
    try:
        cloud_root_path.mkdir(parents=True, exist_ok=True)
    except OSError:
        return []

    copied: list[dict[str, str]] = []
    for src_rel, dst_rel in files:
        src = Path(local_root) / src_rel
        dst = cloud_root_path / dst_rel
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied.append({"src": str(src), "dst": str(dst), "status": "copied"})
        except OSError as exc:
            copied.append({"src": str(src), "dst": str(dst), "status": f"skipped: {exc}"})
    return copied


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


def remote_upload_taili_workspace(host: str, port: int, user: str, password: str, local_root: str, local_robots_subdir: str, cloud_asset_path: str, cloud_task_init_path: str, cloud_task_cfg_root: str, timeout_seconds: int) -> list[dict[str, str]]:
    """把 taili 本地 workspace 关键产物上传到远端固定 robot_lab 路径。"""

    files = [
        (str(Path(local_root) / local_robots_subdir / "robot.urdf"), "source/robot_lab/robot_lab/data/Robots/robot.urdf"),
        (str(Path(local_root) / ".taili_generated" / "taili_quad.py"), cloud_asset_path),
        (str(Path(local_root) / ".taili_generated" / "__init__.py"), cloud_task_init_path),
        (str(Path(local_root) / ".taili_generated" / "rough_env_cfg.py"), posixpath.join(cloud_task_cfg_root, "rough_env_cfg.py")),
    ]
    return upload_files_via_sftp(host, port, user, password, files, timeout_seconds)


def render_taili_asset_py(local_robot_root: str, task_name: str) -> str:
    """生成 Taili 资产 Python 草案。

    注意 (P2-5)：生成的代码中 `from isaaclab_assets import ArticulationCfg` 等
    import 路径为占位模板，需确保与实际 robot_lab 环境的 API 一致。
    如果 robot_lab 的 API 发生变更，此模板也需同步更新。
    """

    urdf_path = Path(local_robot_root) / "urdf" / "robot.urdf"
    return f'''from __future__ import annotations

"""Taili Quad 机器人资产定义。"""

from isaaclab_assets import ArticulationCfg
from isaaclab_assets.robots import URDF

TAILI_QUAD_CFG = ArticulationCfg(
    spawn=URDF(
        usd_path=r"{urdf_path.as_posix()}",
    ),
    soft_joint_pos_limit_factor=0.9,
)
'''


def render_taili_task_init_py(task_name: str) -> str:
    """生成 Taili 任务注册初始化文件草案。"""

    return f'''from __future__ import annotations

"""Taili Quad 速度控制任务注册。"""

import gymnasium as gym

from . import agents

gym.register(
    id="{task_name}",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={{
        "env_cfg_entry_point": f"{{__name__}}.rough_env_cfg:TailiQuadRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{{agents.__name__}}.rsl_rl_ppo_cfg:TailiQuadRoughPPORunnerCfg",
    }},
)
'''


def render_taili_task_cfg_py(task_name: str, reward: dict, hyperparams: dict) -> str:
    """生成 Taili 任务配置草案。"""

    reward_lines = "\n".join(f"        {k!r}: {v!r}," for k, v in reward.items())
    hyper_lines = "\n".join(f"    {k!r}: {v!r}," for k, v in hyperparams.items())
    return f'''from __future__ import annotations

"""Taili Quad Rough 环境配置。"""

from isaaclab.utils import configclass


@configclass
class TailiQuadRoughEnvCfg:
    """Taili Quad rough 环境占位配置。"""

    def __post_init__(self):
        self.task_name = {task_name!r}
        self.rewards = {{
{reward_lines}
        }}
        self.hyperparams = {{
{hyper_lines}
        }}
'''


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


def remote_find_latest_matching_file(host: str, port: int, user: str, password: str, root: str, glob_pattern: str, timeout_seconds: int) -> str:
    cmd = (
        "python - <<'PY'\n"
        "from pathlib import Path\n"
        f"root = Path(r'''{root}''')\n"
        f"matches = sorted(root.rglob(r'''{glob_pattern}'''), key=lambda p: p.stat().st_mtime)\n"
        "print(matches[-1] if matches else '')\n"
        "PY"
    )
    out, err, code = execute_ssh_command(host, port, user, password, cmd, timeout_seconds)
    if code != 0:
        raise TailiCloudToolError(err or out)
    return out.strip()


def start_remote_training(host: str, port: int, user: str, password: str, command: str, timeout_seconds: int) -> dict[str, str]:
    wrapped = f"nohup bash -lc {shlex.quote(command)} > /tmp/taili_train.log 2>&1 < /dev/null & echo $!"
    out, err, code = execute_ssh_command(host, port, user, password, wrapped, timeout_seconds)
    if code != 0:
        raise TailiCloudToolError(err or out)
    return {"pid": out.strip(), "log_path": "/tmp/taili_train.log"}


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


def remote_tail_log(host: str, port: int, user: str, password: str, log_path: str, timeout_seconds: int) -> str:
    cmd = f"python - <<'PY'\nfrom pathlib import Path\np = Path(r'''{log_path}''')\nprint(p.read_text(encoding='utf-8', errors='replace') if p.exists() else '')\nPY"
    out, err, code = execute_ssh_command(host, port, user, password, cmd, timeout_seconds)
    if code != 0:
        raise TailiCloudToolError(err or out)
    return out


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


def remote_tensorboard_summary(host: str, port: int, user: str, password: str, run_root: str, tb_dir_name: str, timeout_seconds: int) -> str:
    cmd = (
        "python - <<'PY'\n"
        "from pathlib import Path\n"
        f"root = Path(r'''{run_root}''')\n"
        f"tb = root / '{tb_dir_name}'\n"
        "print({'exists': tb.exists(), 'path': str(tb)})\n"
        "PY"
    )
    out, err, code = execute_ssh_command(host, port, user, password, cmd, timeout_seconds)
    if code != 0:
        raise TailiCloudToolError(err or out)
    return out.strip()
