from __future__ import annotations

"""SSH 工具层：用于探测实例连通性与执行远端命令。

设计原则：
- 仅负责 SSH 连接与命令执行，不依赖 ADK；
- Phase-1 与 Phase-2 可复用同一工具；
- 明确异常类型，便于上层做重试与降级策略。
"""

import socket
from typing import Tuple

import paramiko


class SSHConnectError(RuntimeError):
    """SSH 连接或命令执行失败时抛出。"""


def parse_proxy_host(snapshot: dict) -> Tuple[str, int, str, str]:
    """从 AutoDL 快照中提取 SSH 连接信息。"""

    host = snapshot.get("proxy_host")
    port = snapshot.get("ssh_port")
    password = snapshot.get("root_password")

    if not host or not port or not password:
        raise SSHConnectError("Snapshot missing proxy_host/ssh_port/root_password")

    return str(host), int(port), "root", str(password)


def execute_ssh_command(
    host: str,
    port: int,
    username: str,
    password: str,
    command: str,
    timeout_seconds: int,
    strict_host_key_check: bool = False,
) -> tuple[str, str, int]:
    """执行远端命令并返回标准输出、标准错误与退出码。

    返回值：
    - stdout_text
    - stderr_text
    - exit_status（远端命令退出码）

    说明：
    - 与 `test_ssh_connection` 的区别在于这里会返回退出码，适合训练任务场景。
    """

    client = paramiko.SSHClient()
    if strict_host_key_check:
        client.load_system_host_keys()
    else:
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(
            hostname=host,
            port=port,
            username=username,
            password=password,
            timeout=timeout_seconds,
            banner_timeout=timeout_seconds,
            auth_timeout=timeout_seconds,
        )
        _, stdout, stderr = client.exec_command(command, timeout=timeout_seconds)

        stdout_text = stdout.read().decode("utf-8", errors="replace").strip()
        stderr_text = stderr.read().decode("utf-8", errors="replace").strip()
        exit_status = int(stdout.channel.recv_exit_status())
        return stdout_text, stderr_text, exit_status
    except (paramiko.SSHException, socket.error) as exc:
        raise SSHConnectError(f"SSH failed: {exc}") from exc
    finally:
        client.close()


def test_ssh_connection(
    host: str,
    port: int,
    username: str,
    password: str,
    command: str,
    timeout_seconds: int,
    strict_host_key_check: bool = False,
) -> str:
    """执行 SSH 探测命令。

    用于 phase1 的连通性校验，保持原有返回接口兼容。
    """

    stdout_text, stderr_text, _exit_status = execute_ssh_command(
        host=host,
        port=port,
        username=username,
        password=password,
        command=command,
        timeout_seconds=timeout_seconds,
        strict_host_key_check=strict_host_key_check,
    )

    if stderr_text:
        return f"{stdout_text}\n{stderr_text}".strip()
    return stdout_text
