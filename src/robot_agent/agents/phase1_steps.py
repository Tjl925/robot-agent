from __future__ import annotations

"""Phase-1 子步骤 Agent 定义（基于 Google ADK）。

本文件将“AutoDL 开机并 SSH 验证”拆分为 4 个确定性步骤：
1) 开机
2) 等待 running
3) 拉取快照并提取 SSH 连接信息
4) 执行 SSH 探测命令

设计原则：
- 每个步骤只做一件事，失败边界清晰；
- 所有关键结果都写入 `ctx.session.state`，便于编排器恢复执行；
- 每一步都产生日志和可观测事件，便于在 ADK Runner 中追踪。
"""

from typing import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types

from robot_agent.schemas.config import AutoDLConfig
from robot_agent.schemas.state import (
    STATE_P1_EVENTS,
    STATE_P1_INSTANCE_UUID,
    STATE_P1_SSH_COMMAND,
    STATE_P1_SSH_CONNECTED,
    STATE_P1_SSH_HOST,
    STATE_P1_SSH_PASSWORD,
    STATE_P1_SSH_PORT,
    STATE_P1_SSH_USER,
    STATE_P1_STAGE,
    STATE_P1_STATUS,
    Phase1Stage,
)
from robot_agent.tools.autodl_api import AutoDLClient
from robot_agent.tools.ssh_client import parse_proxy_host, test_ssh_connection


class _StepBaseAgent(BaseAgent):
    """步骤 Agent 的公共基类。

    作用：
    - 注入并持有共享配置 `cfg` 与 API 客户端 `client`；
    - 提供统一日志写入工具 `_add_log`；
    - 提供统一文本事件构造 `_yield_text`。

    注意：
    - 这里不做业务逻辑，只做通用能力封装。
    """

    cfg: AutoDLConfig
    client: AutoDLClient
    model_config = {"arbitrary_types_allowed": True}

    def _add_log(self, ctx: InvocationContext, text: str) -> None:
        """将日志追加到会话状态中的事件列表。

        使用列表持久化可保证：
        - Runner 中断后可恢复查看历史；
        - 后续可接入审计系统或可视化面板。
        """

        logs = list(ctx.session.state.get(STATE_P1_EVENTS, []))
        logs.append(text)
        ctx.session.state[STATE_P1_EVENTS] = logs

    def _yield_text(self, text: str) -> Event:
        """构造一个 ADK 文本事件，用于实时反馈步骤进度。"""

        return Event(author=self.name, content=types.Content(role="model", parts=[types.Part(text=text)]))


class PowerOnStepAgent(_StepBaseAgent):
    """步骤1：请求 AutoDL 开机。"""

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        # 读取目标实例 ID，并将当前阶段写入状态
        instance_uuid = str(ctx.session.state[STATE_P1_INSTANCE_UUID])
        ctx.session.state[STATE_P1_STAGE] = Phase1Stage.POWER_ON
        self._add_log(ctx, f"[{self.name}] 发起开机，instance={instance_uuid}")

        # 调用 AutoDL 开机接口
        self.client.power_on_instance(instance_uuid, payload=self.cfg.power_on_payload)

        # 发出进度事件
        yield self._yield_text(f"{self.name}: 已发送开机请求")


class WaitRunningStepAgent(_StepBaseAgent):
    """步骤2：轮询实例状态，直到变为 running。"""

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        instance_uuid = str(ctx.session.state[STATE_P1_INSTANCE_UUID])
        ctx.session.state[STATE_P1_STAGE] = Phase1Stage.WAIT_RUNNING
        self._add_log(ctx, f"[{self.name}] 轮询实例状态，instance={instance_uuid}")

        # 等待运行态，内部包含超时与轮询间隔控制
        status = self.client.wait_until_running(
            instance_uuid=instance_uuid,
            timeout_seconds=self.cfg.boot_timeout_seconds,
            poll_interval_seconds=self.cfg.poll_interval_seconds,
        )

        # 持久化当前状态，供后续步骤/恢复逻辑使用
        ctx.session.state[STATE_P1_STATUS] = status
        yield self._yield_text(f"{self.name}: 实例状态={status}")


class FetchSnapshotStepAgent(_StepBaseAgent):
    """步骤3：获取实例快照并提取 SSH 连接元信息。"""

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        instance_uuid = str(ctx.session.state[STATE_P1_INSTANCE_UUID])
        ctx.session.state[STATE_P1_STAGE] = Phase1Stage.FETCH_SNAPSHOT
        self._add_log(ctx, f"[{self.name}] 获取快照，instance={instance_uuid}")

        # snapshot 中包含 proxy_host、ssh_port、root_password 等字段
        snapshot = self.client.get_instance_snapshot(instance_uuid)
        host, port, user, pwd = parse_proxy_host(snapshot)

        # 将 SSH 连接信息写入会话状态
        ctx.session.state[STATE_P1_SSH_HOST] = host
        ctx.session.state[STATE_P1_SSH_PORT] = port
        ctx.session.state[STATE_P1_SSH_USER] = user
        ctx.session.state[STATE_P1_SSH_PASSWORD] = pwd
        ctx.session.state[STATE_P1_SSH_COMMAND] = snapshot.get("ssh_command")

        yield self._yield_text(f"{self.name}: 已提取 SSH 信息 {user}@{host}:{port}")


class SSHConnectStepAgent(_StepBaseAgent):
    """步骤4：执行 SSH 连接探测并标记连通性。"""

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        ctx.session.state[STATE_P1_STAGE] = Phase1Stage.SSH_CONNECT
        self._add_log(ctx, f"[{self.name}] 开始 SSH 探测")

        # 使用快照提取出的连接参数进行真实 SSH 登录与命令执行
        output = test_ssh_connection(
            host=str(ctx.session.state.get(STATE_P1_SSH_HOST, "")),
            port=int(ctx.session.state.get(STATE_P1_SSH_PORT, 22)),
            username=str(ctx.session.state.get(STATE_P1_SSH_USER, "root")),
            password=str(ctx.session.state.get(STATE_P1_SSH_PASSWORD, "")),
            command=self.cfg.ssh_test_command,
            timeout_seconds=self.cfg.ssh_timeout_seconds,
            strict_host_key_check=self.cfg.strict_host_key_check,
        )

        # 标记成功，并记录探测输出
        ctx.session.state[STATE_P1_SSH_CONNECTED] = True
        self._add_log(ctx, f"[{self.name}] SSH 输出: {output}")
        yield self._yield_text(f"{self.name}: SSH 探测成功")
