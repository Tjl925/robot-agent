from __future__ import annotations

"""Phase-1 编排器（Orchestrator）。

目标：
- 将“开机 -> 等待运行 -> 拉取快照 -> SSH 探测”串成可恢复、可重试的稳定工作流；
- 与 ADK Runner 对接，作为 Phase-1 root agent 执行入口；
- 所有关键状态统一落在 `ctx.session.state`，便于后续由总编排器复用 Phase-1 的 SSH 结果。

核心特性：
1) 确定性步骤顺序：保证执行可预测。
2) 断点恢复：根据 `phase1.stage` 决定从哪一步继续。
3) 重试机制：每一步有最大重试次数 + 退避时间。
4) 明确终态：最终一定收敛到 DONE 或 FAILED。
"""

import asyncio
from typing import AsyncGenerator, List, Tuple
from typing_extensions import override

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types

from robot_agent.agents.phase1_steps import (
    FetchSnapshotStepAgent,
    PowerOnStepAgent,
    SSHConnectStepAgent,
    WaitRunningStepAgent,
)
from robot_agent.schemas.config import AutoDLConfig
from robot_agent.schemas.state import (
    STATE_P1_EVENTS,
    STATE_P1_FAILURE_REASON,
    STATE_P1_INSTANCE_UUID,
    STATE_P1_RETRY_COUNT,
    STATE_P1_SSH_CONNECTED,
    STATE_P1_STAGE,
    Phase1Stage,
)
from robot_agent.tools.autodl_api import AutoDLClient


class Phase1OrchestratorAgent(BaseAgent):
    """Phase-1 的 ADK 自定义编排器。"""

    cfg: AutoDLConfig
    power_on: PowerOnStepAgent
    wait_running: WaitRunningStepAgent
    fetch_snapshot: FetchSnapshotStepAgent
    ssh_connect: SSHConnectStepAgent
    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, cfg: AutoDLConfig):
        client = AutoDLClient(api_base=cfg.api_base, token=cfg.token)

        power_on = PowerOnStepAgent(name="power_on_step", cfg=cfg, client=client)
        wait_running = WaitRunningStepAgent(name="wait_running_step", cfg=cfg, client=client)
        fetch_snapshot = FetchSnapshotStepAgent(name="fetch_snapshot_step", cfg=cfg, client=client)
        ssh_connect = SSHConnectStepAgent(name="ssh_connect_step", cfg=cfg, client=client)

        super().__init__(
            name="phase1_orchestrator",
            cfg=cfg,
            power_on=power_on,
            wait_running=wait_running,
            fetch_snapshot=fetch_snapshot,
            ssh_connect=ssh_connect,
            sub_agents=[power_on, wait_running, fetch_snapshot, ssh_connect],
        )

    def _ordered_steps(self) -> List[Tuple[Phase1Stage, BaseAgent]]:
        return [
            (Phase1Stage.POWER_ON, self.power_on),
            (Phase1Stage.WAIT_RUNNING, self.wait_running),
            (Phase1Stage.FETCH_SNAPSHOT, self.fetch_snapshot),
            (Phase1Stage.SSH_CONNECT, self.ssh_connect),
        ]

    def _start_index_from_state(self, current_stage: str) -> int:
        order = [s for s, _ in self._ordered_steps()]
        if current_stage in (Phase1Stage.DONE, Phase1Stage.FAILED):
            return len(order)
        if current_stage in order:
            return order.index(current_stage)
        return 0

    def _yield_text(self, text: str) -> Event:
        return Event(author=self.name, content=types.Content(role="model", parts=[types.Part(text=text)]))

    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        if STATE_P1_EVENTS not in ctx.session.state:
            ctx.session.state[STATE_P1_EVENTS] = []
        if STATE_P1_RETRY_COUNT not in ctx.session.state:
            ctx.session.state[STATE_P1_RETRY_COUNT] = 0
        if STATE_P1_STAGE not in ctx.session.state:
            ctx.session.state[STATE_P1_STAGE] = Phase1Stage.INIT
        if STATE_P1_INSTANCE_UUID not in ctx.session.state:
            ctx.session.state[STATE_P1_INSTANCE_UUID] = self.cfg.instance_uuid

        current_stage = str(ctx.session.state.get(STATE_P1_STAGE, Phase1Stage.INIT))
        start_idx = self._start_index_from_state(current_stage)
        if current_stage == Phase1Stage.DONE:
            yield self._yield_text("phase1 已完成，跳过执行")
            return

        for _stage, step_agent in self._ordered_steps()[start_idx:]:
            attempts = 0
            while True:
                try:
                    async for event in step_agent.run_async(ctx):
                        yield event
                    break
                except Exception as exc:  # noqa: BLE001
                    attempts += 1
                    ctx.session.state[STATE_P1_RETRY_COUNT] = int(ctx.session.state.get(STATE_P1_RETRY_COUNT, 0)) + 1
                    ctx.session.state[STATE_P1_FAILURE_REASON] = str(exc)
                    if attempts > self.cfg.max_retries_per_step:
                        ctx.session.state[STATE_P1_STAGE] = Phase1Stage.FAILED
                        yield self._yield_text(f"{step_agent.name} 重试耗尽，失败: {exc}")
                        return
                    yield self._yield_text(f"{step_agent.name} 执行失败，第{attempts}次重试: {exc}")
                    await asyncio.sleep(self.cfg.retry_backoff_seconds)

        if bool(ctx.session.state.get(STATE_P1_SSH_CONNECTED, False)):
            ctx.session.state[STATE_P1_STAGE] = Phase1Stage.DONE
            yield self._yield_text("Phase-1 完成：实例已开机且 SSH 可连通")
        else:
            ctx.session.state[STATE_P1_STAGE] = Phase1Stage.FAILED
            ctx.session.state[STATE_P1_FAILURE_REASON] = "步骤执行后 SSH 仍未连通"
            yield self._yield_text("Phase-1 失败：SSH 未连通")

        await ctx.session_service.append_event(ctx.session, Event(author=self.name, actions=EventActions(state_delta=ctx.session.state)))
