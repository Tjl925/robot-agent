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
from typing import Any, AsyncGenerator, List, Tuple
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
    STATE_P1_SSH_HOST,
    STATE_P1_SSH_PORT,
    STATE_P1_SSH_USER,
    STATE_P1_SSH_PASSWORD,
    STATE_P1_SSH_COMMAND,
    STATE_P1_STAGE,
    STATE_P1_USE_BACKUP,
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
    remote_platform: str = "linux"
    windows_cfg: Any = None
    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, cfg: AutoDLConfig, remote_platform: str = "linux", windows_cfg: Any = None):
        client = AutoDLClient(api_base=cfg.api_base, token=cfg.token)

        power_on = PowerOnStepAgent(name="power_on_step", cfg=cfg, client=client)
        wait_running = WaitRunningStepAgent(name="wait_running_step", cfg=cfg, client=client)
        fetch_snapshot = FetchSnapshotStepAgent(name="fetch_snapshot_step", cfg=cfg, client=client)
        ssh_connect = SSHConnectStepAgent(name="ssh_connect_step", cfg=cfg, client=client)

        super().__init__(
            name="phase1_orchestrator",
            cfg=cfg,
            remote_platform=remote_platform,
            windows_cfg=windows_cfg,
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
        if STATE_P1_USE_BACKUP not in ctx.session.state:
            ctx.session.state[STATE_P1_USE_BACKUP] = False

        current_stage = str(ctx.session.state.get(STATE_P1_STAGE, Phase1Stage.INIT))
        if current_stage == Phase1Stage.DONE:
            yield self._yield_text("phase1 已完成，跳过执行")
            return

        # 0. Windows 主机直连模式：不走 AutoDL 开机，也不走 Linux 备用服务器，
        #    直接把 SSH 连接信息指向 Windows 主机并只做 SSH 探活（带重试）。
        if self.remote_platform == "windows" and self.windows_cfg is not None:
            win = self.windows_cfg
            ctx.session.state[STATE_P1_SSH_HOST] = win.ssh_host
            ctx.session.state[STATE_P1_SSH_PORT] = win.ssh_port
            ctx.session.state[STATE_P1_SSH_USER] = win.ssh_user
            ctx.session.state[STATE_P1_SSH_PASSWORD] = win.ssh_password
            ctx.session.state[STATE_P1_SSH_COMMAND] = f"ssh {win.ssh_user}@{win.ssh_host} -p {win.ssh_port}"
            ctx.session.state[STATE_P1_SSH_CONNECTED] = False
            ctx.session.state[STATE_P1_STAGE] = Phase1Stage.SSH_CONNECT
            yield self._yield_text(
                f"已启用 Windows 主机直连模式（{win.ssh_user}@{win.ssh_host}:{win.ssh_port}），"
                f"跳过 AutoDL 开机，直接进行 SSH 探活..."
            )
            attempts = 0
            while True:
                try:
                    async for event in self.ssh_connect.run_async(ctx):
                        yield event
                    break
                except Exception as exc:  # noqa: BLE001
                    attempts += 1
                    ctx.session.state[STATE_P1_RETRY_COUNT] = int(ctx.session.state.get(STATE_P1_RETRY_COUNT, 0)) + 1
                    ctx.session.state[STATE_P1_FAILURE_REASON] = str(exc)
                    if attempts > self.cfg.max_retries_per_step:
                        ctx.session.state[STATE_P1_STAGE] = Phase1Stage.FAILED
                        yield self._yield_text(f"Windows 主机 SSH 探测重试耗尽，失败: {exc}")
                        await ctx.session_service.append_event(ctx.session, Event(author=self.name, actions=EventActions(state_delta=ctx.session.state)))
                        return
                    yield self._yield_text(f"Windows 主机 SSH 探测失败，第{attempts}次重试: {exc}")
                    await asyncio.sleep(self.cfg.retry_backoff_seconds)

            if bool(ctx.session.state.get(STATE_P1_SSH_CONNECTED, False)):
                ctx.session.state[STATE_P1_STAGE] = Phase1Stage.DONE
                yield self._yield_text("Phase-1 完成：Windows 主机 SSH 已连通")
            else:
                ctx.session.state[STATE_P1_STAGE] = Phase1Stage.FAILED
                ctx.session.state[STATE_P1_FAILURE_REASON] = "Windows 主机 SSH 未连通"
                yield self._yield_text("Phase-1 失败：Windows 主机 SSH 未连通")
            await ctx.session_service.append_event(ctx.session, Event(author=self.name, actions=EventActions(state_delta=ctx.session.state)))
            return

        use_backup = bool(ctx.session.state.get(STATE_P1_USE_BACKUP, False))

        # 1. 正常/备用 阶段步骤准备
        if use_backup:
            steps_to_run = [(Phase1Stage.SSH_CONNECT, self.ssh_connect)]
            yield self._yield_text("已启用备用服务器模式，将跳过 AutoDL 开机流程，直接对备用服务器进行 SSH 探活...")
        else:
            start_idx = self._start_index_from_state(current_stage)
            steps_to_run = self._ordered_steps()[start_idx:]

        autodl_failed = False

        # 2. 执行待运行的步骤
        try:
            for _stage, step_agent in steps_to_run:
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
                            yield self._yield_text(f"{step_agent.name} 重试耗尽，失败: {exc}")
                            if not use_backup:
                                autodl_failed = True
                                break
                            else:
                                ctx.session.state[STATE_P1_STAGE] = Phase1Stage.FAILED
                                return
                        yield self._yield_text(f"{step_agent.name} 执行失败，第{attempts}次重试: {exc}")
                        await asyncio.sleep(self.cfg.retry_backoff_seconds)

                if autodl_failed:
                    break
        except Exception as exc:
            if not use_backup:
                autodl_failed = True
                ctx.session.state[STATE_P1_FAILURE_REASON] = str(exc)
                yield self._yield_text(f"AutoDL 执行过程抛出异常: {exc}")
            else:
                ctx.session.state[STATE_P1_STAGE] = Phase1Stage.FAILED
                ctx.session.state[STATE_P1_FAILURE_REASON] = str(exc)
                yield self._yield_text(f"备用服务器执行失败: {exc}")
                return

        # 3. 检查是否需要切换并运行备用服务器
        if not use_backup and (autodl_failed or not bool(ctx.session.state.get(STATE_P1_SSH_CONNECTED, False))):
            yield self._yield_text("AutoDL 自动开机或 SSH 探测失败。开始尝试备用服务器...")

            ctx.session.state[STATE_P1_USE_BACKUP] = True
            ctx.session.state[STATE_P1_SSH_HOST] = self.cfg.backup_ssh_host
            ctx.session.state[STATE_P1_SSH_PORT] = self.cfg.backup_ssh_port
            ctx.session.state[STATE_P1_SSH_USER] = self.cfg.backup_ssh_user
            ctx.session.state[STATE_P1_SSH_PASSWORD] = self.cfg.backup_ssh_password
            ctx.session.state[STATE_P1_SSH_COMMAND] = f"ssh {self.cfg.backup_ssh_user}@{self.cfg.backup_ssh_host} -p {self.cfg.backup_ssh_port}"
            ctx.session.state[STATE_P1_SSH_CONNECTED] = False
            ctx.session.state[STATE_P1_STAGE] = Phase1Stage.SSH_CONNECT

            attempts = 0
            while True:
                try:
                    async for event in self.ssh_connect.run_async(ctx):
                        yield event
                    break
                except Exception as exc:
                    attempts += 1
                    ctx.session.state[STATE_P1_RETRY_COUNT] = int(ctx.session.state.get(STATE_P1_RETRY_COUNT, 0)) + 1
                    ctx.session.state[STATE_P1_FAILURE_REASON] = str(exc)
                    if attempts > self.cfg.max_retries_per_step:
                        ctx.session.state[STATE_P1_STAGE] = Phase1Stage.FAILED
                        yield self._yield_text(f"备用服务器 SSH 探测重试耗尽，失败: {exc}")
                        return
                    yield self._yield_text(f"备用服务器 SSH 探测失败，第{attempts}次重试: {exc}")
                    await asyncio.sleep(self.cfg.retry_backoff_seconds)

        # 4. 判定最终成果并收尾
        if bool(ctx.session.state.get(STATE_P1_SSH_CONNECTED, False)):
            ctx.session.state[STATE_P1_STAGE] = Phase1Stage.DONE
            yield self._yield_text("Phase-1 完成：已成功开机且 SSH 可连通")
        else:
            ctx.session.state[STATE_P1_STAGE] = Phase1Stage.FAILED
            ctx.session.state[STATE_P1_FAILURE_REASON] = "所有可用服务器的 SSH 均未连通"
            yield self._yield_text("Phase-1 失败：SSH 未连通")

        await ctx.session_service.append_event(ctx.session, Event(author=self.name, actions=EventActions(state_delta=ctx.session.state)))
