from __future__ import annotations

"""总编排器（Top-level Orchestrator）。

这个文件把 Phase-1 和 Phase-2 串成一个完整闭环：
1. 先跑 Phase-1，拿到 AutoDL 开机结果和 SSH 信息；
2. 再把 Phase-1 的 SSH 结果直接交给 Phase-2；
3. Phase-2 完成 Taili 配置生成、云端同步、训练、评估与迭代；
4. 评估通过则归档，失败则进入 revise / HITL。

整个项目真正的入口编排器。
"""

from typing import Any, AsyncGenerator
from typing_extensions import override

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types

from robot_agent.agents.phase1_orchestrator import Phase1OrchestratorAgent
from robot_agent.agents.taili_orchestrator import TailiOrchestratorAgent
from robot_agent.schemas.config import AutoDLConfig, TailiCloudConfig
from robot_agent.schemas.state import (
    STATE_P1_SSH_COMMAND,
    STATE_P1_SSH_CONNECTED,
    STATE_P1_SSH_HOST,
    STATE_P1_SSH_PASSWORD,
    STATE_P1_SSH_PORT,
    STATE_P1_SSH_USER,
    STATE_P1_STAGE,
    STATE_P1_STATUS,
    STATE_P1_USE_BACKUP,
    Phase1Stage,
    STATE_P2_FAILURE_REASON,
    STATE_P2_HITL_REASON,
    STATE_P2_HITL_REQUIRED,
    STATE_P2_STAGE,
    STATE_P2_STATUS,
    Phase2Stage,
)


class OrchestratorAgent(BaseAgent):
    """整合 Phase-1 和 Phase-2 的总编排器。"""

    phase1: Phase1OrchestratorAgent
    phase2: TailiOrchestratorAgent
    auto_cfg: AutoDLConfig
    taili_cfg: TailiCloudConfig
    remote_platform: str = "linux"
    windows_cfg: Any = None
    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, auto_cfg: AutoDLConfig, taili_cfg: TailiCloudConfig, remote_platform: str = "linux", windows_cfg: Any = None):
        phase1 = Phase1OrchestratorAgent(cfg=auto_cfg, remote_platform=remote_platform, windows_cfg=windows_cfg)
        phase2 = TailiOrchestratorAgent(cfg=taili_cfg)
        super().__init__(
            name="orchestrator",
            auto_cfg=auto_cfg,
            taili_cfg=taili_cfg,
            remote_platform=remote_platform,
            windows_cfg=windows_cfg,
            phase1=phase1,
            phase2=phase2,
            sub_agents=[phase1, phase2],
        )

    def _yield_text(self, text: str) -> Event:
        return Event(author=self.name, content=types.Content(role="model", parts=[types.Part(text=text)]))

    def _handoff_phase1_to_phase2(self, ctx: InvocationContext) -> None:
        """把 Phase-1 的 SSH 结果写入 Phase-2 可直接消费的状态。"""

        self.phase2.cfg.remote_host = str(ctx.session.state.get(STATE_P1_SSH_HOST, self.phase2.cfg.remote_host or ""))
        self.phase2.cfg.remote_port = int(ctx.session.state.get(STATE_P1_SSH_PORT, self.phase2.cfg.remote_port or 22))
        self.phase2.cfg.remote_user = str(ctx.session.state.get(STATE_P1_SSH_USER, self.phase2.cfg.remote_user or "root"))
        self.phase2.cfg.remote_password = str(ctx.session.state.get(STATE_P1_SSH_PASSWORD, self.phase2.cfg.remote_password or ""))

        # Windows 直连：切换 Phase-2 到 pwsh 后端，并用 Windows 主机的远程地址覆盖全部云端路径与命令模板。
        if self.remote_platform == "windows" and self.windows_cfg is not None:
            win = self.windows_cfg
            self.phase2.cfg.remote_platform = "windows"
            self.phase2.cfg.cloud_robot_lab_root = win.robot_lab_root
            self.phase2.cfg.cloud_tmp_dir = win.tmp_dir
            self.phase2.cfg.cloud_asset_path = win.asset_path
            self.phase2.cfg.cloud_task_cfg_root = win.task_cfg_root
            self.phase2.cfg.eval_log_root = win.eval_log_root
            self.phase2.cfg.train_command_template = win.train_command_template
            self.phase2.cfg.play_eval_command_template = win.play_eval_command_template
            return

        # 如果使用了备用服务器，将前缀 /root/autodl-tmp/ 自动替换为 /root/gpufree-data/
        use_backup = bool(ctx.session.state.get(STATE_P1_USE_BACKUP, False))
        if use_backup:
            old_prefix = "/root/autodl-tmp/"
            new_prefix = "/root/gpufree-data/"

            # 替换 cloud_robot_lab_root
            if self.phase2.cfg.cloud_robot_lab_root and old_prefix in self.phase2.cfg.cloud_robot_lab_root:
                self.phase2.cfg.cloud_robot_lab_root = self.phase2.cfg.cloud_robot_lab_root.replace(old_prefix, new_prefix)

            # 替换 cloud_tmp_dir
            if self.phase2.cfg.cloud_tmp_dir and old_prefix in self.phase2.cfg.cloud_tmp_dir:
                self.phase2.cfg.cloud_tmp_dir = self.phase2.cfg.cloud_tmp_dir.replace(old_prefix, new_prefix)

            # 替换 eval_log_root
            if self.phase2.cfg.eval_log_root and old_prefix in self.phase2.cfg.eval_log_root:
                self.phase2.cfg.eval_log_root = self.phase2.cfg.eval_log_root.replace(old_prefix, new_prefix)

            # 替换 train_command_template
            if self.phase2.cfg.train_command_template and old_prefix in self.phase2.cfg.train_command_template:
                self.phase2.cfg.train_command_template = self.phase2.cfg.train_command_template.replace(old_prefix, new_prefix)

            # 替换 play_eval_command_template
            if self.phase2.cfg.play_eval_command_template and old_prefix in self.phase2.cfg.play_eval_command_template:
                self.phase2.cfg.play_eval_command_template = self.phase2.cfg.play_eval_command_template.replace(old_prefix, new_prefix)

    def _phase2_is_waiting_human(self, ctx: InvocationContext) -> bool:
        return (
            ctx.session.state.get(STATE_P2_STAGE) == Phase2Stage.WAIT_HUMAN
            or bool(ctx.session.state.get(STATE_P2_HITL_REQUIRED, False))
        )

    async def _run_phase(self, ctx: InvocationContext, agent: BaseAgent) -> AsyncGenerator[Event, None]:
        async for event in agent.run_async(ctx):
            yield event

    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        if STATE_P1_STAGE not in ctx.session.state:
            ctx.session.state[STATE_P1_STAGE] = Phase1Stage.INIT
        if STATE_P1_STATUS not in ctx.session.state:
            ctx.session.state[STATE_P1_STATUS] = "pending"

        try:
            # Windows 直连不开机，阶段由 phase1 自行推进到 SSH_CONNECT；Linux 才从 POWER_ON 起步。
            if self.remote_platform != "windows":
                ctx.session.state[STATE_P1_STAGE] = Phase1Stage.POWER_ON
            async for event in self._run_phase(ctx, self.phase1):
                yield event

            if not bool(ctx.session.state.get(STATE_P1_SSH_CONNECTED, False)):
                await ctx.session_service.append_event(ctx.session, Event(author=self.name, actions=EventActions(state_delta=dict(ctx.session.state))))
                yield self._yield_text("orchestrator: Phase-1 失败，终止整个流程")
                return

            self._handoff_phase1_to_phase2(ctx)

            async for event in self._run_phase(ctx, self.phase2):
                yield event

            if bool(ctx.session.state.get(STATE_P2_STATUS) == "succeeded"):
                ctx.session.state[STATE_P2_HITL_REQUIRED] = False
                ctx.session.state[STATE_P2_HITL_REASON] = ""
                await ctx.session_service.append_event(ctx.session, Event(author=self.name, actions=EventActions(state_delta=dict(ctx.session.state))))
                yield self._yield_text("orchestrator: Phase-1 + Phase-2 全流程完成")
                return

            if self._phase2_is_waiting_human(ctx):
                ctx.session.state[STATE_P2_STAGE] = Phase2Stage.WAIT_HUMAN
                ctx.session.state[STATE_P2_STATUS] = "pending"
                ctx.session.state.setdefault(STATE_P2_FAILURE_REASON, ctx.session.state.get(STATE_P2_HITL_REASON, "Phase-2 进入 HITL 待人工介入"))
                await ctx.session_service.append_event(ctx.session, Event(author=self.name, actions=EventActions(state_delta=dict(ctx.session.state))))
                yield self._yield_text("orchestrator: Phase-2 已进入 HITL，等待人工介入")
                return

            ctx.session.state[STATE_P2_STAGE] = Phase2Stage.FAILED
            ctx.session.state[STATE_P2_STATUS] = "failed"
            ctx.session.state.setdefault(STATE_P2_FAILURE_REASON, "Phase-2 未收敛到成功态")
            await ctx.session_service.append_event(ctx.session, Event(author=self.name, actions=EventActions(state_delta=dict(ctx.session.state))))
            yield self._yield_text("orchestrator: Phase-2 失败，流程结束")

        except Exception as exc:  # noqa: BLE001
            ctx.session.state[STATE_P2_STAGE] = Phase2Stage.FAILED
            ctx.session.state[STATE_P2_STATUS] = "failed"
            ctx.session.state[STATE_P2_FAILURE_REASON] = str(exc)
            await ctx.session_service.append_event(ctx.session, Event(author=self.name, actions=EventActions(state_delta=dict(ctx.session.state))))
            yield self._yield_text(f"orchestrator: 流程失败：{exc}")
