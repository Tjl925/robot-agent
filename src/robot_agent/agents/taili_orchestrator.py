from __future__ import annotations

"""taili_quad 专用编排器。

这个编排器只服务于一条固定主线：
- 本地 `taili_quad/`
- 云端 `robot_lab/`

职责：
1. 串联所有 Taili 子 Agent 的执行顺序；
2. 维护 Phase-2 的状态机；
3. 在每个步骤结束后把 session state 持久化到事件流；
4. 在失败时控制自动迭代与人工介入。

理解这个文件时可以按“流程总控”来读：
- 它本身不负责生成配置，也不负责训练；
- 它只负责决定“下一步谁执行、什么时候回退、什么时候结束”。
"""

from pathlib import Path
from typing import AsyncGenerator

from typing_extensions import override

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types

from robot_agent.agents.taili_steps import (
    AnalyzeTailiUrdfStepAgent,
    ArchiveTailiOutputsStepAgent,
    EvaluateTailiTrainingLogAgent,
    EvaluateTailiVideoAgent,
    GenerateTailiFilesStepAgent,
    PublishTailiWorkspaceStepAgent,
    RepairTailiWorkflowStepAgent,
    TailiConfigSynthesisAgent,
    TrainTailiStepAgent,
)
from robot_agent.schemas.config import TailiCloudConfig
from robot_agent.schemas.state import (
    STATE_P2_ARCHIVE_COMPLETED,
    STATE_P2_CONFIG_HISTORY,
    STATE_P2_CONFIG_MODE,
    STATE_P2_CONFIG_PARENT_VERSION,
    STATE_P2_CONFIG_VERSION,
    STATE_P2_EVAL_PASSED,
    STATE_P2_EVENTS,
    STATE_P2_FAILURE_REASON,
    STATE_P2_HITL_REASON,
    STATE_P2_HITL_REQUIRED,
    STATE_P2_ITER_MAX,
    STATE_P2_EVAL_VIDEO_PATH,
    STATE_P2_ITER_ROUND,
    STATE_P2_STAGE,
    STATE_P2_STATUS,
    STATE_P2_URDF_VALID,
    STATE_P2_URDF_RISK,
    Phase2Stage,
)


class TailiOrchestratorAgent(BaseAgent):
    """Taili Phase-2 总编排器。

    字段说明：
    - `cfg`：整套 Taili 流程的运行配置；
    - 其余字段：各子 Agent 的实例引用，便于统一编排。
    """

    cfg: TailiCloudConfig
    analyze_urdf: AnalyzeTailiUrdfStepAgent
    config_synthesis: TailiConfigSynthesisAgent
    generate_files: GenerateTailiFilesStepAgent
    publish_cloud: PublishTailiWorkspaceStepAgent
    train: TrainTailiStepAgent
    evaluate_train_log: EvaluateTailiTrainingLogAgent
    evaluate_video: EvaluateTailiVideoAgent
    repair: RepairTailiWorkflowStepAgent
    archive_outputs: ArchiveTailiOutputsStepAgent
    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, cfg: TailiCloudConfig):
        # 子 Agent 的构造集中在编排器里，避免外部调用时需要手动组装整条链路。
        analyze_urdf = AnalyzeTailiUrdfStepAgent(name="taili_analyze_urdf", cfg=cfg)
        config_synthesis = TailiConfigSynthesisAgent(name="taili_config_synthesis", cfg=cfg)
        generate_files = GenerateTailiFilesStepAgent(name="taili_generate_files", cfg=cfg)
        publish_cloud = PublishTailiWorkspaceStepAgent(name="taili_publish_workspace", cfg=cfg)
        evaluate_train_log = EvaluateTailiTrainingLogAgent(name="taili_evaluate_train_log", cfg=cfg)
        train = TrainTailiStepAgent(name="taili_train", cfg=cfg, evaluate_training_log=evaluate_train_log)
        evaluate_video = EvaluateTailiVideoAgent(name="taili_evaluate_video", cfg=cfg)
        repair = RepairTailiWorkflowStepAgent(name="taili_repair", cfg=cfg)
        archive_outputs = ArchiveTailiOutputsStepAgent(name="taili_archive_outputs", cfg=cfg)
        super().__init__(
            name="taili_orchestrator",
            cfg=cfg,
            analyze_urdf=analyze_urdf,
            config_synthesis=config_synthesis,
            generate_files=generate_files,
            publish_cloud=publish_cloud,
            train=train,
            evaluate_train_log=evaluate_train_log,
            evaluate_video=evaluate_video,
            repair=repair,
            archive_outputs=archive_outputs,
            sub_agents=[
                analyze_urdf,
                config_synthesis,
                generate_files,
                publish_cloud,
                train,
                evaluate_train_log,
                evaluate_video,
                repair,
                archive_outputs,
            ],
        )

    def _yield_text(self, text: str) -> Event:
        # 编排器自己的文本输出，主要用于告诉外部“流程走到哪一步了”。
        return Event(author=self.name, content=types.Content(role="model", parts=[types.Part(text=text)]))

    async def _commit_state(self, ctx: InvocationContext) -> None:
        # 关键：把当前 session.state 作为事件 delta 持久化，方便后续回放和审计。
        await ctx.session_service.append_event(
            ctx.session,
            Event(author=self.name, actions=EventActions(state_delta=dict(ctx.session.state))),
        )

    async def _append_config_revision(self, ctx: InvocationContext, reason: str) -> None:
        # 当评估失败时，把当前配置版本写入 history，并切换到 revise 模式。
        history = list(ctx.session.state.get(STATE_P2_CONFIG_HISTORY, []))
        history.append(
            {
                "version": int(ctx.session.state.get(STATE_P2_CONFIG_VERSION, 0)),
                "mode": ctx.session.state.get(STATE_P2_CONFIG_MODE, "create"),
                "reason": reason,
            }
        )
        ctx.session.state[STATE_P2_CONFIG_HISTORY] = history
        ctx.session.state[STATE_P2_CONFIG_VERSION] = int(ctx.session.state.get(STATE_P2_CONFIG_VERSION, 0)) + 1
        ctx.session.state[STATE_P2_CONFIG_MODE] = "revise"
        ctx.session.state[STATE_P2_CONFIG_PARENT_VERSION] = int(ctx.session.state[STATE_P2_CONFIG_VERSION]) - 1
        await self._commit_state(ctx)

    async def _run_step(self, ctx: InvocationContext, step_agent: BaseAgent) -> AsyncGenerator[Event, None]:
        # 统一封装“运行一步 + 提交状态”。
        async for event in step_agent.run_async(ctx):
            yield event
        await self._commit_state(ctx)

    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        # 初始化 Phase-2 的关键状态。
        if STATE_P2_EVENTS not in ctx.session.state:
            ctx.session.state[STATE_P2_EVENTS] = []
        if STATE_P2_STAGE not in ctx.session.state:
            ctx.session.state[STATE_P2_STAGE] = Phase2Stage.INIT
        if STATE_P2_STATUS not in ctx.session.state:
            ctx.session.state[STATE_P2_STATUS] = "pending"
        if STATE_P2_ITER_ROUND not in ctx.session.state:
            ctx.session.state[STATE_P2_ITER_ROUND] = 0
        if STATE_P2_ITER_MAX not in ctx.session.state:
            ctx.session.state[STATE_P2_ITER_MAX] = self.cfg.max_auto_iterations
        if STATE_P2_CONFIG_HISTORY not in ctx.session.state:
            ctx.session.state[STATE_P2_CONFIG_HISTORY] = []
        if STATE_P2_CONFIG_VERSION not in ctx.session.state:
            ctx.session.state[STATE_P2_CONFIG_VERSION] = 0
        if STATE_P2_CONFIG_MODE not in ctx.session.state:
            ctx.session.state[STATE_P2_CONFIG_MODE] = "create"

        await self._commit_state(ctx)

        try:
            # 1. 分析 URDF，得到可训练风险等级。
            async for event in self._run_step(ctx, self.analyze_urdf):
                yield event
                
            urdf_valid = ctx.session.state.get(STATE_P2_URDF_VALID, False)
            urdf_risk = ctx.session.state.get(STATE_P2_URDF_RISK, "high")
            if not urdf_valid or urdf_risk == "high":
                ctx.session.state[STATE_P2_STATUS] = "failed"
                ctx.session.state[STATE_P2_FAILURE_REASON] = f"URDF 诊断未通过或风险过高 (valid={urdf_valid}, risk={urdf_risk})，流程中止，请人工介入修复。"
                yield self._yield_text(f"taili_orchestrator: {ctx.session.state[STATE_P2_FAILURE_REASON]}")
                await self._commit_state(ctx)
                return

            # 2. 第一次生成配置前，先让配置生成 Agent 产出一版草案。
            async for event in self._run_step(ctx, self.config_synthesis):
                yield event
            # 3. 生成本地草案文件。
            async for event in self._run_step(ctx, self.generate_files):
                yield event
                
            yield self._yield_text("taili_orchestrator: [TEST MODE] 测试完成 AnalyzeURDF -> ConfigSynthesis -> GenerateFiles，暂停后续流程。")
            return
            # 4. 上传到云端，并准备远端执行。
            async for event in self._run_step(ctx, self.publish_cloud):
                yield event
            # 5. 启动训练任务。
            async for event in self._run_step(ctx, self.train):
                yield event
            if bool(ctx.session.state.get("phase2.train.should_stop", False)):
                ctx.session.state["phase2.train.should_stop"] = False
                if bool(ctx.session.state.get("phase2.train.converged", False)):
                    # 训练已收敛，TrainAgent 已设置 EVAL_PASSED=True，进入视频评估确认。
                    ctx.session.state[STATE_P2_STAGE] = Phase2Stage.EVALUATE_VIDEO
                    await self._commit_state(ctx)
                    async for event in self._run_step(ctx, self.evaluate_video):
                        yield event
                else:
                    # 训练失败早停，进入修订流程。
                    ctx.session.state[STATE_P2_EVAL_PASSED] = False
                    ctx.session.state[STATE_P2_HITL_REQUIRED] = False
                    ctx.session.state[STATE_P2_HITL_REASON] = "训练日志评估建议早停"
                    await self._commit_state(ctx)
                    yield self._yield_text("taili_phase2: 训练失败早停，进入修订流程")
            else:
                # 6. 训练完整结束后播放视频并交给视频裁判。
                ctx.session.state[STATE_P2_STAGE] = Phase2Stage.EVALUATE_VIDEO
                ctx.session.state[STATE_P2_EVAL_VIDEO_PATH] = str(Path(self.cfg.cloud_robot_lab_root) / "videos" / self.cfg.video_output_name)
                await self._commit_state(ctx)
                async for event in self._run_step(ctx, self.evaluate_video):
                    yield event

            # 7. 如果没通过，进入 revise 迭代，直到达到最大自动轮数。
            while not bool(ctx.session.state.get(STATE_P2_EVAL_PASSED, False)) and int(ctx.session.state.get(STATE_P2_ITER_ROUND, 0)) < int(ctx.session.state.get(STATE_P2_ITER_MAX, self.cfg.max_auto_iterations)):
                failure_reasons = list(ctx.session.state.get("phase2.eval.gate_failed_reasons", []))
                reason = "; ".join(str(item) for item in failure_reasons) if failure_reasons else str(ctx.session.state.get(STATE_P2_HITL_REASON, "revise"))
                await self._append_config_revision(ctx, reason=reason)
                async for event in self._run_step(ctx, self.repair):
                    yield event
                async for event in self._run_step(ctx, self.config_synthesis):
                    yield event
                # 关键：revise 后必须重新生成本地发布文件，否则上传到云端的仍是旧配置。
                async for event in self._run_step(ctx, self.generate_files):
                    yield event
                async for event in self._run_step(ctx, self.publish_cloud):
                    yield event
                async for event in self._run_step(ctx, self.train):
                    yield event
                # 注意：train 内部已在训练轮询中执行了日志评估（EvaluateTailiTrainingLogAgent），
                # 不需要在此重复调用。直接检查 should_stop 标志。
                if bool(ctx.session.state.get("phase2.train.should_stop", False)):
                    # 重置标志，避免影响可能的下一轮迭代。
                    ctx.session.state["phase2.train.should_stop"] = False
                    continue
                async for event in self._run_step(ctx, self.evaluate_video):
                    yield event

            # 8. 自动迭代结束后仍不通过，则进入 HITL。
            if not bool(ctx.session.state.get(STATE_P2_EVAL_PASSED, False)):
                ctx.session.state[STATE_P2_STAGE] = Phase2Stage.WAIT_HUMAN
                ctx.session.state[STATE_P2_HITL_REQUIRED] = True
                ctx.session.state[STATE_P2_HITL_REASON] = str(ctx.session.state.get(STATE_P2_HITL_REASON, "达到最大自动迭代轮数，需人工介入"))
                ctx.session.state[STATE_P2_STATUS] = "pending"
                await self._commit_state(ctx)
                yield self._yield_text("taili_phase2: 达到最大自动迭代轮数，进入 HITL")
                return

            # 9. 通过后归档输出。
            async for event in self._run_step(ctx, self.archive_outputs):
                yield event

            if not bool(ctx.session.state.get(STATE_P2_ARCHIVE_COMPLETED, False)):
                raise RuntimeError("归档完成标志未写入")

            ctx.session.state[STATE_P2_STAGE] = Phase2Stage.DONE
            ctx.session.state[STATE_P2_STATUS] = "succeeded"
            await self._commit_state(ctx)
            yield self._yield_text("taili_phase2: 系统骨架已完成")

        except Exception as exc:  # noqa: BLE001
            # 任意未处理错误都进入 FAILED，并写入 failure_reason。
            ctx.session.state[STATE_P2_STAGE] = Phase2Stage.FAILED
            ctx.session.state[STATE_P2_STATUS] = "failed"
            ctx.session.state[STATE_P2_FAILURE_REASON] = str(exc)
            await self._commit_state(ctx)
            yield self._yield_text(f"taili_phase2: 失败：{exc}")
