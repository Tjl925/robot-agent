from __future__ import annotations

"""taili_quad 专用多 Agent 步骤集。

这个模块建议按“从上到下的流程”来读：
1. 接任务
2. 分析 URDF
3. 生成 / 修订配置
4. 生成本地发布文件
5. 同步到云端并发起训练
6. 读取远端证据并做评估
7. 失败后进入下一轮修订
8. 归档

该模块面向固定链路：
- 本地输入 `taili_quad/`
- 云端执行框架 `robot_lab/`
- 固定的云端落点与训练闭环
"""

import asyncio
import json
import re
from pathlib import Path
from typing import Any, AsyncGenerator, ClassVar

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types

from robot_agent.schemas.config import (
    TailiCloudConfig,
    TailiConfigContext,
    TailiConfigDraft,
    TailiTrainingLogJudgeResult,
    TailiUrdfAnalysisResult,
    TailiVideoJudgeResult,
)
from robot_agent.schemas.state import (
    STATE_P2_ARCHIVE_COMPLETED,
    STATE_P2_ARCHIVE_SUMMARY,
    STATE_P2_CONFIG_HISTORY,
    STATE_P2_CONFIG_MODE,
    STATE_P2_CONFIG_PARENT_VERSION,
    STATE_P2_CONFIG_TEMPLATE,
    STATE_P2_CONFIG_TEXT,
    STATE_P2_CONFIG_VERSION,
    STATE_P2_EVAL_FAIL_REASON,
    STATE_P2_EVAL_PASSED,
    STATE_P2_EVAL_SCORE,
    STATE_P2_EVAL_VIDEO_PATH,
    STATE_P2_EVAL_VIDEO_REMOTE_PATH,
    STATE_P2_EVENTS,
    STATE_P2_HITL_REASON,
    STATE_P2_HITL_REQUIRED,
    STATE_P2_HITL_RESPONSE,
    STATE_P2_HITL_RESOLVED,
    STATE_P2_ITER_ROUND,
    STATE_P2_PLAY_EXIT_CODE,
    STATE_P2_PLAY_FAILED,
    STATE_P2_PLAY_STDERR,
    STATE_P2_PLAY_STDOUT,
    STATE_P2_STAGE,
    STATE_P2_STATUS,
    STATE_P2_TRAIN_COMMAND,
    STATE_P2_TRAIN_LOG_INPUT,
    STATE_P2_TRAIN_LOG_JUDGE_RESULT,
    STATE_P2_TRAIN_LOG_PATH,
    STATE_P2_TRAIN_METRIC_HISTORY,
    STATE_P2_TRAIN_PID,
    STATE_P2_TRAIN_STATUS,
    STATE_P2_URDF_ISSUES,
    STATE_P2_URDF_RISK,
    STATE_P2_URDF_VALID,
    STATE_P2_VIDEO_INPUT_PAYLOAD,
    STATE_P2_VIDEO_JUDGE_RESULT,
    Phase2Stage,
)
from robot_agent.tools.llm_client import UnifiedLLMClient
from robot_agent.tools.ssh_client import execute_ssh_command
from robot_agent.tools.taili_cloud import (
    download_remote_file_to_temp,
    fetch_remote_file,
    remote_check_training_status,
    remote_execute_play_in_tmux,
    remote_find_latest_video_file,
    remote_kill_training,
    remote_list_latest_run,
    remote_tail_log,
    remote_upload_taili_workspace,
    start_remote_training,
    wait_for_remote_file_stable,
)


class _TailiStepBaseAgent(BaseAgent):
    cfg: TailiCloudConfig
    model_config = {"arbitrary_types_allowed": True}

    def _add_log(self, ctx: InvocationContext, text: str) -> None:
        logs = list(ctx.session.state.get(STATE_P2_EVENTS, []))
        logs.append(text)
        ctx.session.state[STATE_P2_EVENTS] = logs

    def _yield_text(self, text: str) -> Event:
        return Event(author=self.name, content=types.Content(role="model", parts=[types.Part(text=text)]))

    def _extract_checkpoint_blocks(self, text: str) -> list[tuple[int, str]]:
        pattern = r"(?ms)^[^\n]*?(?:Learning iteration|Iteration)\s*[:=]?\s*(\d+).*?(?=^[^\n]*?(?:Learning iteration|Iteration)\s*[:=]?\s*\d+|^[^\n]*?Training time:|\Z)"
        blocks: list[tuple[int, str]] = []
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            try:
                idx = int(match.group(1))
            except (TypeError, ValueError):
                continue
            block = match.group(0).strip()
            if block:
                blocks.append((idx, block))
        blocks.sort(key=lambda item: item[0])
        return blocks

    def _extract_recent_iteration_window(self, text: str, window_size: int = 5) -> str:
        blocks = self._extract_checkpoint_blocks(text)
        if not blocks:
            return ""
        recent = blocks[-window_size:]
        return "\n\n".join(block for _, block in recent)

    def _extract_metrics_dict(self, block: str, iteration_index: int) -> dict:
        metrics: dict[str, float | int | str] = {"iteration_index": iteration_index}
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        block = ansi_escape.sub('', block)
        for line in block.splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            left, right = line.split(":", 1)
            key = left.strip()
            value_text = right.strip().rstrip(",")
            if not key:
                continue
            normalized_key = re.sub(r"\s+", "_", key)
            try:
                metrics[normalized_key] = float(value_text)
            except ValueError:
                metrics[normalized_key] = value_text
        return metrics


class AnalyzeTailiUrdfStepAgent(_TailiStepBaseAgent):
    cfg: TailiCloudConfig
    description: str = "Analyzes a URDF and returns a strict JSON diagnosis for training readiness."
    input_schema: ClassVar[Any] = TailiConfigContext
    output_schema: ClassVar[Any] = TailiUrdfAnalysisResult
    instruction: str = (
        "你是 Taili 的 URDF 诊断专家。你必须只输出严格 JSON，不要输出任何 Markdown、解释性前缀或多余文本。\n"
        "你的任务是基于输入的 URDF 文本、任务目标和参考模板，对其可训练性进行诊断。\n"
        "你要重点关注：\n"
        "1. 结构完整性（robot / link / joint / inertial / visual / collision）\n"
        "2. 命名、关节连通性、层级是否合理\n"
        "3. 是否存在明显会影响训练的风险\n\n"
        "输出的 issues 必须用中文显示，方便人工审核。\n"
        "输出必须符合以下 JSON 结构：\n"
        "{\n"
        '  "valid": boolean,\n'
        '  "risk": "low" | "medium" | "high",\n'
        '  "issues": [string, ...]\n'
        "}\n"
    )
    model_config = {"arbitrary_types_allowed": True}

    def _build_input_payload(self, ctx: InvocationContext, urdf_path: Path, urdf_text: str) -> TailiConfigContext:
        return TailiConfigContext(
            mode=str(ctx.session.state.get(STATE_P2_CONFIG_MODE, "create")),
            version=int(ctx.session.state.get(STATE_P2_CONFIG_VERSION, 0)) + 1,
            parent_version=ctx.session.state.get(STATE_P2_CONFIG_PARENT_VERSION),
            task_goal="taili_quad 速度控制训练",
            urdf_path=str(urdf_path),
            urdf_text=urdf_text,
            history=list(ctx.session.state.get(STATE_P2_CONFIG_HISTORY, [])),
            failure_reason=ctx.session.state.get(STATE_P2_EVAL_FAIL_REASON, ""),
            failure_summary=str(ctx.session.state.get(STATE_P2_HITL_REASON, "")),
            iteration_round=int(ctx.session.state.get(STATE_P2_ITER_ROUND, 0)),
            max_iterations=self.cfg.max_auto_iterations,
        )

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        ctx.session.state[STATE_P2_STAGE] = Phase2Stage.ANALYZE_URDF
        yield self._yield_text(f"[{self.name}] 正在评估 URDF 结构并分析可训练风险，请稍候...")
        urdf_path = Path(self.cfg.local_robot_root) / self.cfg.local_robots_subdir / "taili_quad.urdf"
        urdf_text = urdf_path.read_text(encoding="utf-8", errors="replace") if urdf_path.exists() else ""
        input_payload = self._build_input_payload(ctx, urdf_path, urdf_text)
        prompt_text = f"请基于以下完整事实执行你的任务:\n{json.dumps(input_payload.model_dump(), ensure_ascii=False)}"
        result = UnifiedLLMClient().generate_json(
            system_prompt=self.instruction,
            user_prompt=prompt_text,
            schema=TailiUrdfAnalysisResult,
        )
        ctx.session.state[STATE_P2_URDF_VALID] = result.valid
        ctx.session.state[STATE_P2_URDF_ISSUES] = result.issues
        ctx.session.state[STATE_P2_URDF_RISK] = result.risk
        self._add_log(ctx, f"[{self.name}] URDF 诊断完成 risk={result.risk} valid={result.valid}")
        yield self._yield_text("URDF诊断结果:\n" + result.model_dump_json(indent=2))


class TailiConfigSynthesisAgent(_TailiStepBaseAgent):
    cfg: TailiCloudConfig
    description: str = "Synthesizes a strict JSON Taili configuration draft from task, URDF, and failure evidence."
    instruction: str = (
        "你是 Taili 的配置生成专家。你必须只输出严格 JSON，不要输出任何 Markdown、解释性前缀或多余文本。\n"
        "你要根据输入上下文生成一版可直接进入发布流程的配置草案。\n"
        "当 mode=create 时，你要给出首版合理配置；当 mode=revise 时，你必须基于失败证据做最小必要修改。\n"
        "你必须显式考虑：任务目标、URDF 诊断、参考模板、历史版本、失败原因、迭代轮次。\n"
        "输出必须符合如下 JSON 结构：\n"
        "{\n"
        '  "mode": "create" | "revise",\n'
        '  "version": integer,\n'
        '  "parent_version": integer | null,\n'
        '  "task_name": string,\n'
        '  "reasoning": string,\n'
        '  "asset_code": string,\n'
        '  "agents_init_code": string,\n'
        '  "agents_ppo_cfg_code": string,\n'
        '  "task_init_code": string,\n'
        '  "flat_env_cfg_code": string,\n'
        '  "rough_env_cfg_code": string\n'
        "}\n"
    )
    output_schema: ClassVar[Any] = TailiConfigDraft
    output_key: str = STATE_P2_CONFIG_TEXT
    model_config = {"arbitrary_types_allowed": True}

    def _read_generated_files(self) -> dict[str, str] | None:
        """从 .taili_generated/ 目录读取上一版实际生成的 6 个文件。
        
        revise 模式下，大模型需要看到上次的完整代码才能做精准修改。
        如果目录不存在或为空，返回 None（说明是首次生成）。
        """
        gen_dir = Path(self.cfg.local_robot_root) / ".taili_generated"
        if not gen_dir.exists():
            return None
        file_map = {
            "taili_quad.py": gen_dir / "taili_quad.py",
            "agents/__init__.py": gen_dir / "agents" / "__init__.py",
            "agents/rsl_rl_ppo_cfg.py": gen_dir / "agents" / "rsl_rl_ppo_cfg.py",
            "__init__.py": gen_dir / "__init__.py",
            "flat_env_cfg.py": gen_dir / "flat_env_cfg.py",
            "rough_env_cfg.py": gen_dir / "rough_env_cfg.py",
        }
        result = {}
        for name, path in file_map.items():
            if path.exists():
                result[name] = path.read_text(encoding="utf-8", errors="replace")
        return result if result else None

    def _build_context(self, ctx: InvocationContext) -> TailiConfigContext:
        mode = str(ctx.session.state.get(STATE_P2_CONFIG_MODE, "create"))
        risk = str(ctx.session.state.get(STATE_P2_URDF_RISK, "medium"))
        version_index = int(ctx.session.state.get(STATE_P2_CONFIG_VERSION, 0))
        history = list(ctx.session.state.get(STATE_P2_CONFIG_HISTORY, []))
        
        ref_root = Path("reference/robot_lab")
        unitree_b2_root = ref_root / "tasks/manager_based/locomotion/velocity/config/quadruped/unitree_b2"
        ref_files = {
            "unitree.py": ref_root / "assets/unitree.py",
            "agents/__init__.py": unitree_b2_root / "agents/__init__.py",
            "agents/rsl_rl_ppo_cfg.py": unitree_b2_root / "agents/rsl_rl_ppo_cfg.py",
            "__init__.py": unitree_b2_root / "__init__.py",
            "flat_env_cfg.py": unitree_b2_root / "flat_env_cfg.py",
            "rough_env_cfg.py": unitree_b2_root / "rough_env_cfg.py"
        }
        reference_templates = {}
        for name, path in ref_files.items():
            reference_templates[name] = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        
        # revise 模式下从磁盘读取上一版生成的实际代码，create 模式下为 None
        current_draft = self._read_generated_files() if mode == "revise" else None
            
        return TailiConfigContext(
            mode=mode,
            version=version_index + 1,
            parent_version=ctx.session.state.get(STATE_P2_CONFIG_PARENT_VERSION),
            task_goal="taili_quad 速度控制训练",
            urdf_path=str(Path(self.cfg.local_robot_root) / self.cfg.local_robots_subdir / "taili_quad.urdf"),
            urdf_text=Path(self.cfg.local_robot_root, self.cfg.local_robots_subdir, "taili_quad.urdf").read_text(encoding="utf-8", errors="replace") if Path(self.cfg.local_robot_root, self.cfg.local_robots_subdir, "taili_quad.urdf").exists() else None,
            urdf_risk=risk,
            urdf_issues=ctx.session.state.get(STATE_P2_URDF_ISSUES, []),
            current_draft=current_draft,
            history=history,
            failure_reason=ctx.session.state.get(STATE_P2_EVAL_FAIL_REASON, ""),
            failure_summary=str(ctx.session.state.get(STATE_P2_HITL_REASON, "")),
            iteration_round=int(ctx.session.state.get(STATE_P2_ITER_ROUND, 0)),
            max_iterations=self.cfg.max_auto_iterations,
            reference_templates=reference_templates,
        )

    def _build_prompt_payload(self, context: TailiConfigContext) -> dict:
        return {
            "mode": context.mode,
            "version": context.version,
            "parent_version": context.parent_version,
            "task_goal": context.task_goal,
            "urdf_risk": context.urdf_risk,
            "urdf_issues": context.urdf_issues,
            "failure_reasons": context.failure_reasons,
            "failure_summary": context.failure_summary,
            "iteration_round": context.iteration_round,
            "max_iterations": context.max_iterations,
            "current_draft": context.current_draft,
            "history": context.history[-3:],
            "reference_templates": context.reference_templates,
        }

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        ctx.session.state[STATE_P2_STAGE] = Phase2Stage.SYNTHESIZE_CONFIG
        yield self._yield_text(f"[{self.name}] 正在生成核心配置草案，请稍候...")
        context = self._build_context(ctx)
        prompt_text = f"请基于以下完整事实执行你的任务:\n{json.dumps(self._build_prompt_payload(context), ensure_ascii=False)}"
        draft = UnifiedLLMClient().generate_json(
            system_prompt=self.instruction,
            user_prompt=prompt_text,
            schema=TailiConfigDraft,
        )
        history = list(ctx.session.state.get(STATE_P2_CONFIG_HISTORY, []))
        summary = {
            "mode": draft.mode,
            "version": draft.version,
            "task_name": draft.task_name,
            "reasoning": draft.reasoning,
        }
        history.append(summary)
        ctx.session.state[STATE_P2_CONFIG_HISTORY] = history
        ctx.session.state[STATE_P2_CONFIG_TEMPLATE] = "quadruped_taili_velocity"
        ctx.session.state[STATE_P2_CONFIG_TEXT] = draft.model_dump_json(indent=2)
        ctx.session.state[STATE_P2_CONFIG_MODE] = draft.mode
        ctx.session.state[STATE_P2_CONFIG_PARENT_VERSION] = draft.parent_version
        self._add_log(ctx, f"[{self.name}] 配置生成完成 mode={draft.mode} version={draft.version}")
        yield self._yield_text(f"[{self.name}] 配置生成完毕:\n{json.dumps(summary, ensure_ascii=False, indent=2)}")


class GenerateTailiFilesStepAgent(_TailiStepBaseAgent):
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        ctx.session.state[STATE_P2_STAGE] = Phase2Stage.GENERATE_FILES
        local_outputs = Path(self.cfg.local_robot_root) / ".taili_generated"
        local_outputs.mkdir(parents=True, exist_ok=True)
        (local_outputs / "agents").mkdir(parents=True, exist_ok=True)
        
        draft_json = ctx.session.state.get(STATE_P2_CONFIG_TEXT, "{}")
        draft = TailiConfigDraft.model_validate_json(draft_json)
        
        files_to_write = {
            "taili_quad.py": draft.asset_code,
            "agents/__init__.py": draft.agents_init_code,
            "agents/rsl_rl_ppo_cfg.py": draft.agents_ppo_cfg_code,
            "__init__.py": draft.task_init_code,
            "flat_env_cfg.py": draft.flat_env_cfg_code,
            "rough_env_cfg.py": draft.rough_env_cfg_code,
        }
        
        written_files = []
        for rel_path, content in files_to_write.items():
            file_path = local_outputs / rel_path
            file_path.write_text(content, encoding="utf-8")
            written_files.append(str(file_path))

        generated_manifest = {
            "local_root": self.cfg.local_robot_root,
            "generated_dir": str(local_outputs),
            "files": written_files,
            "task_name": draft.task_name,
        }
        ctx.session.state[STATE_P2_CONFIG_TEXT] = json.dumps(generated_manifest, ensure_ascii=False, indent=2)
        self._add_log(ctx, f"[{self.name}] 本地发布文件已生成")
        yield self._yield_text(f"{self.name}: 本地发布文件已生成")


class PublishTailiWorkspaceStepAgent(_TailiStepBaseAgent):
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        ctx.session.state[STATE_P2_STAGE] = Phase2Stage.PUBLISH_TO_CLOUD

        uploaded = remote_upload_taili_workspace(
            host=self.cfg.remote_host,
            port=self.cfg.remote_port,
            user=self.cfg.remote_user,
            password=self.cfg.remote_password,
            local_root=self.cfg.local_robot_root,
            cloud_root=self.cfg.cloud_robot_lab_root,
            cloud_asset_path=self.cfg.cloud_asset_path,
            cloud_task_cfg_root=self.cfg.cloud_task_cfg_root,
            timeout_seconds=self.cfg.remote_timeout_seconds,
        )
        self._add_log(ctx, f"[{self.name}] 远端发布完成: uploaded {len(uploaded)} files")
        yield self._yield_text(f"{self.name}: 云端发布完成，准备训练")

class TrainTailiStepAgent(_TailiStepBaseAgent):
    """云端训练执行与日志轮询 Agent。

    【输出状态协议】
    训练结束后只通过 STATE_P2_TRAIN_STATUS 向编排器传达退出状态：
    - "completed":     训练正常跑完或被判定已收敛，且视频渲染命令正常执行成功，等待视频裁判终审。
    - "early_stopped":  日志裁判判定训练发散、爆炸或崩溃，已强制杀死远端进程且不渲染视频。
    - "play_failed":    训练已完成但 play.py 视频渲染命令报错被熔断拦截。
    - "train_failed":   远端训练命令非零退出，或训练会话异常消失，无法正常捕获状态。
    - "train_timeout":  训练时间超过预设的 max_training_minutes，已被强行杀死终止。
    
    TrainAgent 不设置 EVAL_PASSED，最终判定通过与否交由 EvaluateTailiVideoAgent 视频裁判决定。
    """
    evaluate_training_log: EvaluateTailiTrainingLogAgent | None = None

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        ctx.session.state[STATE_P2_STAGE] = Phase2Stage.RUN_TRAINING
        host = self.cfg.remote_host
        port = self.cfg.remote_port
        user = self.cfg.remote_user
        password = self.cfg.remote_password
        if not all([host, port, user, password]):
            raise RuntimeError("缺少远端 SSH 信息，无法启动训练")
        if not ctx.session.state.get(STATE_P2_TRAIN_COMMAND):
            ctx.session.state[STATE_P2_TRAIN_COMMAND] = self.cfg.train_command_template.format(task_name=self.cfg.task_name)

        # 【阶段 1：通过 tmux 会话下发训练指令】
        start_info = start_remote_training(host, port, user, password, str(ctx.session.state[STATE_P2_TRAIN_COMMAND]), self.cfg.cloud_tmp_dir, self.cfg.remote_timeout_seconds)
        train_session = start_info["session_name"]
        train_exit_code_path = start_info["exit_code_path"]
        ctx.session.state[STATE_P2_TRAIN_PID] = train_session  # 存储 tmux session name 作为进程句柄
        ctx.session.state[STATE_P2_TRAIN_LOG_PATH] = start_info["log_path"]
        ctx.session.state[STATE_P2_TRAIN_STATUS] = "running"
        yield self._yield_text(f"[训练已启动] tmux session={train_session}, 可通过 tmux attach -t {train_session} 查看实时输出")

        last_evaluated_iteration = 0
        byte_offset = 0
        pending_log_buffer = ""
        sleep_seconds = 300.0
        metric_history: list[dict] = []
        poll_round = 0
        total_iterations = None
        sample_window_size = self.cfg.eval_sample_window_size
        ctx.session.state[STATE_P2_TRAIN_METRIC_HISTORY] = metric_history

        # 【阶段 2：自适应预热与步长估算】
        warmup_hit = False
        yield self._yield_text(f"\033[36m[{self.name}] 进入训练启动自适应预热阶段，预计等待 180 秒以估算步长耗时...\033[0m")
        for i in range(3):
            yield self._yield_text(f"\033[90m[{self.name}] [预热轮次 {i+1}/3] 正在拉取远端启动日志以进行探活估算...\033[0m")
            await asyncio.sleep(60)
            warmup_text, byte_offset = remote_tail_log(host, port, user, password, start_info["log_path"], self.cfg.remote_timeout_seconds, byte_offset)
            if "Learning iteration" in warmup_text and "Iteration time" in warmup_text:
                warmup_hit = True
                # 使用 findall 获取所有耗时特征，取最后一个匹配值（即最新稳态速度），完美规避第 0 轮虚高的初始化开销
                matches = re.findall(r"Iteration time:\s*([\d.]+)s", warmup_text, flags=re.IGNORECASE)
                if matches:
                    try:
                        iteration_time = float(matches[-1])
                        sleep_seconds = max(20.0, min(500.0, iteration_time * float(self.cfg.eval_check_interval)))
                        yield self._yield_text(f"\033[92m[{self.name}] 预热成功！单步 iteration 稳态耗时约 {iteration_time:.2f}s，自适应轮询评估周期设为 {sleep_seconds:.1f} 秒\033[0m")
                    except ValueError:
                        sleep_seconds = 300.0
                else:
                    sleep_seconds = 300.0
                pending_log_buffer = warmup_text
                break
        if not warmup_hit:
            yield self._yield_text(f"\033[93m[{self.name}] 预热未捕获到特定指标特征（可能 Robot Lab 加载 GPU 缓存较慢），将采用默认评估周期 {sleep_seconds} 秒\033[0m")

        # 【阶段 3：长时轮询与采样窗口日志截取】
        # 核心设计：
        #   1. 每轮从远端增量拉取新日志，拼接到 pending_log_buffer。
        #   2. 用正则解析出所有完整的 checkpoint blocks。
        #   3. 最后一个 block 可能尚未写完（远端训练仍在输出），
        #      因此将其保留回 pending_log_buffer，只处理前面确认完整的 blocks。
        #   4. 在确认完整的新增 blocks 中，只取最后 sample_window_size 个
        #      组成一个采样窗口，追加到 metric_history。
        #   5. last_evaluated_iteration 更新为所有新增 blocks 的最大 iteration，
        #      避免下一轮重复处理被跳过的中间 iteration。
        max_checks = max(1, int((self.cfg.max_training_minutes * 60) // max(1.0, sleep_seconds)))
        for _ in range(max_checks):
            # 1. 每一轮 poll 优先无延迟检查远端进程状态，确保没有发生崩溃夭折，避免空等 sleep_seconds
            remote_status = remote_check_training_status(host, port, user, password, train_session, train_exit_code_path, self.cfg.remote_timeout_seconds)
            rs = remote_status["status"]
            if rs == "completed":
                ctx.session.state[STATE_P2_TRAIN_STATUS] = "completed"
                yield self._yield_text(f"{self.name}: 远端训练已自然结束 (exit=0)，进入视频渲染")
                break
            if rs == "failed":
                ctx.session.state[STATE_P2_TRAIN_STATUS] = "train_failed"
                ctx.session.state[STATE_P2_EVAL_FAIL_REASON] = f"远端训练命令非零退出 (exit_code={remote_status['exit_code']})"
                yield self._yield_text(f"{self.name}: 远端训练失败 (exit={remote_status['exit_code']})")
                return
            if rs == "unknown_failed":
                ctx.session.state[STATE_P2_TRAIN_STATUS] = "train_failed"
                ctx.session.state[STATE_P2_EVAL_FAIL_REASON] = "训练会话已不存在，但未生成 exit_code 文件，状态未知"
                yield self._yield_text(f"{self.name}: 训练进程异常退出，状态未知")
                return
            
            # 2. 状态正常运行 (rs == "running")，此时先进行冷却休眠，让远端跑出新步数进度
            await asyncio.sleep(sleep_seconds)

            # 3. 休眠结束，立即扒取远端增量日志
            new_text, byte_offset = remote_tail_log(host, port, user, password, start_info["log_path"], self.cfg.remote_timeout_seconds, byte_offset)
            if new_text:
                pending_log_buffer += new_text
                yield self._yield_text(f"\033[90m[{self.name}] 发现日志新增 {len(new_text)} 字节，正在提取特征块...\033[0m")
            else:
                yield self._yield_text(f"\033[90m[{self.name}] 远端日志本轮未产生增量输出...\033[0m")

            all_blocks = self._extract_checkpoint_blocks(pending_log_buffer)
            if not all_blocks:
                yield self._yield_text(f"\033[93m[{self.name}] 目前尚未在日志中识别到任何有效 Checkpoint 块，等待下一轮重试...\033[0m")
                continue

            # 保护未完成的最后一个 block：
            # "Training time:" 出现表示训练已结束，所有 block 确认完整。
            last_block_text = all_blocks[-1][1]
            last_block_pos = pending_log_buffer.rfind(last_block_text)
            training_finished = "Training time:" in pending_log_buffer
            if training_finished:
                confirmed_blocks = all_blocks
                pending_log_buffer = ""
            else:
                confirmed_blocks = all_blocks[:-1] if len(all_blocks) > 1 else []
                if last_block_pos >= 0:
                    pending_log_buffer = pending_log_buffer[last_block_pos:]
                if not confirmed_blocks:
                    yield self._yield_text(f"\033[90m[{self.name}] 首个 Checkpoint 块尚未完全写毕，等待 {sleep_seconds:.1f} 秒待远端日志写完...\033[0m")
                    continue

            # 筛选出本轮新增的 blocks
            new_blocks = [(idx, blk) for idx, blk in confirmed_blocks if idx > last_evaluated_iteration]
            if not new_blocks:
                yield self._yield_text(f"\033[90m[{self.name}] 未发现相比上一轮新增的已完成 Checkpoint 块，等待 {sleep_seconds:.1f} 秒...\033[0m")
                continue

            # 采样窗口：只取最后 N 个 block 的指标
            sampled = new_blocks[-sample_window_size:]
            sample_metrics = [self._extract_metrics_dict(blk, idx) for idx, blk in sampled]

            # last_evaluated_iteration 更新为所有新增 blocks 的最大 iteration
            last_evaluated_iteration = new_blocks[-1][0]
            poll_round += 1

            window_entry = {
                "poll_round": poll_round,
                "iteration_range": [sampled[0][0], sampled[-1][0]],
                "total_new_blocks": len(new_blocks),
                "samples": sample_metrics,
            }
            metric_history.append(window_entry)
            ctx.session.state[STATE_P2_TRAIN_METRIC_HISTORY] = metric_history
            self._add_log(ctx, f"[{self.name}] poll#{poll_round} sampled iter {sampled[0][0]}~{sampled[-1][0]} ({len(sampled)}/{len(new_blocks)})")

            sampled_text = "\n\n".join(blk for _, blk in sampled)
            yield self._yield_text(f"[poll #{poll_round}] iteration {sampled[0][0]}~{sampled[-1][0]}\n{sampled_text}")

            # 动态从最新日志中解析出总迭代次数 y，一旦抓取成功后不再重复匹配
            if total_iterations is None and pending_log_buffer:
                iter_match = re.search(
                    r"(?:Learning iteration|Iteration)\s*[:=]?\s*\d+\s*/\s*(\d+)",
                    pending_log_buffer,
                    flags=re.IGNORECASE
                )
                if iter_match:
                    try:
                        total_iterations = int(iter_match.group(1))
                    except (TypeError, ValueError):
                        pass

            # 【阶段 4：唤起日志裁判】
            judge_input = {
                "metric_history": metric_history,  # 采样窗口列表，非平铺指标
                "check_interval": self.cfg.eval_check_interval,
                "sleep_seconds": sleep_seconds,
                "total_iterations": total_iterations,
            }
            ctx.session.state[STATE_P2_TRAIN_LOG_INPUT] = judge_input
            async for judge_event in self.evaluate_training_log.run_async(ctx):
                yield judge_event

            # 【阶段 5：裁决行动响应】
            # 只通过 STATE_P2_TRAIN_STATUS 传达退出状态，不设置 EVAL_PASSED。
            judge_result = ctx.session.state.get(STATE_P2_TRAIN_LOG_JUDGE_RESULT, {}) or {}
            action = str(judge_result.get("action", "continue"))
            if action == "stop_failed":
                remote_kill_training(host, port, user, password, train_session, self.cfg.remote_timeout_seconds)
                ctx.session.state[STATE_P2_TRAIN_STATUS] = "early_stopped"
                ctx.session.state[STATE_P2_EVAL_FAIL_REASON] = judge_result.get("reason", "")
                yield self._yield_text(f"{self.name}: 训练发散，已执行 early stop")
                return
            if action == "stop_converged":
                remote_kill_training(host, port, user, password, train_session, self.cfg.remote_timeout_seconds)
                ctx.session.state[STATE_P2_TRAIN_STATUS] = "completed"
                yield self._yield_text(f"{self.name}: 训练已收敛，进入视频渲染")
                break
        else:
            # max_checks 到期：不允许直接标记 completed，必须再次检查远端状态。
            remote_status = remote_check_training_status(host, port, user, password, train_session, train_exit_code_path, self.cfg.remote_timeout_seconds)
            rs = remote_status["status"]
            if rs == "completed":
                ctx.session.state[STATE_P2_TRAIN_STATUS] = "completed"
                yield self._yield_text(f"{self.name}: 轮询到期，远端训练已正常结束 (exit=0)")
            elif rs == "failed":
                ctx.session.state[STATE_P2_TRAIN_STATUS] = "train_failed"
                ctx.session.state[STATE_P2_EVAL_FAIL_REASON] = f"轮询到期，远端训练非零退出 (exit_code={remote_status['exit_code']})"
                yield self._yield_text(f"{self.name}: 轮询到期，训练失败 (exit={remote_status['exit_code']})")
                return
            elif rs == "running":
                remote_kill_training(host, port, user, password, train_session, self.cfg.remote_timeout_seconds)
                ctx.session.state[STATE_P2_TRAIN_STATUS] = "train_timeout"
                ctx.session.state[STATE_P2_EVAL_FAIL_REASON] = f"训练超过 {self.cfg.max_training_minutes} 分钟仍未结束，已终止 tmux 会话"
                yield self._yield_text(f"{self.name}: 训练超时，已终止 tmux 会话")
                return
            else:
                ctx.session.state[STATE_P2_TRAIN_STATUS] = "train_failed"
                ctx.session.state[STATE_P2_EVAL_FAIL_REASON] = "轮询到期，训练会话已不存在且未生成 exit_code"
                yield self._yield_text(f"{self.name}: 训练会话异常退出")
                return
                
        # 【阶段 6：视频渲染 (Play) 流程】
        # 当训练达到最大设定步数自然完成，或被裁判认定为“已收敛”时，会执行此部分。
        # 阻塞式地调用 play.py 渲染策略输出视频，以备最后的视频裁判 Agent 使用。
        yield self._yield_text(f"{self.name}: 开始渲染评估视频...")
        play_cmd = self.cfg.play_command_template.format(
            task_name=self.cfg.task_name,
            video_length=self.cfg.play_video_length
        )
        try:
            play_out, play_code = remote_execute_play_in_tmux(
                host, port, user, password,
                train_session, play_cmd, self.cfg.cloud_tmp_dir, self.cfg.play_timeout_seconds,
            )
        except Exception as exc:
            ctx.session.state[STATE_P2_PLAY_STDOUT] = ""
            ctx.session.state[STATE_P2_PLAY_STDERR] = str(exc)
            ctx.session.state[STATE_P2_PLAY_EXIT_CODE] = -1
            ctx.session.state[STATE_P2_PLAY_FAILED] = True
            ctx.session.state[STATE_P2_TRAIN_STATUS] = "play_failed"
            ctx.session.state[STATE_P2_EVAL_FAIL_REASON] = f"play.py tmux 执行异常: {exc}"
            yield self._yield_text(f"{self.name}: 视频渲染异常: {exc}")
            return
        ctx.session.state[STATE_P2_PLAY_STDOUT] = play_out
        ctx.session.state[STATE_P2_PLAY_STDERR] = ""
        ctx.session.state[STATE_P2_PLAY_EXIT_CODE] = play_code
        if play_code != 0:
            self._add_log(ctx, f"视频渲染失败: {play_out[:300]}")
            ctx.session.state[STATE_P2_PLAY_FAILED] = True
            ctx.session.state[STATE_P2_TRAIN_STATUS] = "play_failed"
            ctx.session.state[STATE_P2_EVAL_FAIL_REASON] = f"play.py 渲染失败 (exit={play_code}): {play_out[:200]}"
            yield self._yield_text(f"{self.name}: 视频渲染失败 (exit={play_code})")
            return
        ctx.session.state[STATE_P2_PLAY_FAILED] = False
        yield self._yield_text(f"{self.name}: 训练完成，视频已渲染")


class EvaluateTailiTrainingLogAgent(_TailiStepBaseAgent):
    cfg: TailiCloudConfig
    description: str = "Evaluates sampled metric windows and decides whether to stop."

    instruction: str = (
        "你是 Taili 的强化学习训练裁判。你必须只输出严格 JSON，不要输出 Markdown、解释文字或代码块。\n"
        "\n"
        "你会接收到一个 payload，包含了历史所有提取到的指标日志块列表 metric_history，以及总迭代次数 total_iterations。\n"
        "1. metric_history 包含了每一次长时轮询（poll）结束时，提取到的最后几个迭代步（iteration）的全量高维指标细节（包括核心奖励、价值损失、熵，以及各种细分姿态奖励惩罚项，如 Episode_Reward/xxx）。\n"
        "   请注意：每一轮 poll 里的 samples 都包含了数个具体的 iteration 指标。\n"
        "\n"
        "你的任务是全面剖析这一完整的历史高维指标序列，精细化地判断训练是走向良性收敛、出现发散/崩溃，还是仍在健康地上升。\n"
        "\n"
        "判定原则：\n"
        "1. 发散/崩溃 stop_failed：\n"
        "   - 最新指标出现 NaN、Inf；\n"
        "   - 经过较多 iteration 后，主奖励或核心姿态惩罚项长期没有改善且极差；\n"
        "   - 主奖励持续暴跌，或 value_loss 出现明显的指数级爆炸发散。\n"
        "\n"
        "2. 收敛 stop_converged：\n"
        "   - 性能极其优秀：Mean_reward 稳定在较高水平，且各项细分动作姿态惩罚项（如关节力矩惩罚、足端碰撞惩罚、高度抖动惩罚等）均已被最小化并达到稳态；\n"
        "   - 增长完全停滞：最近许多个 iteration 之间，所有核心及细节指标均已进入几乎绝对平坦的平台期（无上升空间）；\n"
        "   - 姿态健康：必须确保那些指示“姿态质量”的惩罚项（如 lin_vel_z_l2, ang_vel_xy_l2, joint_torques_l2 等）没有发生恶化，走路姿势是优雅合理的，而非牺牲动作质量来换取高 reward；\n"
        "   - episode_length 稳定维持在最大期望步数（如 1000 左右）；\n"
        "   - 只有在“各项细分性能极佳 + 姿态质量优秀 + 指标绝对进入平台期 + 损失稳定”同时满足时，才返回 stop_converged。\n"
        "\n"
        "3. 正常 continue：\n"
        "   - 核心奖励或关键姿态奖励仍在上升/改善，尚未达到绝对瓶颈；\n"
        "   - 训练早期震荡属正常，或因迭代次数太少（如小于 2000 步且处于上升期）无法轻易断言已经完全收敛；\n"
        "   - 在不完全确定发散或绝对收敛前，请宽容并返回 continue。\n"
        "\n"
        "输出必须是严格 JSON，格式如下：\n"
        "{\n"
        "  \"action\": \"continue\" | \"stop_failed\" | \"stop_converged\",\n"
        "  \"score\": {},\n"
        "  \"reason\": \"\"\n"
        "}\n"
        "\n"
        "score 中建议包含你关注到的关键姿态指标和整体奖励走势。\n"
    )

    output_schema: ClassVar[Any] = TailiTrainingLogJudgeResult
    output_key: str = "phase2.train.log_judge"
    model_config = {"arbitrary_types_allowed": True}

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        ctx.session.state[STATE_P2_STAGE] = Phase2Stage.EVALUATE_TRAIN_LOG

        raw_metric_history = list(ctx.session.state.get(STATE_P2_TRAIN_METRIC_HISTORY, []))

        judge_input = ctx.session.state.get(STATE_P2_TRAIN_LOG_INPUT, {})
        total_iterations = judge_input.get("total_iterations")

        payload = {
            "metric_history": raw_metric_history,
            "raw_metric_window_count": len(raw_metric_history),
            "total_iterations": total_iterations,
        }

        ctx.session.state[STATE_P2_TRAIN_LOG_INPUT] = payload

        prompt_text = (
            "请基于以下事实执行日志趋势裁判任务。\n"
            f"{json.dumps(payload, ensure_ascii=False)}"
        )

        result = UnifiedLLMClient().generate_json(
            system_prompt=self.instruction,
            user_prompt=prompt_text,
            schema=TailiTrainingLogJudgeResult,
        )

        ctx.session.state[STATE_P2_TRAIN_LOG_JUDGE_RESULT] = result.model_dump()

        # 美化终端控制台输出
        action = result.action
        confidence = result.confidence
        score = result.score or {}
        reasons = result.reasons or []
        
        latest_iter = "未知"
        if raw_metric_history:
            last_window = raw_metric_history[-1]
            samples = last_window.get("samples", [])
            if samples and isinstance(samples[-1], dict):
                latest_iter = samples[-1].get("iteration_index", "未知")
        
        # 根据决策选择颜色
        if action == "stop_failed":
            color_prefix = "\033[1;91m"  # 粗体亮红
            action_desc = "终止运行 - 训练失败/发散 (STOP_FAILED)"
        elif action == "stop_converged":
            color_prefix = "\033[1;92m"  # 粗体亮绿
            action_desc = "早停终止 - 训练已收敛 (STOP_CONVERGED)"
        else:
            color_prefix = "\033[90m"    # 灰色
            action_desc = "正常进行 - 继续训练 (CONTINUE)"
            
        reset = "\033[0m"
        bold = "\033[1m"
        
        reasons_str = "\n".join(f"  \033[90m•\033[0m {color_prefix}{r}{reset}" for r in reasons)
        score_details = ", ".join(f"{k}: {v}" for k, v in score.items())
        
        judge_report = (
            f"{color_prefix}┌─────────────────────────── [ 日志裁判实时裁决报告 ] ───────────────────────────┐{reset}\n"
            f"{color_prefix}│{reset} {bold}当前迭代数{reset}: {latest_iter:<68} {color_prefix}│{reset}\n"
            f"{color_prefix}│{reset} {bold}裁决决策{reset}  : {color_prefix}{action_desc:<64}{reset} {color_prefix}│{reset}\n"
            f"{color_prefix}│{reset} {bold}决策置信度{reset}: {f'{confidence:.2f}':<68} {color_prefix}│{reset}\n"
            f"{color_prefix}│{reset} {bold}核心指标{reset}  : {score_details:<68} {color_prefix}│{reset}\n"
            f"{color_prefix}├────────────────────────────────────────────────────────────────────────────────┤{reset}\n"
            f"{color_prefix}│{reset} {bold}裁决原因深度剖析{reset}:\n"
            f"{reasons_str}\n"
            f"{color_prefix}└────────────────────────────────────────────────────────────────────────────────┘{reset}"
        )
        
        yield self._yield_text(judge_report)


class EvaluateTailiVideoAgent(_TailiStepBaseAgent):
    cfg: TailiCloudConfig
    description: str = "Evaluates the played video file and decides final pass/fail."
    instruction: str = (
        "你是 Taili 的视频裁判。你必须只输出严格 JSON，不要输出任何 Markdown、解释性前缀或多余文本。\n"
        "你可以通过多模态能力直接观看机器人运行视频，基于视频画面中机器人的步态质量、稳定性以及是否摔倒，给出最终的验收结论。\n"
        "输出必须符合如下 JSON 结构：\n"
        "{\n"
        '  "passed": boolean,\n'
        '  "score": object,\n'
        '  "reason": string\n'
        "}\n"
    )
    output_schema: ClassVar[Any] = TailiVideoJudgeResult
    output_key: str = "phase2.video.judge"
    model_config = {"arbitrary_types_allowed": True}

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        ctx.session.state[STATE_P2_STAGE] = Phase2Stage.EVALUATE_VIDEO
        host = self.cfg.remote_host
        port = self.cfg.remote_port
        user = self.cfg.remote_user
        password = self.cfg.remote_password
        run_root = remote_list_latest_run(host, port, user, password, self.cfg.eval_log_root, self.cfg.remote_timeout_seconds)
        video_remote_path = remote_find_latest_video_file(host, port, user, password, run_root, self.cfg.remote_timeout_seconds)
        if not video_remote_path:
            raise RuntimeError("未找到远端视频文件，无法进入视频评估")
        if not wait_for_remote_file_stable(host, port, user, password, video_remote_path, self.cfg.remote_timeout_seconds):
            raise RuntimeError("远端视频文件未稳定落盘，无法进入视频评估")
        video_local_path = download_remote_file_to_temp(host, port, user, password, video_remote_path, self.cfg.remote_timeout_seconds)
        ctx.session.state[STATE_P2_EVAL_VIDEO_PATH] = video_local_path
        ctx.session.state[STATE_P2_EVAL_VIDEO_REMOTE_PATH] = video_remote_path

        payload = {
            "play_exit_code": ctx.session.state.get(STATE_P2_PLAY_EXIT_CODE),
            "play_stderr": ctx.session.state.get(STATE_P2_PLAY_STDERR, ""),
            "note": "真正的视频内容在下方的多模态输入中提供，请仔细观察机器人走路姿势是否合理，是否摔倒，有无明显失真。",
        }
        ctx.session.state[STATE_P2_VIDEO_INPUT_PAYLOAD] = payload
        
        # 将视频文件转为 base64
        import base64
        with open(video_local_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("utf-8")
        
        prompt_content = [
            {"type": "text", "text": f"请基于以下完整事实执行你的任务:\n{json.dumps(payload, ensure_ascii=False)}\n\n以下是机器人运行的真实评估视频："},
            {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}}
        ]
        
        from robot_agent.tools.llm_client import LLMCallConfig
        dashscope_config = LLMCallConfig(
            api_key_env="DASHSCOPE_API_KEY",
            base_url_env="DASHSCOPE_BASE_URL",
            default_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model="qwen3.6-plus"
        )
        result = UnifiedLLMClient(dashscope_config).generate_json(
            system_prompt=self.instruction,
            user_prompt=prompt_content,
            schema=TailiVideoJudgeResult,
        )
        ctx.session.state[STATE_P2_VIDEO_JUDGE_RESULT] = result.model_dump()

        # 关键：把视频评估结果写回编排器期望的 state 键，否则编排器永远认为评估没通过。
        ctx.session.state[STATE_P2_EVAL_PASSED] = result.passed
        ctx.session.state[STATE_P2_EVAL_SCORE] = result.score
        if result.passed:
            ctx.session.state[STATE_P2_EVAL_FAIL_REASON] = ""
            ctx.session.state[STATE_P2_HITL_REQUIRED] = False
            ctx.session.state[STATE_P2_HITL_REASON] = ""
        else:
            ctx.session.state[STATE_P2_EVAL_FAIL_REASON] = result.reason
            ctx.session.state[STATE_P2_HITL_REQUIRED] = True
            ctx.session.state[STATE_P2_HITL_REASON] = result.reason if result.reason else "视频评估未通过"
        self._add_log(ctx, f"[{self.name}] 视频评估完成 passed={result.passed}")
        yield self._yield_text(result.model_dump_json(indent=2))


class RepairTailiWorkflowStepAgent(_TailiStepBaseAgent):
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        ctx.session.state[STATE_P2_STAGE] = Phase2Stage.ITERATE_TUNING
        ctx.session.state[STATE_P2_ITER_ROUND] = int(ctx.session.state.get(STATE_P2_ITER_ROUND, 0)) + 1
        ctx.session.state[STATE_P2_HITL_RESPONSE] = self.cfg.hitl_response_text or ""
        ctx.session.state[STATE_P2_HITL_RESOLVED] = bool(self.cfg.hitl_response_text)
        ctx.session.state[STATE_P2_HITL_REQUIRED] = False if self.cfg.hitl_response_text else True
        # 失败原因已由上游写入 STATE_P2_HITL_REASON / STATE_P2_EVAL_FAIL_REASON，无需再冗余复制
        self._add_log(ctx, f"[{self.name}] 进入迭代轮次 {ctx.session.state[STATE_P2_ITER_ROUND]}")
        yield self._yield_text(f"{self.name}: 已进入下一轮迭代")


class ArchiveTailiOutputsStepAgent(_TailiStepBaseAgent):
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        ctx.session.state[STATE_P2_STAGE] = Phase2Stage.ARCHIVE_OUTPUTS
        summary = {
            "task_id": f"taili-{self.cfg.session_id}",
            "task_name": self.cfg.task_name,
            "cloud_asset_path": self.cfg.cloud_asset_path,
            "cloud_task_root": self.cfg.cloud_task_cfg_root,
            "train_command": ctx.session.state.get(STATE_P2_TRAIN_COMMAND),
            "eval": ctx.session.state.get(STATE_P2_EVAL_SCORE, {}),
            "passed": ctx.session.state.get(STATE_P2_EVAL_PASSED, False),
        }
        ctx.session.state[STATE_P2_ARCHIVE_SUMMARY] = summary
        ctx.session.state[STATE_P2_ARCHIVE_COMPLETED] = True
        self._add_log(ctx, f"[{self.name}] 归档完成")
        yield self._yield_text(f"{self.name}: 归档完成")
