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
    STATE_P2_EVAL_CHECK_INTERVAL,
    STATE_P2_EVAL_CHECKPOINT_GLOB,
    STATE_P2_EVAL_CHECKPOINT_PATH,
    STATE_P2_EVAL_FAIL_REASONS,
    STATE_P2_EVAL_LAST_CHECK_ITER,
    STATE_P2_EVAL_LAST_CHECK_SUMMARY,
    STATE_P2_EVAL_LOG_ROOT,
    STATE_P2_EVAL_EARLY_STOP_RULES,
    STATE_P2_EVAL_METRIC_SPECS,
    STATE_P2_EVAL_PASSED,
    STATE_P2_EVAL_SCORE,
    STATE_P2_EVAL_VIDEO_NAME,
    STATE_P2_EVAL_VIDEO_PATH,
    STATE_P2_EVENTS,
    STATE_P2_HITL_REASON,
    STATE_P2_HITL_REQUIRED,
    STATE_P2_HITL_RESPONSE,
    STATE_P2_HITL_RESOLVED,
    STATE_P2_ITER_ROUND,
    STATE_P2_STAGE,
    STATE_P2_STATUS,
    STATE_P2_TRAIN_COMMAND,
    STATE_P2_TRAIN_EARLY_STOPPED,
    STATE_P2_TRAIN_RUN_ID,
    STATE_P2_TRAIN_STATUS,
    STATE_P2_TRAIN_TOTAL_ITERATIONS,
    STATE_P2_URDF_ISSUES,
    STATE_P2_URDF_RISK,
    STATE_P2_URDF_VALID,
    Phase2Stage,
)
from robot_agent.tools.llm_client import UnifiedLLMClient
from robot_agent.tools.ssh_client import execute_ssh_command
from robot_agent.tools.taili_cloud import (
    render_taili_asset_py,
    render_taili_task_cfg_py,
    render_taili_task_init_py,
    download_remote_file_to_temp,
    fetch_remote_file,
    remote_find_latest_matching_file,
    remote_find_latest_video_file,
    remote_kill_process,
    remote_list_latest_run,
    remote_tail_log,
    remote_upload_taili_workspace,
    start_remote_training,
    sync_local_tree_to_cloud,
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
        pattern = r"(?ms)^.*?(?:Learning iteration|Iteration)\s*[:=]?\s*(\d+).*?(?=^(?:Learning iteration|Iteration)\s*[:=]?\s*\d+|^Training time:|\Z)"
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
            failure_reasons=list(ctx.session.state.get(STATE_P2_EVAL_FAIL_REASONS, [])),
            failure_summary=str(ctx.session.state.get(STATE_P2_HITL_REASON, "")),
            iteration_round=int(ctx.session.state.get(STATE_P2_ITER_ROUND, 0)),
            max_iterations=int(ctx.session.state.get("phase2.train.max_iterations", self.cfg.max_training_iterations)),
            cloud_asset_path=self.cfg.cloud_asset_path,
            cloud_task_init_path=self.cfg.cloud_task_init_path,
            cloud_task_cfg_root=self.cfg.cloud_task_cfg_root,
            cloud_data_root=f"{self.cfg.cloud_robot_lab_root}/source/robot_lab/data/Robots",
        )

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        ctx.session.state[STATE_P2_STAGE] = Phase2Stage.ANALYZE_URDF
        yield self._yield_text(f"[{self.name}] 正在评估 URDF 结构并分析可训练风险，请稍候...")
        urdf_path = Path(self.cfg.local_robot_root) / self.cfg.local_robots_subdir / "robot.urdf"
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
            urdf_path=str(Path(self.cfg.local_robot_root) / self.cfg.local_robots_subdir / "robot.urdf"),
            urdf_text=Path(self.cfg.local_robot_root, self.cfg.local_robots_subdir, "robot.urdf").read_text(encoding="utf-8", errors="replace") if Path(self.cfg.local_robot_root, self.cfg.local_robots_subdir, "robot.urdf").exists() else None,
            urdf_risk=risk,
            urdf_issues=ctx.session.state.get(STATE_P2_URDF_ISSUES, []),
            current_draft=current_draft,
            history=history,
            failure_reasons=list(ctx.session.state.get(STATE_P2_EVAL_FAIL_REASONS, [])),
            failure_summary=str(ctx.session.state.get(STATE_P2_HITL_REASON, "")),
            iteration_round=int(ctx.session.state.get(STATE_P2_ITER_ROUND, 0)),
            max_iterations=int(ctx.session.state.get("phase2.train.max_iterations", self.cfg.max_training_iterations)),
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
        ctx.session.state[STATE_P2_TRAIN_RUN_ID] = f"taili-run-taili-{self.cfg.session_id}-r{ctx.session.state.get(STATE_P2_ITER_ROUND, 0)}"
        ctx.session.state[STATE_P2_TRAIN_COMMAND] = self.cfg.train_command_template.format(task_name=self.cfg.task_name)
        ctx.session.state[STATE_P2_TRAIN_STATUS] = "published"

        copied = sync_local_tree_to_cloud(
            local_root=self.cfg.local_robot_root,
            cloud_root=self.cfg.cloud_robot_lab_root,
            files=[
                (f"{self.cfg.local_robots_subdir}/robot.urdf", "source/robot_lab/robot_lab/data/Robots/robot.urdf"),
                (".taili_generated/taili_quad.py", self.cfg.cloud_asset_path.replace(self.cfg.cloud_robot_lab_root + "/", "")),
                (".taili_generated/agents/__init__.py", f"{self.cfg.cloud_task_cfg_root.replace(self.cfg.cloud_robot_lab_root + '/', '')}/agents/__init__.py"),
                (".taili_generated/agents/rsl_rl_ppo_cfg.py", f"{self.cfg.cloud_task_cfg_root.replace(self.cfg.cloud_robot_lab_root + '/', '')}/agents/rsl_rl_ppo_cfg.py"),
                (".taili_generated/__init__.py", self.cfg.cloud_task_init_path.replace(self.cfg.cloud_robot_lab_root + "/", "")),
                (".taili_generated/flat_env_cfg.py", f"{self.cfg.cloud_task_cfg_root.replace(self.cfg.cloud_robot_lab_root + '/', '')}/flat_env_cfg.py"),
                (".taili_generated/rough_env_cfg.py", f"{self.cfg.cloud_task_cfg_root.replace(self.cfg.cloud_robot_lab_root + '/', '')}/rough_env_cfg.py"),
            ],
        )
        uploaded = remote_upload_taili_workspace(
            host=self.cfg.remote_host,
            port=self.cfg.remote_port,
            user=self.cfg.remote_user,
            password=self.cfg.remote_password,
            local_root=self.cfg.local_robot_root,
            local_robots_subdir=self.cfg.local_robots_subdir,
            cloud_asset_path=self.cfg.cloud_asset_path,
            cloud_task_init_path=self.cfg.cloud_task_init_path,
            cloud_task_cfg_root=self.cfg.cloud_task_cfg_root,
            timeout_seconds=self.cfg.remote_timeout_seconds,
        )
        ctx.session.state["phase2.publish.summary"] = {
            "copied": copied,
            "uploaded": uploaded,
            "cloud_root": self.cfg.cloud_robot_lab_root,
            "asset_path": self.cfg.cloud_asset_path,
            "task_init_path": self.cfg.cloud_task_init_path,
            "task_cfg_root": self.cfg.cloud_task_cfg_root,
        }
        self._add_log(ctx, f"[{self.name}] 远端发布完成: copied={copied}, uploaded={uploaded}")
        yield self._yield_text(f"{self.name}: 云端发布完成，准备验证")

class TrainTailiStepAgent(_TailiStepBaseAgent):
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
        start_info = start_remote_training(host, port, user, password, str(ctx.session.state[STATE_P2_TRAIN_COMMAND]), self.cfg.remote_timeout_seconds)
        ctx.session.state["phase2.train.pid"] = start_info["pid"]
        ctx.session.state["phase2.train.log_path"] = start_info["log_path"]
        ctx.session.state["phase2.train.log_offset"] = 0
        ctx.session.state[STATE_P2_TRAIN_STATUS] = "running"

        last_evaluated_iteration = 0
        last_output = ""
        last_offset = 0
        sleep_seconds = 300.0
        current_check_interval = 20
        metric_history: list[dict] = []
        ctx.session.state["phase2.train.metric_history"] = metric_history

        warmup_hit = False
        for _ in range(3):
            await asyncio.sleep(20)
            current_output = remote_tail_log(host, port, user, password, start_info["log_path"], self.cfg.remote_timeout_seconds)
            if "Learning iteration" in current_output and "Iteration time:" in current_output:
                warmup_hit = True
                match = re.search(r"Iteration time:\s*([\d.]+)s", current_output, flags=re.IGNORECASE)
                if match:
                    try:
                        iteration_time = float(match.group(1))
                        sleep_seconds = max(20.0, min(500.0, iteration_time * float(self.cfg.eval_check_interval)))
                    except ValueError:
                        sleep_seconds = 300.0
                else:
                    sleep_seconds = 300.0
                break
        if not warmup_hit:
            sleep_seconds = 300.0

        max_checks = max(1, int((self.cfg.max_training_minutes * 60) // max(1.0, sleep_seconds)))
        for _ in range(max_checks):
            current_output = remote_tail_log(host, port, user, password, start_info["log_path"], self.cfg.remote_timeout_seconds).strip()
            if not current_output:
                await asyncio.sleep(sleep_seconds)
                continue
            current_offset = len(current_output)
            if current_offset <= last_offset:
                await asyncio.sleep(sleep_seconds)
                continue
            new_text = current_output[last_offset:]
            last_offset = current_offset
            recent_window = self._extract_recent_iteration_window(new_text, window_size=5)
            blocks = self._extract_checkpoint_blocks(recent_window)
            if not blocks:
                await asyncio.sleep(sleep_seconds)
                continue
            checkpoint_index, checkpoint_block = blocks[-1]
            if checkpoint_index <= last_evaluated_iteration or checkpoint_block == last_output:
                await asyncio.sleep(sleep_seconds)
                continue
            last_evaluated_iteration = checkpoint_index
            last_output = checkpoint_block
            ctx.session.state["phase2.train.current_checkpoint_output"] = checkpoint_block
            ctx.session.state["phase2.train.current_checkpoint_index"] = checkpoint_index
            ctx.session.state["phase2.train.last_evaluated_iteration"] = checkpoint_index
            ctx.session.state["phase2.train.recent_iteration_window"] = recent_window
            metrics_dict = self._extract_metrics_dict(checkpoint_block, checkpoint_index)
            metric_history.append(metrics_dict)
            ctx.session.state["phase2.train.metric_history"] = metric_history
            self._add_log(ctx, f"[{self.name}] checkpoint={checkpoint_index} captured")
            yield self._yield_text(recent_window)

            judge_input = {
                "metric_history": metric_history,
                "iteration_round": int(ctx.session.state.get(STATE_P2_ITER_ROUND, 0)),
                "check_interval": int(ctx.session.state.get(STATE_P2_EVAL_CHECK_INTERVAL, self.cfg.eval_check_interval)),
                "current_check_interval": current_check_interval,
                "sleep_seconds": sleep_seconds,
                "metric_specs": self.cfg.eval_metric_specs,
                "early_stop_rules": self.cfg.eval_early_stop_rules,
            }
            ctx.session.state["phase2.train.log_input_payload"] = judge_input
            prompt_text = json.dumps(judge_input, ensure_ascii=False)
            await ctx.session_service.append_event(
                ctx.session,
                Event(
                    author="system_context",
                    content=types.Content(role="user", parts=[types.Part(text=f"请基于以下输入执行你的任务:\n{prompt_text}")]),
                ),
            )
            async for judge_event in self.evaluate_training_log.run_async(ctx):
                yield judge_event
            judge_result = ctx.session.state.get("phase2.train.log_judge_result", {}) or {}
            action = str(judge_result.get("action", "continue"))
            if action == "stop_failed":
                remote_kill_process(host, port, user, password, str(start_info["pid"]), self.cfg.remote_timeout_seconds)
                ctx.session.state[STATE_P2_TRAIN_EARLY_STOPPED] = True
                ctx.session.state[STATE_P2_TRAIN_STATUS] = "early_stopped"
                ctx.session.state[STATE_P2_EVAL_PASSED] = False
                ctx.session.state["phase2.train.early_stop_reason"] = "; ".join(judge_result.get("reasons", []))
                yield self._yield_text(f"{self.name}: early stop executed")
                return
            if action == "stop_converged":
                remote_kill_process(host, port, user, password, str(start_info["pid"]), self.cfg.remote_timeout_seconds)
                ctx.session.state[STATE_P2_TRAIN_EARLY_STOPPED] = False
                ctx.session.state[STATE_P2_TRAIN_STATUS] = "completed"
                ctx.session.state[STATE_P2_EVAL_PASSED] = True
                ctx.session.state["phase2.train.converged"] = True
                ctx.session.state["phase2.train.should_stop"] = True
                yield self._yield_text(f"{self.name}: training converged, moving to play")
                break
            current_check_interval = min(500, max(current_check_interval + 1, current_check_interval * 2 if current_check_interval < 60 else current_check_interval))
            await asyncio.sleep(sleep_seconds)
        else:
            ctx.session.state[STATE_P2_TRAIN_STATUS] = "completed"

        yield self._yield_text(f"{self.name}: 开始渲染评估视频...")
        play_cmd = self.cfg.play_command_template.format(task_name=self.cfg.task_name)
        if self.cfg.play_num_envs is not None:
            play_cmd = f"{play_cmd} --num_envs={self.cfg.play_num_envs}"
        play_out, play_err, play_code = execute_ssh_command(
            host=host,
            port=port,
            username=user,
            password=password,
            command=f"cd /root/robot_lab && {play_cmd}",
            timeout_seconds=300,
        )
        ctx.session.state["phase2.play.command"] = play_cmd
        ctx.session.state["phase2.play.stdout"] = play_out
        ctx.session.state["phase2.play.stderr"] = play_err
        ctx.session.state["phase2.play.exit_code"] = play_code
        ctx.session.state[STATE_P2_TRAIN_STATUS] = "completed"
        if play_code != 0:
            self._add_log(ctx, f"视频渲染失败: {play_err or play_out}")
            ctx.session.state["phase2.play.failed"] = True
        else:
            ctx.session.state["phase2.play.failed"] = False
        yield self._yield_text(f"{self.name}: training completed, play rendered")


class EvaluateTailiTrainingLogAgent(_TailiStepBaseAgent):
    cfg: TailiCloudConfig
    model: str = "gemini-2.5-flash"
    description: str = "Evaluates full metric history and decides whether to stop."
    instruction: str = (
        "你是 Taili 的强化学习训练裁判。你必须只输出严格 JSON。\n"
        "你会接收到一个 metric_history 数组，包含了训练从开始到现在的全量指标历史（每次采样的所有 loss 和 reward）。\n"
        "你的任务是进行单次、无状态的趋势判定。\n"
        "判定原则：\n"
        "1. 发散/崩溃 (stop_failed)：观察最新采样的各项 Loss 和 Reward，如果出现 NaN/Inf，或者 reward 在探索几百步后仍然长期为大负数无上升趋势，立即返回 early stop。\n"
        "2. 收敛 (stop_converged)：观察全局历史。如果 Mean reward 已经较高且在最近的多次大跨度采样中几乎停滞（涨幅极微），且各 Loss（如 surrogate loss, value loss）已稳定，说明策略已收敛，为了节约算力，返回 early stop。\n"
        "3. 正常 (continue)：如果仍在快速上升期或正常探索震荡期，返回 continue。\n"
        "输出 JSON: { 'action': 'continue'|'stop_failed'|'stop_converged', 'score': {...}, 'reasons': [...], 'confidence': 0.0 }\n"
    )
    output_schema: ClassVar[Any] = TailiTrainingLogJudgeResult
    output_key: str = "phase2.train.log_judge"
    model_config = {"arbitrary_types_allowed": True}

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        ctx.session.state[STATE_P2_STAGE] = Phase2Stage.EVALUATE_TRAIN_LOG
        payload = {
            "metric_history": list(ctx.session.state.get("phase2.train.metric_history", [])),
            "iteration_round": int(ctx.session.state.get(STATE_P2_ITER_ROUND, 0)),
            "metric_specs": self.cfg.eval_metric_specs,
            "early_stop_rules": self.cfg.eval_early_stop_rules,
        }
        ctx.session.state["phase2.train.log_input_payload"] = payload
        prompt_text = f"请基于以下完整事实执行你的任务:\n{json.dumps(payload, ensure_ascii=False)}"
        result = UnifiedLLMClient().generate_json(
            system_prompt=self.instruction,
            user_prompt=prompt_text,
            schema=TailiTrainingLogJudgeResult,
        )
        ctx.session.state["phase2.train.log_judge_result"] = result.model_dump()
        if result.action in {"stop_failed", "stop_converged"}:
            ctx.session.state["phase2.train.should_stop"] = True
        ctx.session.state["phase2.train.last_checkpoint_index"] = ctx.session.state.get("phase2.train.current_checkpoint_index")
        ctx.session.state["phase2.train.last_checkpoint_output"] = ctx.session.state.get("phase2.train.current_checkpoint_output", "")
        yield self._yield_text(result.model_dump_json(indent=2))


class EvaluateTailiVideoAgent(_TailiStepBaseAgent):
    cfg: TailiCloudConfig
    model: str = "gemini-2.5-flash"
    description: str = "Evaluates the played video file and decides final pass/fail."
    instruction: str = (
        "你是 Taili 的视频裁判。你必须只输出严格 JSON，不要输出任何 Markdown、解释性前缀或多余文本。\n"
        "你只看视频文件及其上下文，不要把训练日志当成最终结论依据。\n"
        "输出必须符合如下 JSON 结构：\n"
        "{\n"
        '  "passed": boolean,\n'
        '  "score": object,\n'
        '  "reasons": [string, ...],\n'
        '  "next_action": "archive" | "revise",\n'
        '  "evidence_mode": string\n'
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
        ctx.session.state["phase2.video.remote_path"] = video_remote_path

        # TODO(P2-3): 当前视频评估是"元信息裁判"模式——下载视频到本地但只把路径等
        # 元信息传给 LLM，并不传入视频帧。原因是 DeepSeek API 暂不支持视频多模态输入。
        # 后续扩展方向：使用 Gemini 的视频理解能力，或本地抽帧后传入图片序列。
        payload = {
            "video_remote_path": video_remote_path,
            "video_local_path": video_local_path,
            "checkpoint_path": ctx.session.state.get(STATE_P2_EVAL_CHECKPOINT_PATH),
            "train_run_id": ctx.session.state.get(STATE_P2_TRAIN_RUN_ID),
            "play_command": ctx.session.state.get("phase2.play.command"),
            "note": "当前为元信息评估模式，LLM 无法直接观看视频内容",
        }
        ctx.session.state["phase2.video.input_payload"] = payload
        prompt_text = f"请基于以下完整事实执行你的任务:\n{json.dumps(payload, ensure_ascii=False)}"
        result = UnifiedLLMClient().generate_json(
            system_prompt=self.instruction,
            user_prompt=prompt_text,
            schema=TailiVideoJudgeResult,
        )
        ctx.session.state["phase2.video.judge_result"] = result.model_dump()

        # 关键：把视频评估结果写回编排器期望的 state 键，否则编排器永远认为评估没通过。
        ctx.session.state[STATE_P2_EVAL_PASSED] = result.passed
        ctx.session.state[STATE_P2_EVAL_SCORE] = result.score
        if result.passed:
            ctx.session.state[STATE_P2_EVAL_FAIL_REASONS] = []
            ctx.session.state[STATE_P2_HITL_REQUIRED] = False
            ctx.session.state[STATE_P2_HITL_REASON] = ""
        else:
            ctx.session.state[STATE_P2_EVAL_FAIL_REASONS] = result.reasons
            ctx.session.state[STATE_P2_HITL_REQUIRED] = True
            ctx.session.state[STATE_P2_HITL_REASON] = result.reasons[0] if result.reasons else "视频评估未通过"
        self._add_log(ctx, f"[{self.name}] 视频评估完成 passed={result.passed}")
        yield self._yield_text(result.model_dump_json(indent=2))


class RepairTailiWorkflowStepAgent(_TailiStepBaseAgent):
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        ctx.session.state[STATE_P2_STAGE] = Phase2Stage.ITERATE_TUNING
        ctx.session.state[STATE_P2_ITER_ROUND] = int(ctx.session.state.get(STATE_P2_ITER_ROUND, 0)) + 1
        ctx.session.state[STATE_P2_HITL_RESPONSE] = self.cfg.hitl_response_text or ""
        ctx.session.state[STATE_P2_HITL_RESOLVED] = bool(self.cfg.hitl_response_text)
        ctx.session.state[STATE_P2_HITL_REQUIRED] = False if self.cfg.hitl_response_text else True
        ctx.session.state[STATE_P2_CONFIG_LAST_REASON] = str(ctx.session.state.get(STATE_P2_HITL_REASON, ctx.session.state.get(STATE_P2_CONFIG_LAST_REASON, "")))
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
