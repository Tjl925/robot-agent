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
import posixpath
import re
from pathlib import Path
from typing import Any, AsyncGenerator, ClassVar
import base64

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types

from robot_agent.schemas.config import (
    TailiCloudConfig,
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
    STATE_P2_ITER_MAX,
    STATE_P2_PLAY_EXIT_CODE,
    STATE_P2_PLAY_FAILED,
    STATE_P2_PLAY_STDERR,
    STATE_P2_PLAY_STDOUT,
    STATE_P2_STAGE,
    STATE_P2_STATUS,
    STATE_P2_TRAIN_COMMAND,
    STATE_P2_TRAIN_RESUME_INFO,
    STATE_P2_TRAIN_RESUME_SOURCE,
    STATE_P2_TRAIN_ITER_SECONDS,
    STATE_P2_TRAIN_ACCEPTANCE,
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
    STATE_P2_PLAY_EVAL_RESULTS,
    STATE_P2_VIDEO_JUDGE_SUMMARY,
    STATE_P2_FAILED_TERRAINS,
    STATE_P2_FAILURE_TAGS,
    STATE_P2_CHECKPOINT_HISTORY,
    STATE_P2_BEST_CHECKPOINT,
    Phase2Stage,
)
from robot_agent.tools.llm_client import LLMCallConfig, UnifiedLLMClient
from robot_agent.tools import taili_cloud, taili_cloud_windows


class _TailiStepBaseAgent(BaseAgent):
    cfg: TailiCloudConfig
    model_config = {"arbitrary_types_allowed": True}

    @property
    def _remote(self):
        """按 remote_platform 选择远端命令执行后端：windows -> pwsh 后端，否则 Linux 后端。

        两个后端函数签名逐一对齐，调用点统一写 self._remote.xxx(...) 即可无缝切换。
        """
        if getattr(self.cfg, "remote_platform", "linux") == "windows":
            return taili_cloud_windows
        return taili_cloud

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

    def _get_experiment_name(self) -> str:
        """根据当前任务（flat 或 rough）从 rsl_rl_ppo_cfg.py 中解析出对应的 experiment_name"""
        
        local_robot_root = Path(self.cfg.local_robot_root)
        ppo_cfg_path = local_robot_root / ".taili_generated" / "agents" / "rsl_rl_ppo_cfg.py"
        experiment_name = None
        
        is_flat = "flat" in self.cfg.task_name.lower()
        target_class = "TailiQuadFlatPPORunnerCfg" if is_flat else "TailiQuadRoughPPORunnerCfg"
        
        if ppo_cfg_path.exists():
            content = ppo_cfg_path.read_text(encoding="utf-8")
            match = re.search(rf"class {target_class}.*?experiment_name\s*=\s*['\"]([^'\"]+)['\"]", content, re.DOTALL)
            if not match:
                match = re.search(rf"class {target_class}.*?self\.experiment_name\s*=\s*['\"]([^'\"]+)['\"]", content, re.DOTALL)
            if match:
                experiment_name = match.group(1)

        if not experiment_name:
            experiment_name = "taili_quad_flat" if is_flat else "taili_quad_rough"

        return experiment_name

    def _grade_training_acceptance(self, metric_history: list) -> dict:
        """从训练日志 metric_history 抽取最后一拍的"最终验收"指标，对照目标打分。

        【重要边界】这只是给人看的验收快照（达到了你就知道差不多了），绝不参与训练早停决策——
        日志裁判仍只按趋势/发散/Reward-Hacking 判 continue/stop，不会因为没够到这些绝对值而死等。
        指标键用子串匹配（terrain_levels / error_vel_xy / error_vel_yaw），对日志前缀/格式变化更鲁棒；
        没抓到的项 met=False 并把 value 置 None，方便你一眼看出是"没达到"还是"日志里没这项"。
        """
        cfg = self.cfg
        if not getattr(cfg, "accept_metrics_enabled", True):
            return {}
        last_sample: dict = {}
        for window in reversed(metric_history or []):
            samples = window.get("samples") if isinstance(window, dict) else None
            if samples and isinstance(samples[-1], dict):
                last_sample = samples[-1]
                break

        def _find(substr: str):
            for k, v in last_sample.items():
                if substr.lower() in str(k).lower() and isinstance(v, (int, float)):
                    return float(v)
            return None

        tl, exy, eyaw = _find("terrain_levels"), _find("error_vel_xy"), _find("error_vel_yaw")
        checks = {
            "terrain_levels": {"value": tl, "target": f">= {cfg.accept_min_terrain_levels}",
                               "met": tl is not None and tl >= cfg.accept_min_terrain_levels},
            "error_vel_xy": {"value": exy, "target": f"<= {cfg.accept_max_error_vel_xy}",
                             "met": exy is not None and exy <= cfg.accept_max_error_vel_xy},
            "error_vel_yaw": {"value": eyaw, "target": f"<= {cfg.accept_max_error_vel_yaw}",
                              "met": eyaw is not None and eyaw <= cfg.accept_max_error_vel_yaw},
        }
        return {
            "checks": checks,
            "all_met": all(c["met"] for c in checks.values()),
            "note": "训练级验收指标（仅参考，不参与训练早停）。terrain_levels 达标=已爬到高难度地形；error_vel_xy/yaw 达标=速度跟踪足够准。",
        }

    @staticmethod
    def _acceptance_signals(acceptance: dict | None) -> tuple[bool, float, float]:
        """从验收快照里抽出用于排序的三个信号：(全部达标, terrain_levels, error_vel_xy)。

        terrain_levels 越大越好（缺失记 -inf）；error_vel_xy 越小越好（缺失记 +inf）。
        这样'有训练指标'的记录总能排在'没有任何指标'的记录之前，便于 timeout 轮与 completed 轮公平比较。
        """
        acc = acceptance or {}
        checks = acc.get("checks", {}) if isinstance(acc, dict) else {}
        tl = checks.get("terrain_levels", {}).get("value") if isinstance(checks, dict) else None
        exy = checks.get("error_vel_xy", {}).get("value") if isinstance(checks, dict) else None
        return (
            bool(acc.get("all_met")),
            float(tl) if isinstance(tl, (int, float)) else float("-inf"),
            float(exy) if isinstance(exy, (int, float)) else float("inf"),
        )

    @classmethod
    def _champion_rank_key(cls, record: dict) -> tuple:
        """统一的'冠军'排序键（越大越优），续训源与交付 best 共用同一标准：
        1) 四地形视频是否全过（金标准）
        2) 视频通过的地形数（越多越好）
        3) 视频综合分（越高越好）
        4) terrain_levels 越大越好（仅作同分兜底）
        5) error_vel_xy 越小越好（仅作同分兜底）

        关键：视频证据（通过地形数 / 综合分）始终凌驾于 terrain_levels。terrain_levels 是课程进度，
        "碎步不摔倒照样爬课程"会把它刷高——正是碎步失败模式恰好能 game 的指标。若让它排在视频分前面，
        系统会专挑"碎步更狠但课程更高"的那版当冠军并续训接力，导致越练越差（41.25→30 下滑螺旋的根因）。
        故 terrain_levels/error_vel 降级为仅在视频表现完全打平时的兜底信号。
        无视频评估的轮（healthy-timeout 抢救轮）记 num_video_passed=-1，排在任何"有视频"轮之后。
        """
        all_met, tl, exy = cls._acceptance_signals(record.get("acceptance"))
        num_passed = record.get("num_video_passed")
        num_passed = int(num_passed) if isinstance(num_passed, (int, float)) else -1
        return (
            1 if record.get("overall_passed") else 0,
            num_passed,
            float(record.get("overall_score", 0.0) or 0.0),
            tl,
            -exy,
        )

    @classmethod
    def _is_champion_better(cls, new_rec: dict, cur_rec: dict | None) -> bool:
        """new_rec 是否严格优于现有冠军（cur_rec 为空时一律视为更优）。"""
        if not cur_rec:
            return True
        return cls._champion_rank_key(new_rec) > cls._champion_rank_key(cur_rec)

    def _promote_checkpoint(self, ctx: InvocationContext, record: dict) -> dict:
        """把一轮可采纳的 checkpoint 记录进历史，并按统一冠军标准更新"续训源"与"交付 best"。

        - record 至少含：round / run_dir / checkpoint_remote / status / acceptance / overall_passed / overall_score。
        - 续训源(RESUME_SOURCE)：更优即更新——下一轮 warm-start 从这里接力（含 healthy-timeout 轮）。
        - 交付 best(BEST_CHECKPOINT)：更优即下载物料到 logs/taili_best/ 保底，并写 BEST_MANIFEST.json。
        - 发散/失败轮由调用方决定不进来（=回退，保留上一轮已采纳源）。

        返回 {resume_updated, best_updated, bundle, best_record}，调用方据此打印自己的提示。
        """
        host, port = self.cfg.remote_host, self.cfg.remote_port
        user, password = self.cfg.remote_user, self.cfg.remote_password
        out = {"resume_updated": False, "best_updated": False, "bundle": None, "best_record": None}

        ckpt_remote = str(record.get("checkpoint_remote") or "")
        run_dir = str(record.get("run_dir") or "")

        # 1. 追加到 checkpoint 历史（每个可采纳轮都留痕，便于事后追溯任意一轮的远端 pt）
        history = list(ctx.session.state.get(STATE_P2_CHECKPOINT_HISTORY, []))
        history.append(record)
        ctx.session.state[STATE_P2_CHECKPOINT_HISTORY] = history

        if not ckpt_remote or not run_dir:
            return out  # 没拿到具体 checkpoint，仅留痕

        # 2. 续训源：优于现有即更新（与视频是否通过解耦，纯按统一冠军标准）
        cur_resume = ctx.session.state.get(STATE_P2_TRAIN_RESUME_SOURCE) or None
        if self._is_champion_better(record, cur_resume):
            ctx.session.state[STATE_P2_TRAIN_RESUME_SOURCE] = dict(record)
            out["resume_updated"] = True

        # 3. 交付 best：优于现有即下载物料保底
        cur_best = ctx.session.state.get(STATE_P2_BEST_CHECKPOINT) or None
        if self._is_champion_better(record, cur_best):
            best_local_dir = str(Path("logs") / "taili_best")
            bundle = self._remote.download_checkpoint_bundle(
                host, port, user, password, ckpt_remote, run_dir, best_local_dir, self.cfg.remote_timeout_seconds
            )
            best_record = {
                **record,
                "local_dir": bundle["local_dir"],
                "local_files": bundle["files"],
                "missing_artifacts": bundle["missing"],
            }
            try:
                (Path(bundle["local_dir"]) / "BEST_MANIFEST.json").write_text(
                    json.dumps(best_record, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            except Exception:
                pass
            ctx.session.state[STATE_P2_BEST_CHECKPOINT] = best_record
            out["best_updated"] = True
            out["bundle"] = bundle
            out["best_record"] = best_record
        return out


class AnalyzeTailiUrdfStepAgent(_TailiStepBaseAgent):
    cfg: TailiCloudConfig
    description: str = "Analyzes a URDF and returns a strict JSON diagnosis for training readiness."
    input_schema: ClassVar[Any] = dict
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

    def _build_input_payload(self, ctx: InvocationContext, urdf_path: Path, urdf_text: str) -> dict:
        return {
            "task_goal": "taili_quad 速度控制训练",
            "urdf_path": str(urdf_path),
            "urdf_text": urdf_text,
        }

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        ctx.session.state[STATE_P2_STAGE] = Phase2Stage.ANALYZE_URDF
        yield self._yield_text(f"[{self.name}] 正在评估 URDF 结构并分析可训练风险，请稍候...")
        urdf_path = Path(self.cfg.local_robot_root) / self.cfg.local_robots_subdir / "taili_quad.urdf"
        urdf_text = urdf_path.read_text(encoding="utf-8", errors="replace") if urdf_path.exists() else ""
        input_payload = self._build_input_payload(ctx, urdf_path, urdf_text)
        prompt_text = f"请基于以下完整事实执行你的任务:\n{json.dumps(input_payload, ensure_ascii=False)}"
        result = UnifiedLLMClient(LLMCallConfig(reasoning_effort="high")).generate_json(
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
        "当 mode=create 时，你要给出首版合理配置，且必须提供所有 6 个代码字段。\n"
        "当 mode=revise 时，为了节省成本，你只需输出修改后的全量代码（通常只有 rough_env_cfg_code 或 agents_ppo_cfg_code）。不需要修改的代码字段请直接在 JSON 中忽略（不要输出该键）。\n"
        "【极其重要】在 revise 模式下，你扮演的是超参优化与奖励函数塑形的角色。你只能修改奖励/惩罚权重、PPO 算法超参数、课程学习阈值等数值逻辑。严禁修改类名、继承关系、文件间 Import 或核心框架代码结构！\n"
        "你必须显式考虑：任务目标、URDF 诊断、参考模板、训练日志趋势、四地形视频评估结果、历史轨迹（特别是上一版的 failure_reason）。\n"
        "当 video_eval_summary 存在时，必须优先根据 failed_terrains、failure_tags、terrain_results 定位问题；不要只根据一个总分泛泛修改。\n"
        "修改必须遵守 allowed_edit_policy：只改白名单参数，每轮尽量少改，严禁越权重构代码。\n"
        "输出必须符合如下 JSON 结构：\n"
        "{\n"
        '  "mode": "create" | "revise",\n'
        '  "version": integer,\n'
        '  "parent_version": integer | null,\n'
        '  "task_name": string,\n'
        '  "reasoning": string,\n'
        '  "asset_code": string (可选),\n'
        '  "agents_init_code": string (可选),\n'
        '  "agents_ppo_cfg_code": string (可选),\n'
        '  "task_init_code": string (可选),\n'
        '  "flat_env_cfg_code": string (可选),\n'
        '  "rough_env_cfg_code": string (可选)\n'
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

    def _build_prompt_payload(self, ctx: InvocationContext) -> dict:
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
            
        video_eval_summary = ctx.session.state.get(STATE_P2_VIDEO_JUDGE_SUMMARY)

        allowed_edit_policy = {
            "allowed_files": ["rough_env_cfg.py", "agents/rsl_rl_ppo_cfg.py"],
            "rough_env_cfg_allowed_params": [
                "track_lin_vel_xy_exp.weight", "track_ang_vel_z_exp.weight",
                "lin_vel_z_l2.weight", "ang_vel_xy_l2.weight", "flat_orientation_l2.weight",
                "base_height_l2.weight", "joint_torques_l2.weight", "joint_acc_l2.weight",
                "joint_pos_limits.weight", "joint_power.weight", "stand_still.weight",
                "joint_pos_penalty.weight", "joint_mirror.weight", "action_rate_l2.weight",
                "undesired_contacts.weight", "contact_forces.weight", "feet_air_time.weight",
                "feet_contact.weight", "feet_contact_without_cmd.weight", "feet_stumble.weight",
                "feet_slide.weight", "feet_height.weight", "feet_height_body.weight",
                "feet_gait.weight", "upward.weight",
                "terrain_generator.sub_terrains.*.step_height_range",
                "terrain_generator.sub_terrains.*.grid_height_range",
                "terrain_generator.sub_terrains.*.noise_range",
            ],
            "ppo_allowed_params": [
                "learning_rate", "entropy_coef", "desired_kl", "num_learning_epochs",
                "num_mini_batches", "max_grad_norm", "max_iterations",
            ],
            "edit_limits": [
                "每轮最多修改 3~5 个关键参数",
                "reward weight 单轮通常不超过 20%~50% 的相对变化，除非失败原因极其明确",
                "PPO 参数单轮最多按 0.5x 或 2x 调整",
                "禁止修改类名、继承关系、注册名、实验名(experiment_name)、import 路径、机器人 joint/body 名称和核心框架结构（experiment_name 一旦变动会导致续训无法定位上一轮 checkpoint）",
                "【续训防发散·warm_start_pending=true 时强制】本轮会从上一轮已收敛的 checkpoint 热启动续训："
                "(1) 必须把 learning_rate 降到上一轮的 0.3~0.5x（如 1e-3 -> 3e-4~5e-4），避免大步更新打散已学策略导致发散；"
                "(2) 禁止把此前权重=0 的奖励项一次性提到较大正值（如 feet_gait 0 -> 0.5）——应拆成多轮小幅渐进（先 0 -> 0.1~0.2）；"
                "(3) reward 权重单轮相对变化收紧到≤20%；不连续的奖励地形突变是续训发散的首要诱因。",
                "max_iterations 不要盲目调大：单轮训练有墙钟时间预算，超出预算的部分系统会自动用 --max_iterations 截断，"
                "设过大不会让你多训、只会让你误判进度。如需更多训练步数，靠多轮续训累积，而不是单轮设超大 max_iterations。",
            ],
            # failure_tags / achieved_metrics 异常 -> 建议优先调整的奖励旋钮（仍需结合具体数值与历史，按需取舍）
            "failure_tag_to_knob_hints": {
                "high_step_frequency": "碎步/高频踏步：把 feet_gait.weight 从 0 提到正值(如 0.5~1.0，强制对角 trot)；增大 feet_air_time.weight 鼓励更长摆动/更大步；增大 action_rate_l2 惩罚(如 -0.01 -> -0.02~-0.03)与 joint_acc_l2 惩罚使动作更平滑。",
                "short_stride": "步幅过短/拖步：增大 feet_air_time.weight 与其 threshold；可适当增大 feet_height/feet_height_body 鼓励抬脚；配合 feet_gait.weight 提升步态规整。",
                "body_bounce": "上下颠簸/跳：增大 lin_vel_z_l2、flat_orientation_l2 惩罚；启用/增大 base_height_l2.weight 稳定机身高度。",
                "velocity_tracking": "速度跟随差：增大 track_lin_vel_xy_exp.weight / track_ang_vel_z_exp.weight；检查 stand_still / joint_pos_penalty 惩罚是否过强压制了运动。",
                "fall_or_out_of_bounds": "摔倒/出界：适当增大 flat_orientation_l2、undesired_contacts 惩罚；检查是否惩罚过激导致摆烂，必要时温和回调能量类惩罚。",
                "gait_coordination": "步态不协调：提升 feet_gait.weight、joint_mirror.weight 促进对称协调。",
                "foot_clearance": "抬脚不足/拖地：主调 feet_height —— 提高 feet_height.params['target_height'](0.05->0.07~0.09)并增大 feet_height.weight(0.2->0.35~0.6)放大抬脚激励；若 feet_height_body.weight 绝对值过大(-3.0)把脚压在机身下，适当减小到 -1.5 给抬脚让路。参考 gait.swing_clearance.* 数值。",
                "foot_slip": "触地打滑/拖滑：核心是 feet_slide.weight 当前=0(完全没罚足端滑移)，小步开成负值 0->-0.05~-0.1(续训须遵守防发散:禁0一次提大值、单轮≤20%、降LR)；辅以增大 feet_air_time.weight 鼓励'抬-落'而非贴地蹭。参考 gait.foot_slip_*。",
                "gait_asymmetry": "步态不对称/跛行：提升 joint_mirror.weight 强制左右/对角对称、feet_gait.weight 规整对角 trot；检查是否某腿被 joint_pos_limits/joint_pos_penalty 过度压制。参考 gait.foot_touchdown_cv/diag_pair_touchdown_diff/lamest_foot 定位坏腿。",
                "contact_impact": "落地砸地/硬着陆：主调 contact_forces.weight(增大绝对值,如 -1.5e-4->-3e-4)与降低其 threshold 更早惩罚大接触力；辅以增大 lin_vel_z_l2、feet_air_time 鼓励更软的落地。参考 posture.impact.p95_touchdown_grf_bw/hard_landing_ratio。",
            },
        }

        # 下一轮训练是否会 warm-start 续训（据此提醒 LLM 温和改参防发散，见 edit_limits 续训硬规则）
        resume_src = ctx.session.state.get(STATE_P2_TRAIN_RESUME_SOURCE) or ctx.session.state.get(STATE_P2_BEST_CHECKPOINT) or {}
        warm_start_pending = bool(getattr(self.cfg, "resume_from_best", True) and resume_src.get("checkpoint_remote"))
        warm_start_context = {
            "warm_start_pending": warm_start_pending,
            "from_round": resume_src.get("round"),
            "from_status": resume_src.get("status"),
            "note": "本轮改完参数后将从该 checkpoint 续训；务必遵守 edit_limits 的续训防发散硬规则（降 LR、禁突变、收紧幅度）。" if warm_start_pending else "本轮从零训练，无续训发散风险。",
        }

        return {
            "mode": mode,
            "version": version_index + 1,
            "parent_version": version_index,
            "version_note": "version/parent_version 由系统统一管理并保证与轮次一致，你无需纠结其取值，按本字段给定值理解'这是第几轮修订'即可。",
            "warm_start_context": warm_start_context,
            "task_goal": "taili_quad 速度控制训练",
            "urdf_risk": risk,
            "urdf_issues": ctx.session.state.get(STATE_P2_URDF_ISSUES, []),
            "current_draft": current_draft,
            "history": history[-3:],
            "train_log_judge_result": ctx.session.state.get(STATE_P2_TRAIN_LOG_JUDGE_RESULT),
            "video_eval_summary": video_eval_summary,
            "failed_terrains": ctx.session.state.get(STATE_P2_FAILED_TERRAINS, []),
            "failure_tags": ctx.session.state.get(STATE_P2_FAILURE_TAGS, []),
            "allowed_edit_policy": allowed_edit_policy,
            "reference_templates": reference_templates,
        }

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        ctx.session.state[STATE_P2_STAGE] = Phase2Stage.SYNTHESIZE_CONFIG
        yield self._yield_text(f"[{self.name}] 正在生成核心配置草案，请稍候...")
        payload = self._build_prompt_payload(ctx)
        prompt_text = f"请基于以下完整事实执行你的任务:\n{json.dumps(payload, ensure_ascii=False)}"
        draft = UnifiedLLMClient().generate_json(
            system_prompt=self.instruction,
            user_prompt=prompt_text,
            schema=TailiConfigDraft,
        )
        
        new_version = int(ctx.session.state.get(STATE_P2_CONFIG_VERSION, 0)) + 1
        ctx.session.state[STATE_P2_CONFIG_VERSION] = new_version
        ctx.session.state[STATE_P2_CONFIG_PARENT_VERSION] = new_version - 1
        history = list(ctx.session.state.get(STATE_P2_CONFIG_HISTORY, []))
        summary = {
            "mode": draft.mode,
            "version": new_version,
            "parent_version": new_version - 1,
            "task_name": draft.task_name,
            "reasoning": draft.reasoning,
            "failure_reason": "",
        }
        history.append(summary)
        ctx.session.state[STATE_P2_CONFIG_HISTORY] = history
        ctx.session.state[STATE_P2_CONFIG_TEXT] = draft.model_dump_json(indent=2)
        ctx.session.state[STATE_P2_CONFIG_MODE] = draft.mode
        self._add_log(ctx, f"[{self.name}] 配置生成完成 mode={draft.mode} version={new_version}(parent={new_version - 1})")
        yield self._yield_text(f"[{self.name}] 配置生成完毕:\n{json.dumps(summary, ensure_ascii=False, indent=2)}")


class GenerateTailiFilesStepAgent(_TailiStepBaseAgent):
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        ctx.session.state[STATE_P2_STAGE] = Phase2Stage.GENERATE_FILES
        local_outputs = Path(self.cfg.local_robot_root) / ".taili_generated"
        local_outputs.mkdir(parents=True, exist_ok=True)
        (local_outputs / "agents").mkdir(parents=True, exist_ok=True)
        
        draft_json = ctx.session.state.get(STATE_P2_CONFIG_TEXT, "{}")
        draft = TailiConfigDraft.model_validate_json(draft_json)
        
        files_to_write = {}
        if draft.asset_code is not None:
            files_to_write["taili_quad.py"] = draft.asset_code
        if draft.agents_init_code is not None:
            files_to_write["agents/__init__.py"] = draft.agents_init_code
        if draft.agents_ppo_cfg_code is not None:
            files_to_write["agents/rsl_rl_ppo_cfg.py"] = draft.agents_ppo_cfg_code
        if draft.task_init_code is not None:
            files_to_write["__init__.py"] = draft.task_init_code
        if draft.flat_env_cfg_code is not None:
            files_to_write["flat_env_cfg.py"] = draft.flat_env_cfg_code
        if draft.rough_env_cfg_code is not None:
            files_to_write["rough_env_cfg.py"] = draft.rough_env_cfg_code
        
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

        uploaded = self._remote.remote_upload_taili_workspace(
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

    def _configured_max_iterations(self) -> int | None:
        """从本地已生成的 rsl_rl_ppo_cfg.py 解析 LLM 本轮设定的 max_iterations（解析失败返回 None）。"""
        p = Path(self.cfg.local_robot_root) / ".taili_generated" / "agents" / "rsl_rl_ppo_cfg.py"
        if not p.exists():
            return None
        try:
            content = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None
        is_flat = "flat" in self.cfg.task_name.lower()
        target_class = "TailiQuadFlatPPORunnerCfg" if is_flat else "TailiQuadRoughPPORunnerCfg"
        m = re.search(rf"class {target_class}.*?max_iterations\s*=\s*(\d+)", content, re.DOTALL)
        if not m:
            m = re.search(r"max_iterations\s*=\s*(\d+)", content)  # 兜底：全局最后一处
        try:
            return int(m.group(1)) if m else None
        except (TypeError, ValueError):
            return None

    def _apply_iter_budget_cap(self, ctx: InvocationContext, cmd: str) -> str:
        """按时间预算截断 max_iterations：用上一轮实测每步耗时估算本轮墙钟内能跑的步数，超出则用原生 --max_iterations 截断。

        目的：避免 LLM 把 max_iterations 设到墙钟时间内根本跑不完的值（如 10000），导致训练被超时杀死、整轮浪费。
        截断后训练能自然 completed → 正常走视频评估与冠军留存。首轮（无实测耗时）不截断，保留原配置。

        关键：依据 cloud/train.py:224 `runner.learn(num_learning_iterations=agent_cfg.max_iterations)`，rsl_rl 的
        max_iterations 是"本次新增的迭代步数"（tot_iter = 续训起点 + max_iterations），本次墙钟 ≈ max_iterations × 每步耗时，
        与续训起点无关。因此可达步数只按时间预算算，不涉及起点。
        """
        if not getattr(self.cfg, "iter_budget_cap_enabled", True):
            return cmd
        iter_seconds = ctx.session.state.get(STATE_P2_TRAIN_ITER_SECONDS)
        if not iter_seconds or float(iter_seconds) <= 0:
            return cmd  # 首轮无实测，不截断
        reachable = int(self.cfg.max_training_minutes * 60.0 / float(iter_seconds) * float(self.cfg.iter_budget_safety_ratio))
        if reachable <= 0:
            return cmd
        configured = self._configured_max_iterations()
        # 本就可达则不动；超出预算或无法解析配置才截断到可达步数
        if configured is not None and configured <= reachable:
            return cmd
        return f"{cmd} --max_iterations {reachable}"

    def _build_train_command(self, ctx: InvocationContext) -> tuple[str, dict | None]:
        """构建本轮训练命令。

        - create / 无历史最优 / 关闭续训：返回基础命令，从零训练。
        - revise 且存在历史最优 checkpoint：追加 rsl_rl 续训参数，从"历史最优"那一轮的精确
          checkpoint 热启动（注意是历史最优，不一定是上一轮 latest）。

        依据 cloud/train.py + cloud/cli_args.py：--resume 是 store_true 裸旗标（不带值），出现即令
        agent_cfg.resume=True，再用 get_checkpoint_path(log_root, load_run, load_checkpoint) 定位
        checkpoint 并 runner.load。因此续训只追加 --resume --load_run <run 基名> --checkpoint <model_*.pt 基名>。
        切勿写成 --resume True/False：多余的值会被 parse_known_args 漏给 Hydra 当覆盖参数而报错；
        create 轮直接省略 --resume（cli_args 会把 agent_cfg.resume 置 False，从零训练）。
        """
        base = self.cfg.train_command_template.format(task_name=self.cfg.task_name)
        if not getattr(self.cfg, "resume_from_best", True):
            return self._apply_iter_budget_cap(ctx, base), None

        # 续训源优先级：RESUME_SOURCE（与"视频最优"解耦的训练侧续训源指针，含 healthy-timeout 轮）
        # → BEST_CHECKPOINT（视频最优，向后兼容）→ 从零。
        # 这样发散/超时轮的成果只要被采纳进 RESUME_SOURCE，下一轮就能真正接力，而不是永远回退到首轮弱模型。
        src = ctx.session.state.get(STATE_P2_TRAIN_RESUME_SOURCE) or ctx.session.state.get(STATE_P2_BEST_CHECKPOINT) or {}
        ckpt_remote = str(src.get("checkpoint_remote") or "")
        run_dir = str(src.get("run_dir") or "")
        if not ckpt_remote or not run_dir:
            return self._apply_iter_budget_cap(ctx, base), None  # 通常是第 0 轮 create：尚无可续之源

        host, port = self.cfg.remote_host, self.cfg.remote_port
        user, password = self.cfg.remote_user, self.cfg.remote_password
        # 续训前确认远端 checkpoint 仍存在，否则回退从零训练（避免 get_checkpoint_path 直接报错浪费一轮）
        if not self._remote.remote_file_exists(host, port, user, password, ckpt_remote, self.cfg.remote_timeout_seconds):
            return self._apply_iter_budget_cap(ctx, base), None

        run_basename = posixpath.basename(run_dir.rstrip("/"))
        ckpt_basename = posixpath.basename(ckpt_remote)
        resume_cmd = f"{base} --resume --load_run {run_basename} --checkpoint {ckpt_basename}"
        resume_cmd = self._apply_iter_budget_cap(ctx, resume_cmd)
        resume_info = {
            "resumed": True,
            "from_round": src.get("round"),
            "from_score": src.get("overall_score"),
            "from_status": src.get("status"),
            "load_run": run_basename,
            "checkpoint": ckpt_basename,
            "checkpoint_remote": ckpt_remote,
            "verified": None,
        }
        return resume_cmd, resume_info

    def _salvage_resume_checkpoint(self, ctx: InvocationContext, status: str) -> list[str]:
        """抢救"训练已产出有效 checkpoint、但不会走视频评估"那一类轮次（典型：healthy-timeout）。

        背景：超时轮被墙钟杀死前，往往已训出比上一轮更好、且仍在改善的 checkpoint（裁判常判 CONTINUE）。
        若直接 return 丢弃，则下一轮 warm-start 只能回退到更早的弱模型，整轮 GPU 算力清零、技能无法累积。
        这里到当前轮 run 目录抓最新 model_*.pt，连同训练验收指标一起按统一冠军标准更新续训源/交付 best。

        注意：发散(early_stopped)/失败(train_failed) 轮不调用本方法（=回退，保留上一轮已采纳源）。
        """
        msgs: list[str] = []
        host, port = self.cfg.remote_host, self.cfg.remote_port
        user, password = self.cfg.remote_user, self.cfg.remote_password

        # 验收快照（timeout 轮原本走不到下方的快照计算，这里补算并落盘，供归档/HITL 与冠军排序使用）
        metric_history = ctx.session.state.get(STATE_P2_TRAIN_METRIC_HISTORY) or []
        acceptance = self._grade_training_acceptance(metric_history)
        if acceptance:
            ctx.session.state[STATE_P2_TRAIN_ACCEPTANCE] = acceptance

        exp_root = f"{self.cfg.eval_log_root}/{self._get_experiment_name()}"
        run_dir = self._remote.remote_list_latest_run(host, port, user, password, exp_root, self.cfg.remote_timeout_seconds)
        ckpt = self._remote.remote_find_latest_checkpoint(host, port, user, password, run_dir, self.cfg.remote_timeout_seconds) if run_dir else ""
        if not run_dir or not ckpt:
            msgs.append(f"\033[93m[{self.name}] 本轮({status})未能在 {exp_root} 定位到最新 checkpoint，无法抢救续训源\033[0m")
            return msgs

        iter_round = int(ctx.session.state.get(STATE_P2_ITER_ROUND, 0))
        record = {
            "round": iter_round,
            "run_dir": run_dir,
            "checkpoint_remote": ckpt,
            "status": status,
            "acceptance": acceptance,
            "overall_passed": False,   # 未做视频评估
            "overall_score": 0.0,
            "num_video_passed": -1,    # 未做视频评估：冠军排序中排在任何"有视频"轮之后
            "failed_terrains": [],
        }
        res = self._promote_checkpoint(ctx, record)
        _all_met, _tl, _exy = self._acceptance_signals(acceptance)
        tag = []
        if res["resume_updated"]:
            tag.append("已采纳为下一轮续训源")
        if res["best_updated"]:
            tag.append(f"并刷新交付 best→{res['bundle']['local_dir']}")
        if not tag:
            tag.append("未优于现有续训源/最优，保留既有")
        self._add_log(ctx, f"[{self.name}] salvage {status} round={iter_round} terrain={_tl} err_xy={_exy} ckpt={posixpath.basename(ckpt)} -> {'/'.join(tag)}")
        msgs.append(
            f"\033[92m[{self.name}] 已抢救本轮({status})最新 checkpoint {posixpath.basename(ckpt)}"
            f"（terrain_levels={_tl}, error_vel_xy={_exy}）：{'，'.join(tag)}\033[0m"
        )
        return msgs

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        ctx.session.state[STATE_P2_STAGE] = Phase2Stage.RUN_TRAINING
        host = self.cfg.remote_host
        port = self.cfg.remote_port
        user = self.cfg.remote_user
        password = self.cfg.remote_password
        if not all([host, port, user, password]):
            raise RuntimeError("缺少远端 SSH 信息，无法启动训练")

        # 清空上一轮的视频判定状态：本轮只有真正完成视频评估才会重新写入。
        # 否则发散/超时轮会把上一轮的视频失败反馈(碎步细节/失败地形/标签)当成本轮证据，误导下一轮 ConfigSynthesis 改参。
        ctx.session.state[STATE_P2_VIDEO_JUDGE_SUMMARY] = None
        ctx.session.state[STATE_P2_FAILED_TERRAINS] = []
        ctx.session.state[STATE_P2_FAILURE_TAGS] = []

        # 每轮都重建训练命令：revise 轮若有历史最优 checkpoint 则 warm-start 续训，技能跨轮累积。
        train_command, resume_info = self._build_train_command(ctx)
        ctx.session.state[STATE_P2_TRAIN_COMMAND] = train_command
        ctx.session.state[STATE_P2_TRAIN_RESUME_INFO] = resume_info or {"resumed": False}
        if resume_info:
            self._add_log(ctx, f"[{self.name}] warm-start from round={resume_info['from_round']} {resume_info['load_run']}/{resume_info['checkpoint']}")
            yield self._yield_text(
                f"\033[92m[{self.name}] 本轮从历史最优 checkpoint 续训(warm-start)："
                f"round={resume_info['from_round']}, score={resume_info['from_score']}, "
                f"{resume_info['load_run']}/{resume_info['checkpoint']}\033[0m"
            )
        else:
            yield self._yield_text(f"[{self.name}] 本轮从零开始训练（首轮 create / 无可用最优 checkpoint / 已关闭续训）")

        # 【阶段 1：通过 tmux 会话下发训练指令】
        start_info = self._remote.start_remote_training(host, port, user, password, str(ctx.session.state[STATE_P2_TRAIN_COMMAND]), self.cfg.cloud_tmp_dir, self.cfg.remote_timeout_seconds)
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
        resume_requested = bool(resume_info)
        resume_checked = False

        # 【阶段 2：自适应预热与步长估算】
        warmup_hit = False
        yield self._yield_text(f"\033[36m[{self.name}] 进入训练启动自适应预热阶段，预计至多等待 300 秒以估算步长耗时...\033[0m")
        for i in range(3):
            yield self._yield_text(f"\033[90m[{self.name}] [预热轮次 {i+1}/3] 正在拉取远端启动日志以进行探活估算...\033[0m")
            await asyncio.sleep(100)
            warmup_text, byte_offset = self._remote.remote_tail_log(host, port, user, password, start_info["log_path"], self.cfg.remote_timeout_seconds, byte_offset)
            if "Learning iteration" in warmup_text and "Iteration time" in warmup_text:
                warmup_hit = True
                # 使用 findall 获取所有耗时特征，取最后一个匹配值（即最新稳态速度），完美规避第 0 轮虚高的初始化开销
                matches = re.findall(r"Iteration time:\s*([\d.]+)s", warmup_text, flags=re.IGNORECASE)
                if matches:
                    try:
                        iteration_time = float(matches[-1])
                        sleep_seconds = max(20.0, min(500.0, iteration_time * float(self.cfg.eval_check_interval)))
                        # 跨轮保存实测每步耗时：供下一轮 _build_train_command 按时间预算截断 max_iterations
                        ctx.session.state[STATE_P2_TRAIN_ITER_SECONDS] = iteration_time
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
            remote_status = self._remote.remote_check_training_status(host, port, user, password, train_session, train_exit_code_path, self.cfg.remote_timeout_seconds)
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
            new_text, byte_offset = self._remote.remote_tail_log(host, port, user, password, start_info["log_path"], self.cfg.remote_timeout_seconds, byte_offset)
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

            # 【续训确认】训练已确实在迭代，此时整文件 grep 校验 checkpoint 是否真的被加载，
            # 把"续训静默失效→白练一轮"变成立刻可见的强提示（仅校验一次）。
            if resume_requested and not resume_checked:
                resume_checked = True
                loaded = self._remote.remote_log_contains(
                    host, port, user, password, start_info["log_path"],
                    "Loading model checkpoint from", self.cfg.remote_timeout_seconds,
                )
                resume_state = dict(ctx.session.state.get(STATE_P2_TRAIN_RESUME_INFO, {}) or {})
                resume_state["verified"] = bool(loaded)
                ctx.session.state[STATE_P2_TRAIN_RESUME_INFO] = resume_state
                if loaded:
                    yield self._yield_text(f"\033[92m[{self.name}] ✓ 续训已确认：远端日志出现 'Loading model checkpoint from'，本轮确为 warm-start\033[0m")
                else:
                    yield self._yield_text(f"\033[1;91m[{self.name}] ⚠ 续训疑似未生效：已开始迭代但日志未见 checkpoint 加载行，本轮可能从零训练！请核对云端 train.py 的 cli_args 是否支持 --resume/--load_run/--checkpoint\033[0m")

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
                self._remote.remote_kill_training(host, port, user, password, train_session, self.cfg.remote_timeout_seconds)
                ctx.session.state[STATE_P2_TRAIN_STATUS] = "early_stopped"
                ctx.session.state[STATE_P2_EVAL_FAIL_REASON] = judge_result.get("reason", "")
                yield self._yield_text(f"{self.name}: 训练发散，已执行 early stop")
                return
            if action == "stop_converged":
                self._remote.remote_kill_training(host, port, user, password, train_session, self.cfg.remote_timeout_seconds)
                ctx.session.state[STATE_P2_TRAIN_STATUS] = "completed"
                yield self._yield_text(f"{self.name}: 训练已收敛，进入视频渲染")
                break
        else:
            # max_checks 到期：不允许直接标记 completed，必须再次检查远端状态。
            remote_status = self._remote.remote_check_training_status(host, port, user, password, train_session, train_exit_code_path, self.cfg.remote_timeout_seconds)
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
                self._remote.remote_kill_training(host, port, user, password, train_session, self.cfg.remote_timeout_seconds)
                ctx.session.state[STATE_P2_TRAIN_STATUS] = "train_timeout"
                ctx.session.state[STATE_P2_EVAL_FAIL_REASON] = f"训练超过 {self.cfg.max_training_minutes} 分钟仍未结束，已终止 tmux 会话"
                yield self._yield_text(f"{self.name}: 训练超时，已终止 tmux 会话")
                # 抢救：超时前往往已训出更好的 checkpoint（裁判常判 CONTINUE），不能整轮丢弃。
                for _m in self._salvage_resume_checkpoint(ctx, "train_timeout"):
                    yield self._yield_text(_m)
                return
            else:
                ctx.session.state[STATE_P2_TRAIN_STATUS] = "train_failed"
                ctx.session.state[STATE_P2_EVAL_FAIL_REASON] = "轮询到期，训练会话已不存在且未生成 exit_code"
                yield self._yield_text(f"{self.name}: 训练会话异常退出")
                return
                
        # 【训练级"最终验收"指标快照】仅供参考/验收，绝不参与上面的早停决策。
        acceptance = self._grade_training_acceptance(metric_history)
        if acceptance:
            ctx.session.state[STATE_P2_TRAIN_ACCEPTANCE] = acceptance
            _checks = acceptance.get("checks", {})
            _line = ", ".join(
                f"{k}={c.get('value')}(目标{c.get('target')},{'达标' if c.get('met') else '未达'})"
                for k, c in _checks.items()
            )
            yield self._yield_text(f"[{self.name}] 训练验收指标快照(仅参考): {_line} | 全部达标={acceptance.get('all_met')}")

        # 【阶段 6：四地形 play_eval 视频渲染流程】
        # 训练完成或早停收敛后，串行渲染 flat / boxes / stairs_down / stairs_up 四段评估视频。
        eval_terrains_raw = self.cfg.play_eval_terrains
        if isinstance(eval_terrains_raw, str):
            eval_terrains = [item.strip() for item in eval_terrains_raw.split(",") if item.strip()]
        else:
            eval_terrains = list(eval_terrains_raw)
        if not eval_terrains:
            eval_terrains = ["flat", "boxes", "stairs_down", "stairs_up"]

        play_timeout_seconds = self.cfg.play_eval_timeout_seconds
        play_eval_template = self.cfg.play_eval_command_template

        yield self._yield_text(
            f"{self.name}: 开始串行渲染 {len(eval_terrains)} 段评估视频: {', '.join(eval_terrains)}"
        )

        play_eval_results: dict[str, dict] = {}
        stdout_parts: list[str] = []
        for terrain in eval_terrains:
            play_cmd = play_eval_template.format(
                task_name=self.cfg.task_name,
                eval_terrain=terrain
            )
            yield self._yield_text(f"{self.name}: 正在渲染地形 {terrain}，命令: {play_cmd}")
            try:
                play_out, play_code = self._remote.remote_execute_play_in_tmux(
                    host, port, user, password,
                    train_session, play_cmd, self.cfg.cloud_tmp_dir, play_timeout_seconds,
                )
            except Exception as exc:
                ctx.session.state[STATE_P2_PLAY_STDOUT] = "\n\n".join(stdout_parts)
                ctx.session.state[STATE_P2_PLAY_STDERR] = str(exc)
                ctx.session.state[STATE_P2_PLAY_EXIT_CODE] = -1
                ctx.session.state[STATE_P2_PLAY_FAILED] = True
                ctx.session.state[STATE_P2_TRAIN_STATUS] = "play_failed"
                ctx.session.state[STATE_P2_PLAY_EVAL_RESULTS] = play_eval_results
                ctx.session.state[STATE_P2_EVAL_FAIL_REASON] = f"play_eval.py 渲染异常 terrain={terrain}: {exc}"
                yield self._yield_text(f"{self.name}: 地形 {terrain} 视频渲染异常: {exc}")
                return

            stdout_parts.append(f"===== terrain={terrain} exit={play_code} =====\n{play_out}")
            play_eval_results[terrain] = {
                "terrain": terrain,
                "command": play_cmd,
                "exit_code": play_code,
                "stdout_tail": play_out[-2000:],
            }
            if play_code != 0:
                self._add_log(ctx, f"视频渲染失败 terrain={terrain}: {play_out[:300]}")
                ctx.session.state[STATE_P2_PLAY_STDOUT] = "\n\n".join(stdout_parts)
                ctx.session.state[STATE_P2_PLAY_STDERR] = ""
                ctx.session.state[STATE_P2_PLAY_EXIT_CODE] = play_code
                ctx.session.state[STATE_P2_PLAY_FAILED] = True
                ctx.session.state[STATE_P2_TRAIN_STATUS] = "play_failed"
                ctx.session.state[STATE_P2_PLAY_EVAL_RESULTS] = play_eval_results
                ctx.session.state[STATE_P2_EVAL_FAIL_REASON] = f"play_eval.py 渲染失败 terrain={terrain} (exit={play_code}): {play_out[:200]}"
                yield self._yield_text(f"{self.name}: 地形 {terrain} 视频渲染失败 (exit={play_code})")
                return

        ctx.session.state[STATE_P2_PLAY_STDOUT] = "\n\n".join(stdout_parts)
        ctx.session.state[STATE_P2_PLAY_STDERR] = ""
        ctx.session.state[STATE_P2_PLAY_EXIT_CODE] = 0
        ctx.session.state[STATE_P2_PLAY_FAILED] = False
        ctx.session.state[STATE_P2_PLAY_EVAL_RESULTS] = play_eval_results
        yield self._yield_text(f"{self.name}: 训练完成，四地形评估视频已全部渲染")


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
        "2. 正常 continue：\n"
        "   - 核心奖励（特别是跟踪目标速度的奖励，如 tracking_lin_vel）或关键姿态奖励仍在上升/改善，尚未达到绝对瓶颈；\n"
        "   - 训练早期震荡属正常，或因迭代次数太少（如小于 1500 步且处于上升期）无法轻易断言已经完全收敛；\n"
        "   - 在不完全确定发散或绝对收敛前，请宽容并返回 continue。\n"
        "\n"
        "3. 收敛 stop_converged（极度严苛）：\n"
        "   - 性能极其优秀：Mean_reward 稳定在较高水平，且各项细分动作姿态惩罚项（如关节力矩惩罚、足端碰撞惩罚、高度抖动惩罚等）均已被最小化并达到稳态；\n"
        "   - 增长完全停滞：最近许多个 iteration 之间，所有核心及细节指标均已进入几乎绝对平坦的平台期（无上升空间）；\n"
        "   - 姿态健康：必须确保那些指示“姿态质量”的惩罚项没有发生恶化，走路姿势是优雅合理的，而非牺牲动作质量来换取高 reward；\n"
        "   - 只有在“各项细分性能极佳 + 姿态质量优秀 + 损失稳定”同时满足时，才返回 stop_converged。\n"
        "\n"
        "4. 局部最优/死锁 stop_failed（Reward Hacking 防御）：\n"
        "   - 【极其关键】如果机器狗为了逃避能量或姿态惩罚，选择了“躺平”或“僵直不动”：表现为 Episode Length 长时间存活（通常是1000），且总 Reward 停滞不前，但核心的目标跟踪奖励（如 tracking_lin_vel, tracking_ang_vel）几乎为 0 或极低。\n"
        "   - 这种看似“稳定”的状态是强化学习中典型的局部最优陷阱，绝不是真正的收敛！遇到这种情况，必须果断返回 stop_failed，以便外部系统能够介入并启动 revise（局部调参修改惩罚项权重）！\n"
        "\n"
        "输出必须是严格 JSON，格式如下：\n"
        "{\n"
        "  \"action\": \"continue\" | \"stop_failed\" | \"stop_converged\",\n"
        "  \"score\": {},\n"
        "  \"reason\": \"\"\n"
        "}\n"
        "\n"
        "score 中建议包含你关注到的关键姿态指标和整体奖励走势。 reason 输出内容必须是中文。\n"
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
        score = result.score or {}
        reason = result.reason or ""
        
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
        
        score_details = ", ".join(f"{k}: {v}" for k, v in score.items())
        
        judge_report = (
            f"{color_prefix}┌─────────────────────────── [ 日志裁判实时裁决报告 ] ───────────────────────────┐{reset}\n"
            f"{color_prefix}│{reset} {bold}当前迭代数{reset}: {latest_iter:<68} {color_prefix}│{reset}\n"
            f"{color_prefix}│{reset} {bold}裁决决策{reset}  : {color_prefix}{action_desc:<64}{reset} {color_prefix}│{reset}\n"
            f"{color_prefix}│{reset} {bold}核心指标{reset}  : {score_details:<68} {color_prefix}│{reset}\n"
            f"{color_prefix}├────────────────────────────────────────────────────────────────────────────────┤{reset}\n"
            f"{color_prefix}│{reset} {bold}裁决原因深度剖析{reset}:\n"
            f"{reason}\n"
            f"{color_prefix}└────────────────────────────────────────────────────────────────────────────────┘{reset}"
        )
        
        yield self._yield_text(judge_report)


class EvaluateTailiVideoAgent(_TailiStepBaseAgent):
    cfg: TailiCloudConfig
    description: str = "Evaluates four play_eval video files terrain-by-terrain and decides final pass/fail."
    instruction: str = (
        "你是 Taili 的视频裁判。你必须只输出严格 JSON，不要输出任何 Markdown、解释性前缀或多余文本。\n"
        "你每次只会看到一个地形的一段机器人评估视频，以及该视频对应的 eval_meta。\n"
        "注意：eval_meta 里除了命令序列，还可能带有 achieved_metrics —— 这是评估期从仿真里实测出来的客观物理数据，"
        "是判断'到底有没有完成任务'最可靠的依据。视频用来判断'动作自不自然'，数值用来判断'任务完成度'；两者结合，"
        "在任务完成度上数值拥有否决权（视频看着还行但数值证明没走起来，一律判失败）。\n\n"
        "【最高优先级：客观数值裁判（achieved_metrics 存在时必须据此判定，不得只凭视频观感放行）】\n"
        "- 速度跟踪 vel_tracking：对比 cmd_* 与 ach_*。若应当前进的时段 fwd_speed_ratio 明显偏小（如 < 0.5，"
        "即实际连指令速度的一半都没走到），说明在原地踏步/碎步而非真正位移 → passed=false。\n"
        "- 位移 locomotion：net_disp_xy_m 远小于'指令速度×时长'应达到的位移，同样说明原地不动 → false。\n"
        "- 步态频率 gait.foot_contact_freq_hz：正常 trot 每足约 1.5-2.5Hz；显著偏高（碎步/乱跳/高频抖动）→ false。"
        "mean_feet_in_contact 异常（长期接近 0=腾空乱跳，或长期=4 僵直不迈步）也是异常。\n"
        "- 稳定性 posture：mean_tilt_deg 偏大（机身明显倾斜）、base_height_std / vz_abs_mean 偏大（上下剧烈弹跳抖动）→ false。\n"
        "- num_resets > 0：在这段评估时长内基本意味着摔倒或冲出地形 → false。\n"
        "- 碎步以外的足端质量证据（存在则务必一并据此判，不要只盯碎步）：gait.swing_clearance.*（摆动相足端离地高度，过低=抬脚不足/拖地）；"
        "gait.foot_slip_ratio（支撑相足端对地水平速度占比，高=触地打滑/拖滑）；gait.foot_touchdown_cv / diag_pair_touchdown_diff（四足触地不均，大=跛行/瘸腿）；"
        "posture.impact.p95_touchdown_grf_bw（落地法向冲击力体重倍数，大=砸地/硬着陆）。这些当前可能为观测量(未必接硬闸)，但你应在视觉上印证并据此判 passed 与回填 failure_tags。\n"
        "- 若 achieved_metrics 缺失（老版本或采集失败），则退回纯视频判断。\n\n"
        "请同时从以下视觉维度评估，并与数值相互印证：\n"
        "1. 机身稳定性：躯干是否发生剧烈晃动、俯仰或倾斜，是否能保持相对平稳。\n"
        "2. 四肢协调性：四条腿的步态是否连贯、对称、协调，有无明显的步态畸形、碎步或僵硬。\n"
        "3. 足端运动质量：落足点是否准确，抬腿是否干净利落，有无拖地、乱踢、踢障碍、绊台阶。\n"
        "4. 指令跟随：是否大体按命令方向产生有效位移，而不是全程躺平、僵直、原地抖动或原地碎步不前进。\n"
        "5. 地形专项标准：必须结合 terrain_specific_criteria 评估，不同地形关注点不同。\n\n"
        "【容错边界】：仅对'短暂（1-2 秒）且随后恢复正常步态'的停顿、轻微打滑、落脚调整容错。"
        "若 achieved_metrics 已客观证明完成度差（fwd_speed_ratio 低 / 碎步频率高 / 位移不足 / 有 reset），不得以'容错'为由放行。\n"
        "【失败底线】：彻底摔倒或无法起步、严重失真、长时间趴地/僵直/原地鬼畜、原地碎步无有效位移、或关键数值不达标，应判定 passed=false。\n"
        "失败时请在 score.failure_tags 中给标签，例如 foot_clearance、foot_slip、gait_asymmetry、contact_impact、body_stability、gait_coordination、"
        "velocity_tracking、high_step_frequency、short_stride、body_bounce、no_net_displacement、stairs_up_failure、stairs_down_failure、obstacle_collision、local_optimum_no_motion。\n"
        "并在 score 中回填你重点参考的关键数值（如 fwd_speed_ratio、foot_contact_freq_hz、num_resets），便于复核。\n\n"
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

    def _get_eval_terrains(self) -> list[str]:
        eval_terrains_raw = self.cfg.play_eval_terrains
        if isinstance(eval_terrains_raw, str):
            terrains = [item.strip() for item in eval_terrains_raw.split(",") if item.strip()]
        else:
            terrains = list(eval_terrains_raw)
        return terrains or ["flat", "boxes", "stairs_down", "stairs_up"]

    def _terrain_specific_criteria(self, terrain: str) -> list[str]:
        criteria_map = {
            "flat": [
                "平地重点观察前进、横移、转向三个阶段是否都能按指令产生有效运动。",
                "前进阶段应有相对稳定和协调的四足步态，对角腿协同不能明显混乱。",
                "横移和转向阶段允许轻微重心偏移，但不能持续大幅侧倾或摔倒。",
                "停止阶段应能恢复站稳，而不是继续抽搐或趴下。",
            ],
            "boxes": [
                "方块地形重点观察足端抬高和落足质量，不能频繁踢到方块边缘或卡脚。",
                "允许短暂停顿和寻找落脚点，但不能长期不前进或完全拒绝通过。",
                "身体被障碍扰动后应能恢复平衡，不能明显被顶翻或拖地滑行。",
            ],
            "stairs_down": [
                "下楼梯重点观察是否能控制前倾和俯仰，不能连续踩空或头部下栽。",
                "足端应能逐级落脚，下台阶时允许谨慎减速，但不能全程僵直不动。",
                "如果出现摔下楼梯、持续前翻、长时间卡住，应判失败。",
            ],
            "stairs_up": [
                "上楼梯重点观察前脚和后脚是否能抬过台阶，不能反复踢台阶或后脚绊住。",
                "允许短暂停顿蓄力或调整姿态，但应持续尝试向上通过。",
                "如果长时间停在原地、后退、无法上第一级或明显抬脚不足，应判失败。",
            ],
        }
        return criteria_map.get(terrain, ["按通用四足机器人稳定行走标准评估。"])

    def _extract_failure_tags(self, result: TailiVideoJudgeResult, terrain: str) -> list[str]:
        score = result.score or {}
        tags = score.get("failure_tags") if isinstance(score, dict) else None
        if isinstance(tags, list):
            return [str(x) for x in tags if str(x).strip()]
        text = (result.reason or "") + "\n" + json.dumps(score, ensure_ascii=False)
        text_lower = text.lower()
        derived: list[str] = []
        keyword_map = {
            "foot_clearance": ["抬脚", "抬腿", "拖地", "踢", "卡脚", "绊", "clearance", "stumble"],
            "foot_slip": ["打滑", "拖滑", "滑移", "滑步", "蹭地", "slip", "slide"],
            "gait_asymmetry": ["不对称", "跛", "瘸", "瘸腿", "单腿", "失衡", "asymmetr", "limp"],
            "contact_impact": ["砸地", "硬着陆", "跺", "重重", "落地冲击", "砸", "impact", "slam"],
            "body_stability": ["晃动", "侧倾", "俯仰", "翻", "摔", "不稳", "pitch", "roll"],
            "gait_coordination": ["步态", "协调", "对称", "僵硬", "腿部", "gait"],
            "velocity_tracking": ["不动", "没有位移", "原地", "拒绝", "跟随", "tracking"],
            "obstacle_collision": ["方块", "障碍", "碰撞", "撞", "boxes", "collision"],
            "local_optimum_no_motion": ["躺平", "僵直", "全程不动", "原地抖", "局部最优"],
        }
        if terrain == "stairs_up":
            keyword_map["stairs_up_failure"] = ["上楼", "上台阶", "上楼梯", "stairs_up", "pyramid_stairs_inv"]
        if terrain == "stairs_down":
            keyword_map["stairs_down_failure"] = ["下楼", "下台阶", "下楼梯", "stairs_down", "pyramid_stairs"]
        for tag, keywords in keyword_map.items():
            if any(k.lower() in text_lower for k in keywords):
                derived.append(tag)
        return derived

    def _numeric_score(self, result: TailiVideoJudgeResult) -> float:
        score = result.score or {}
        if isinstance(score, dict):
            for key in ("overall", "overall_score", "score", "total"):
                val = score.get(key)
                if isinstance(val, (int, float)):
                    return float(val)
        # VLM 没返回数值分时用语义默认：通过=75(中等正常)，失败=30(明显差)
        return 75.0 if result.passed else 30.0

    def _effective_gate_thresholds(self, terrain: str) -> dict:
        """解析某地形的有效硬闸阈值：全局 gate_* 为基线，叠加 gate_terrain_overrides[terrain]。

        覆盖键支持短名（max_contact_freq_hz）或带 gate_ 前缀的全名；缺省项继承全局（平地通常不覆盖=最严）。
        """
        cfg = self.cfg
        base = {
            "min_fwd_speed_ratio": cfg.gate_min_fwd_speed_ratio,
            "max_resets": cfg.gate_max_resets,
            "max_contact_freq_hz": cfg.gate_max_contact_freq_hz,
            "min_swing_time_s": cfg.gate_min_swing_time_s,
            "min_stride_norm": cfg.gate_min_stride_norm,
            "max_bounce_ratio": cfg.gate_max_bounce_ratio,
            # 碎步以外的失败模式（None=禁用仅观测，标定后填阈值即开闸；规则在 _gate_rules 声明）
            "min_swing_clearance_m": getattr(cfg, "gate_min_swing_clearance_m", None),
            "max_foot_slip_ratio": getattr(cfg, "gate_max_foot_slip_ratio", None),
            "max_foot_touchdown_cv": getattr(cfg, "gate_max_foot_touchdown_cv", None),
            "max_diag_pair_diff": getattr(cfg, "gate_max_diag_pair_diff", None),
            "max_p95_touchdown_grf_bw": getattr(cfg, "gate_max_p95_touchdown_grf_bw", None),
        }
        overrides = (getattr(cfg, "gate_terrain_overrides", {}) or {}).get(terrain, {}) or {}
        for k, v in overrides.items():
            short = k[5:] if str(k).startswith("gate_") else str(k)
            if short in base and (v is None or isinstance(v, (int, float))):
                base[short] = v  # v=None => 在该地形禁用该规则（如 clearance 在台阶上 baseline 失真）
        return base

    def _gate_rules(self) -> list[dict]:
        """声明式硬闸规则表：加一种失败模式 = play_eval 多写一个量 + 这里加一行 + config 加一个阈值字段。

        每条规则: key=对应 _effective_gate_thresholds 里的阈值短名（该阈值为 None 即禁用此规则，metrics-only）；
        extract=从 achieved_metrics 取标量(取不到返回 None 则跳过)；op='lt'/'gt'(小于/大于阈值即判失败)；
        tag=失败标签(对应 failure_tag_to_knob_hints 的修参旋钮)；reason=生成否决文案的函数。
        前 6 条与旧硬代码逐字等价(碎步家族+完成度)；后 5 条是碎步以外的新维度，默认阈值 None=不接闸只观测。
        """
        hip = getattr(self.cfg, "robot_hip_height", 0.53) or 0.53

        def _dig(metrics, *path):
            cur = metrics
            for p in path:
                if not isinstance(cur, dict):
                    return None
                cur = cur.get(p)
            return cur

        def _contact_freq(m):
            cf = _dig(m, "gait", "contact_freq_hz_motion")
            return cf if isinstance(cf, (int, float)) else _dig(m, "gait", "foot_contact_freq_hz")

        def _stride_norm(m):
            stride = _dig(m, "gait", "stride_length_m")
            return round(stride / hip, 3) if isinstance(stride, (int, float)) and hip > 0 else None

        def _resets(m):
            try:
                return int(m.get("num_resets", 0) or 0)
            except Exception:
                return None

        return [
            # —— 任务完成类（与尺寸无关）——
            {"key": "min_fwd_speed_ratio", "op": "lt", "tag": "velocity_tracking",
             "extract": lambda m: _dig(m, "vel_tracking", "fwd_speed_ratio"),
             "reason": lambda v, t: f"fwd_speed_ratio={v}<{t}(几乎没按指令前进)"},
            {"key": "max_resets", "op": "gt", "tag": "fall_or_out_of_bounds",
             "extract": _resets,
             "reason": lambda v, t: f"num_resets={v}>{t}(摔倒/出界)"},
            # —— 步态质量类·碎步家族（物理先验阈值）——
            {"key": "max_contact_freq_hz", "op": "gt", "tag": "high_step_frequency",
             "extract": _contact_freq,
             "reason": lambda v, t: f"contact_freq={v}Hz>{t}(碎步/高频踏步)"},
            {"key": "min_swing_time_s", "op": "lt", "tag": "high_step_frequency",
             "extract": lambda m: _dig(m, "gait", "mean_swing_time_s"),
             "reason": lambda v, t: f"swing_time={v}s<{t}(迈步太碎)"},
            {"key": "min_stride_norm", "op": "lt", "tag": "short_stride",
             "extract": _stride_norm,
             "reason": lambda v, t: f"stride_norm={v}<{t}(步幅过短/拖步)"},
            {"key": "max_bounce_ratio", "op": "gt", "tag": "body_bounce",
             "extract": lambda m: _dig(m, "posture", "bounce_ratio"),
             "reason": lambda v, t: f"bounce_ratio={v}>{t}(上下颠簸/跳)"},
            # —— 碎步以外（默认 config 阈值 None=禁用，仅观测；标定后填阈值即开闸）——
            {"key": "min_swing_clearance_m", "op": "lt", "tag": "foot_clearance",
             "extract": lambda m: _dig(m, "gait", "swing_clearance", "min_swing_clearance_m"),
             "reason": lambda v, t: f"min_swing_clearance={v}m<{t}m(摆动相离地不足/拖地)"},
            {"key": "max_foot_slip_ratio", "op": "gt", "tag": "foot_slip",
             "extract": lambda m: _dig(m, "gait", "foot_slip_ratio"),
             "reason": lambda v, t: f"foot_slip_ratio={v}>{t}(触地打滑/拖滑)"},
            {"key": "max_foot_touchdown_cv", "op": "gt", "tag": "gait_asymmetry",
             "extract": lambda m: _dig(m, "gait", "foot_touchdown_cv"),
             "reason": lambda v, t: f"foot_touchdown_cv={v}>{t}(触地次数离散/跛行)"},
            {"key": "max_diag_pair_diff", "op": "gt", "tag": "gait_asymmetry",
             "extract": lambda m: _dig(m, "gait", "diag_pair_touchdown_diff"),
             "reason": lambda v, t: f"diag_pair_diff={v}>{t}(对角失衡/跛行)"},
            {"key": "max_p95_touchdown_grf_bw", "op": "gt", "tag": "contact_impact",
             "extract": lambda m: _dig(m, "posture", "impact", "p95_touchdown_grf_bw"),
             "reason": lambda v, t: f"p95_grf={v}BW>{t}BW(落地砸地/硬着陆)"},
        ]

    def _metric_gate(self, metrics: dict | None, terrain: str = "") -> dict:
        """基于 play_eval 写入的 achieved_metrics 做客观硬闸（按地形取有效阈值，规则见 _gate_rules）。

        返回 {"passed": bool, "reason": str, "tags": [...], "thresholds": {...}}。任一启用规则不达标 => passed=False；
        阈值为 None（规则禁用）或指标字段缺失 => 自动跳过（向后兼容旧 eval_meta / 采集失败 / metrics-only 模式）。
        语义：硬闸只能"否决"视频裁判的通过，绝不把失败救成通过。
        """
        if not isinstance(metrics, dict) or not getattr(self.cfg, "metric_gate_enabled", True):
            return {"passed": True, "reason": "", "tags": [], "thresholds": {}}
        th = self._effective_gate_thresholds(terrain)
        fails: list[str] = []
        tags: list[str] = []
        for rule in self._gate_rules():
            thr = th.get(rule["key"])
            if thr is None:  # 阈值 None => 规则禁用（metrics-only），跳过
                continue
            try:
                val = rule["extract"](metrics)
            except Exception:
                val = None
            if not isinstance(val, (int, float)):
                continue  # 指标缺失/采集失败 => 跳过该条（向后兼容）
            bad = (val < thr) if rule["op"] == "lt" else (val > thr)
            if bad:
                fails.append(rule["reason"](val, thr))
                tags.append(rule["tag"])
        if fails:
            return {"passed": False, "reason": "；".join(fails), "tags": list(dict.fromkeys(tags)), "thresholds": th}
        return {"passed": True, "reason": "", "tags": [], "thresholds": th}

    async def _retain_best_checkpoint(
        self,
        ctx: InvocationContext,
        terrain_results: list[dict],
        run_root: str,
        overall_passed: bool,
        overall_score: float,
        failed_terrains: list,
    ) -> AsyncGenerator[Event, None]:
        """记录本轮被评估的 checkpoint；按统一冠军标准更新"续训源"与"交付 best"。

        依据 play_eval 写入的 eval_meta.json：其中 checkpoint 字段即本轮被评估的精确 .pt 路径，
        log_dir 是其所在 run 目录（4 地形共用同一 checkpoint）。这样即便最终进 HITL/失败，
        本地 logs/taili_best/ 也始终留存着"迄今最优"的那一份参数。

        关键修复：冠军排序不再只看视频分，而是把训练验收指标(terrain_levels/error_vel)纳入，
        且 completed 轮在这里也会顺带更新续训源(RESUME_SOURCE)，与超时救援轮统一标准（见 _promote_checkpoint）。
        """
        host = self.cfg.remote_host
        port = self.cfg.remote_port
        user = self.cfg.remote_user
        password = self.cfg.remote_password

        # 1. 从任一地形的 eval_meta 中提取本轮 checkpoint 与其 run 目录
        ckpt_remote = ""
        ckpt_run_dir = ""
        for item in terrain_results:
            meta = item.get("eval_meta") or {}
            if not ckpt_remote and meta.get("checkpoint"):
                ckpt_remote = str(meta.get("checkpoint"))
            if not ckpt_run_dir and meta.get("log_dir"):
                ckpt_run_dir = str(meta.get("log_dir"))
            if ckpt_remote and ckpt_run_dir:
                break
        if not ckpt_run_dir:
            ckpt_run_dir = run_root
        # 兜底：meta 缺 checkpoint 时，直接到 run 目录里找迭代号最大的 model_*.pt
        if not ckpt_remote and ckpt_run_dir:
            ckpt_remote = self._remote.remote_find_latest_checkpoint(host, port, user, password, ckpt_run_dir, self.cfg.remote_timeout_seconds)

        # 2. 组装本轮记录（带训练验收指标，供统一冠军排序使用），交由共用例程晋升
        iter_round = int(ctx.session.state.get(STATE_P2_ITER_ROUND, 0))
        round_record = {
            "round": iter_round,
            "run_dir": ckpt_run_dir,
            "checkpoint_remote": ckpt_remote,
            "status": "completed",
            "acceptance": ctx.session.state.get(STATE_P2_TRAIN_ACCEPTANCE, {}),
            "overall_passed": bool(overall_passed),
            "overall_score": float(overall_score),
            # 视频通过的地形数：冠军排序的核心信号（部分通过 > 全挂），优先于会被碎步 game 的 terrain_levels
            "num_video_passed": sum(1 for t in terrain_results if t.get("passed")),
            "failed_terrains": list(failed_terrains),
        }
        res = self._promote_checkpoint(ctx, round_record)

        if not res["best_updated"] and not res["resume_updated"]:
            yield self._yield_text(
                f"{self.name}: 本轮(round={iter_round}, score={overall_score:.2f}, passed={overall_passed}) "
                f"未优于历史最优，保留既有续训源/最优 checkpoint"
            )
            return

        tag = []
        if res["resume_updated"]:
            tag.append("已采纳为下一轮续训源")
        if res["best_updated"]:
            tag.append(f"并下载交付 best→{res['bundle']['local_dir']}（含 {list(res['bundle']['files'].keys())}；缺失 {res['bundle']['missing']}）")
        self._add_log(
            ctx,
            f"[{self.name}] 刷新冠军 checkpoint round={iter_round} score={overall_score:.2f} passed={overall_passed} -> {'/'.join(tag)}",
        )
        yield self._yield_text(
            f"{self.name}: 本轮(round={iter_round}, score={overall_score:.2f}, passed={overall_passed}) {'，'.join(tag)}"
        )

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        ctx.session.state[STATE_P2_STAGE] = Phase2Stage.EVALUATE_VIDEO
        host = self.cfg.remote_host
        port = self.cfg.remote_port
        user = self.cfg.remote_user
        password = self.cfg.remote_password
        eval_terrains = self._get_eval_terrains()

        eval_root = f"{self.cfg.eval_log_root}/{self._get_experiment_name()}"
        run_root = self._remote.remote_list_latest_run(host, port, user, password, eval_root, self.cfg.remote_timeout_seconds)
        if not run_root:
            raise RuntimeError(f"未找到实验运行目录: {eval_root}")

        artifacts = self._remote.remote_find_play_eval_artifacts(
            host, port, user, password, run_root, self.cfg.remote_timeout_seconds, eval_terrains
        )
        missing = [terrain for terrain in eval_terrains if terrain not in artifacts]
        if missing:
            raise RuntimeError(f"未找到部分 play_eval 评估产物: {missing}; 已找到: {list(artifacts.keys())}")

        local_root = Path("logs") / "taili_play_eval"
        local_root.mkdir(parents=True, exist_ok=True)

        terrain_results: list[dict] = []
        local_video_paths: dict[str, str] = {}
        remote_video_paths: dict[str, str] = {}
        all_failure_tags: list[str] = []

        dashscope_config = LLMCallConfig(
            api_key_env="DASHSCOPE_API_KEY",
            base_url_env="DASHSCOPE_BASE_URL",
            default_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model="qwen3.7-plus",
        )

        for terrain in eval_terrains:
            artifact = artifacts[terrain]
            video_remote_path = artifact["video_remote_path"]
            meta_remote_path = artifact["meta_remote_path"]
            if not self._remote.wait_for_remote_file_stable(host, port, user, password, video_remote_path, self.cfg.remote_timeout_seconds):
                raise RuntimeError(f"远端视频文件未稳定落盘，无法评估: {terrain} {video_remote_path}")

            video_local_path = local_root / f"{terrain}.mp4"
            meta_local_path = local_root / f"{terrain}_eval_meta.json"
            self._remote.fetch_remote_file(host, port, user, password, video_remote_path, str(video_local_path), self.cfg.remote_timeout_seconds)
            self._remote.fetch_remote_file(host, port, user, password, meta_remote_path, str(meta_local_path), self.cfg.remote_timeout_seconds)
            local_video_paths[terrain] = str(video_local_path)
            remote_video_paths[terrain] = video_remote_path

            try:
                meta = json.loads(meta_local_path.read_text(encoding="utf-8"))
            except Exception:
                meta = artifact.get("meta", {})

            payload = {
                "terrain": terrain,
                "eval_meta": meta,
                "achieved_metrics": meta.get("achieved_metrics") if isinstance(meta, dict) else None,
                "terrain_specific_criteria": self._terrain_specific_criteria(terrain),
                "play_exit_code": ctx.session.state.get(STATE_P2_PLAY_EXIT_CODE),
                "play_stderr": ctx.session.state.get(STATE_P2_PLAY_STDERR, ""),
                "note": "achieved_metrics 是仿真实测的客观数据，请优先据此判定任务完成度（尤其速度跟踪/位移/步态频率/reset），视频用于判断动作自然度。真正的视频内容在下方多模态输入中提供。",
            }

            with open(video_local_path, "rb") as f:
                video_b64 = base64.b64encode(f.read()).decode("utf-8")

            prompt_content = [
                {
                    "type": "text",
                    "text": f"请基于以下完整事实执行单地形视频裁判任务:\n{json.dumps(payload, ensure_ascii=False)}\n\n以下是 terrain={terrain} 的真实评估视频：",
                },
                {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}},
            ]

            yield self._yield_text(f"{self.name}: 正在评估地形视频 {terrain} ...")
            result = UnifiedLLMClient(dashscope_config).generate_json(
                system_prompt=self.instruction,
                user_prompt=prompt_content,
                schema=TailiVideoJudgeResult,
            )
            # 数值硬闸：用 achieved_metrics 否决 VLM 的误判通过（只会更严，不会救活失败）。
            gate = self._metric_gate(meta.get("achieved_metrics") if isinstance(meta, dict) else None, terrain)
            final_passed = bool(result.passed) and gate["passed"]
            tags = self._extract_failure_tags(result, terrain) if not result.passed else []
            for t in gate["tags"]:
                if t not in tags:
                    tags.append(t)
            for tag in tags:
                if tag not in all_failure_tags:
                    all_failure_tags.append(tag)

            reason_text = result.reason or ""
            if not gate["passed"]:
                reason_text = (f"{reason_text} | 数值硬闸否决: {gate['reason']}").strip(" |")
                yield self._yield_text(
                    f"\033[1;91m[{self.name}] 地形 {terrain} 被数值硬闸否决(VLM passed={result.passed}): {gate['reason']}\033[0m"
                )

            # gate 否决时分数压低，避免碎步高 VLM 分污染"最优"判定
            vlm_score = self._numeric_score(result)
            if not gate["passed"]:
                numeric_score = min(vlm_score, 35.0)
            elif not result.passed:
                numeric_score = vlm_score  # VLM 自己判失败，保留原分
            else:
                numeric_score = vlm_score

            terrain_results.append({
                "terrain": terrain,
                "passed": final_passed,
                "vlm_passed": bool(result.passed),
                "vlm_score": vlm_score,
                "numeric_score": numeric_score,
                "score": result.score or {},
                "reason": reason_text,
                "failure_tags": tags,
                "metric_gate": gate,
                "video_local_path": str(video_local_path),
                "video_remote_path": video_remote_path,
                "meta_remote_path": meta_remote_path,
                "eval_meta": meta,
            })

        overall_passed = all(item["passed"] for item in terrain_results)
        failed_terrains = [item["terrain"] for item in terrain_results if not item["passed"]]
        overall_score = sum(float(item["numeric_score"]) for item in terrain_results) / max(1, len(terrain_results))
        summary = {
            "passed": overall_passed,
            "overall_score": overall_score,
            "failed_terrains": failed_terrains,
            "failure_tags": all_failure_tags,
            "terrain_results": terrain_results,
            "run_root": run_root,
        }

        ctx.session.state[STATE_P2_VIDEO_INPUT_PAYLOAD] = {
            "run_root": run_root,
            "artifacts": artifacts,
            "eval_terrains": eval_terrains,
        }
        ctx.session.state[STATE_P2_EVAL_VIDEO_PATH] = local_video_paths
        ctx.session.state[STATE_P2_EVAL_VIDEO_REMOTE_PATH] = remote_video_paths
        ctx.session.state[STATE_P2_VIDEO_JUDGE_SUMMARY] = summary
        ctx.session.state[STATE_P2_FAILED_TERRAINS] = failed_terrains
        ctx.session.state[STATE_P2_FAILURE_TAGS] = all_failure_tags
        ctx.session.state[STATE_P2_EVAL_PASSED] = overall_passed
        ctx.session.state[STATE_P2_EVAL_SCORE] = {
            "overall_score": overall_score,
            "terrain_results": terrain_results,
            "failed_terrains": failed_terrains,
            "failure_tags": all_failure_tags,
        }

        if overall_passed:
            ctx.session.state[STATE_P2_EVAL_FAIL_REASON] = ""
        else:
            reason_lines = ["四地形视频评估未全部通过。"]
            for item in terrain_results:
                if not item["passed"]:
                    reason_lines.append(
                        f"[{item['terrain']}] tags={item['failure_tags']} reason={item['reason']}"
                    )
            ctx.session.state[STATE_P2_EVAL_FAIL_REASON] = "\n".join(reason_lines)

        # 【最优 checkpoint 留存】记录本轮 checkpoint，并在优于历史最优时实时下载保底到本地。
        async for evt in self._retain_best_checkpoint(
            ctx, terrain_results, run_root, overall_passed, overall_score, failed_terrains
        ):
            yield evt

        self._add_log(ctx, f"[{self.name}] 四地形视频评估完成 passed={overall_passed} failed={failed_terrains}")


class RepairTailiWorkflowStepAgent(_TailiStepBaseAgent):
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        ctx.session.state[STATE_P2_STAGE] = Phase2Stage.ITERATE_TUNING
        
        # 1. 更新迭代轮次
        ctx.session.state[STATE_P2_ITER_ROUND] = int(ctx.session.state.get(STATE_P2_ITER_ROUND, 0)) + 1
        
        # 2. 收集上一轮失败原因：按本轮 train_status 区分——只有真正跑过视频评估(completed)才喂视频摘要，
        #    否则(发散/超时/渲染失败)视频状态已被 TrainAgent 清空，改以训练日志裁判结论为主，避免用陈旧视频反馈误导改参。
        train_status = str(ctx.session.state.get(STATE_P2_TRAIN_STATUS, "completed"))
        failure_reason = ctx.session.state.get(STATE_P2_EVAL_FAIL_REASON, "")
        video_summary = ctx.session.state.get(STATE_P2_VIDEO_JUDGE_SUMMARY)
        failed_terrains = ctx.session.state.get(STATE_P2_FAILED_TERRAINS, [])
        failure_tags = ctx.session.state.get(STATE_P2_FAILURE_TAGS, [])
        if train_status == "completed" and video_summary:
            reason = json.dumps(
                {
                    "stage": "video_eval_failed",
                    "failure_reason": failure_reason,
                    "failed_terrains": failed_terrains,
                    "failure_tags": failure_tags,
                    "video_eval_summary": video_summary,
                },
                ensure_ascii=False,
            )
        else:
            # 训练未完成：本轮没有视频证据，给训练侧的真实失败信号（发散裁判结论 / 超时 / 非零退出）
            reason = json.dumps(
                {
                    "stage": "training_not_completed",
                    "train_status": train_status,
                    "failure_reason": failure_reason or str(ctx.session.state.get(STATE_P2_HITL_REASON, "revise")),
                    "train_log_judge": ctx.session.state.get(STATE_P2_TRAIN_LOG_JUDGE_RESULT, {}),
                    "note": "本轮训练未完成(发散/超时/渲染失败)，未做视频评估；请据 train_status 与训练日志裁判结论调参，"
                            "切勿参考可能过期的视频反馈。若为发散(early_stopped)，优先降学习率/收敛改参幅度/避免奖励突变。",
                },
                ensure_ascii=False,
            )

        # 3. 将失败原因更新到最后一条 history 中
        history = list(ctx.session.state.get(STATE_P2_CONFIG_HISTORY, []))
        if history:
            history[-1]["failure_reason"] = reason
        ctx.session.state[STATE_P2_CONFIG_HISTORY] = history

        # 4. 切换到 revise 模式（版本号/父版本号统一由 ConfigSynthesis 系统管理，这里不再设置，避免与其打架）
        ctx.session.state[STATE_P2_CONFIG_MODE] = "revise"

        self._add_log(ctx, f"[{self.name}] 收集失败原因(train_status={train_status})并进入迭代轮次 {ctx.session.state[STATE_P2_ITER_ROUND]}")
        yield self._yield_text(f"{self.name}: 收集上轮失败原因，准备进行第 {ctx.session.state[STATE_P2_ITER_ROUND]}/{ctx.session.state[STATE_P2_ITER_MAX]} 轮迭代 (revise)")


class ArchiveTailiOutputsStepAgent(_TailiStepBaseAgent):
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        ctx.session.state[STATE_P2_STAGE] = Phase2Stage.ARCHIVE_OUTPUTS
        best_checkpoint = ctx.session.state.get(STATE_P2_BEST_CHECKPOINT, {}) or {}
        summary = {
            "task_id": f"taili-{self.cfg.session_id}",
            "task_name": self.cfg.task_name,
            "cloud_asset_path": self.cfg.cloud_asset_path,
            "cloud_task_root": self.cfg.cloud_task_cfg_root,
            "train_command": ctx.session.state.get(STATE_P2_TRAIN_COMMAND),
            "resume_info": ctx.session.state.get(STATE_P2_TRAIN_RESUME_INFO, {}),
            "eval": ctx.session.state.get(STATE_P2_EVAL_SCORE, {}),
            "passed": ctx.session.state.get(STATE_P2_EVAL_PASSED, False),
            "best_checkpoint": best_checkpoint,
            "checkpoint_history": ctx.session.state.get(STATE_P2_CHECKPOINT_HISTORY, []),
            "training_acceptance": ctx.session.state.get(STATE_P2_TRAIN_ACCEPTANCE, {}),
        }
        ctx.session.state[STATE_P2_ARCHIVE_SUMMARY] = summary
        ctx.session.state[STATE_P2_ARCHIVE_COMPLETED] = True
        best_dir = best_checkpoint.get("local_dir", "")
        self._add_log(ctx, f"[{self.name}] 归档完成 best_dir={best_dir}")
        yield self._yield_text(
            f"{self.name}: 归档完成" + (f"，最优 pt 已保存在本地 {best_dir}" if best_dir else "")
        )
