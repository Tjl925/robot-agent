# 机械狗多 Agent 项目总结 (PROJECT_SUMMARY.md)

## 1. 项目定位
基于 **Google ADK** 的专用多 Agent 闭环系统，单一主线：本地 `taili_quad/`(URDF+STL) → 云端 AutoDL 上的 `robot_lab`(IsaacLab + rsl_rl)。LLM 自主诊断 URDF、生成 PPO/环境配置、同步云端、驱动训练 + 自适应评估闭环，失败自动 revise 或转人工(HITL)，最终交付一份最优 `.pt`。

---

## 2. 整体架构

### 2.1 两阶段编排
- **唯一入口** `main.py` → `OrchestratorAgent`。
- **Phase 1**（`Phase1OrchestratorAgent`）：AutoDL 开机 → 状态轮询 → SSH 探活，SSH 凭证 Handoff 给 Phase 2。
- **Phase 2**（`TailiOrchestratorAgent`）：同一 Session 内驱动 `诊断 → 配置 → 发布 → 训练 → 评估 → 归档/迭代` 全流程，并**集中路由**所有成败/revise/HITL 跳转（Step Agent 不得越权改终态）。

### 2.2 训练退出协议（`STATE_P2_TRAIN_STATUS`）
| 状态 | 后续 |
|---|---|
| `early_stopped` | 日志裁判判发散 → 杀进程、跳视频、进 revise |
| `completed` | 正常完成/收敛 → 抓验收快照 → 渲染四地形视频 → 视频裁判 |
| `play_failed` | 视频渲染报错 → 熔断、跳评估、进 revise |
| `train_failed` | 训练非零退出 → 进 revise |
| `train_timeout` | 超时强杀，但**抢救本轮最新 checkpoint** 后进 revise |

### 2.3 Phase 2 Agent 职责
| Agent | 职责 |
|---|---|
| `AnalyzeTailiUrdfStepAgent` | 读本地 URDF，输出结构化风险诊断 |
| `TailiConfigSynthesisAgent` | create/revise 生成 6 个 Python 配置；revise 只输出变更字段；失败标签→奖励旋钮提示；版本号系统自增 |
| `GenerateTailiFilesStepAgent` | 配置落盘 `.taili_generated/`，按需局部覆写 |
| `PublishTailiWorkspaceStepAgent` | 递归扫描资产，云端按原结构建树 + SFTP 部署 |
| `TrainTailiStepAgent` | 异步起训、warm-start 续训+自检、byte-offset 增量日志、采样投喂日志裁判早停、时间预算 `--max_iterations` 截断、超时抢救、训练后抓验收快照 + 渲染四地形视频 |
| `EvaluateTailiTrainingLogAgent` | 日志趋势裁判，单次无状态判 `continue/stop_failed/stop_converged` |
| `EvaluateTailiVideoAgent` | Qwen 多模态视频打分 + `achieved_metrics` 客观数值硬闸(按地形分档) + 维护冠军/最优 `.pt` |
| `RepairTailiWorkflowStepAgent` | 收集失败原因、推进迭代轮次、准备下一轮 revise |
| `ArchiveTailiOutputsStepAgent` | 归档最终配置/得分/最优 `.pt` 落点/checkpoint 历史/验收快照 |

---

## 3. 关键机制（系统当前能力）

- **增量日志拉取**：`remote_tail_log(..., byte_offset)` + 远端 `wc -c`/`tail -c`，每次只传新增几百字节，避免长训日志数十 MB 全量回传致 SSH 超时。
- **create/revise + 局部覆写**：历史仅存极简摘要；revise 由 Agent 直接读本地真实代码喂 LLM，LLM 只输出变更字段，本地按需覆写。省 Token、防生成漂移。
- **多模型协同**：DeepSeek 负责逻辑/配置/日志裁判，Qwen 负责视频打分（本地 MP4→Base64）。
- **warm-start 续训**：存在历史最优 checkpoint 时追加 `--resume --load_run --checkpoint`（依云端 `cli_args.py` 的 store_true 裸旗标），迭代号接续累积；grep 远端 `Loading model checkpoint from` 自检，杜绝静默从零白练。续训源 `RESUME_SOURCE` 与"视频最优 best"**解耦**，completed/超时抢救轮均按统一标准更新。
- **统一冠军排序 `_champion_rank_key`**：续训源与交付 best 共用。排序键 `(四地形视频全过 → 视频通过地形数 → 视频综合分 → terrain_levels → -error_vel_xy)`——**视频证据优先，terrain_levels 仅作打平兜底**（它是课程进度、会被碎步刷高，不能凌驾视频分）。record 带 `num_video_passed`（无视频轮记 -1）。
- **最优 `.pt` 留存交付**：刷新冠军即把 `model_*.pt` + `policy.pt/onnx` + `params/*.yaml` 下载到 `logs/taili_best/`（含 `BEST_MANIFEST.json`），失败/HITL 也保底。
- **视频客观数值硬闸 `_metric_gate`（声明式规则表 `_gate_rules`）**：`play_eval.py` 实测 `achieved_metrics` 写回 `eval_meta.json`，本地据此否决"速度达标但步态难看"的策略（只严不松）。阈值按地形分档（`gate_terrain_overrides`，楼梯放宽、平地最严）。规则表化后，加一种失败模式 = play_eval 加 1 个量 + 规则表加 1 行 + config 加 1 个阈值。
  - **覆盖维度**：碎步家族（步频/摆动时长/步幅）、颠簸、速度/位移、reset；**碎步以外 4 维**（抬脚高度/拖地、触地打滑、跛行不对称、落地砸地）当前为 **metrics-only**（喂 VLM/修参，阈值默认 `None` 暂不否决，标定后开闸，见 `TESTING_PROGRESS.md` §3.2）。
- **失败标签 → 奖励旋钮映射**：`failure_tag_to_knob_hints` 把 `high_step_frequency/short_stride/foot_clearance/foot_slip/gait_asymmetry/contact_impact/...` 映射到具体调参建议（如开启被禁用的 `feet_gait`、调 `feet_height`/`feet_slide`/`contact_forces` 等），让 revise 对症改参。
- **训练最终验收快照**：训练完成后抓 `terrain_levels/error_vel_xy/error_vel_yaw` 对照目标（默认 `≥6 / ≤0.4 / ≤0.5`），仅作**只读标尺**，**绝不**参与日志裁判早停（避免为绝对目标死等到最大步数）。
- **max_iterations 时间预算截断**：用预热实测每步耗时估墙钟内可达步数，超出则追加原生 `--max_iterations`，让训练自然 `completed` 而非被墙钟杀。注意 rsl_rl 该值是"本次新增步数"。
- **掉线优雅退出**：`main.py` 的 `Tee` 写入吞错（断 `Exception ignored in` 级联），`main()` try/finally + `os._exit`，避免服务器掉线后残留非守护线程拖住进程。

---

## 4. 避坑指南（核心资产）

1. **ADK 状态同步**：仅 `ctx.session.state[K]=V` 改内存、不经 `append_event(state_delta=...)` 原子提交，多阶段序列化时会**丢失**。重要状态变更务必原子同步。
2. **Stage 越权跳转**：Step Agent 只改自己阶段的 `STATE_P2_STAGE`，所有成败/revise/HITL 路由必须由 `taili_orchestrator.py` 集中决策。
3. **远端失败排查顺序**：① 配置是否同步到云端正确路径（90% 报错根源）→ ② SSH 手动跑 `phase2.train.command` → ③ checkpoint 通配符匹配 → ④ `play_eval_timeout_seconds` > 300s。
4. **路径怀疑优先**：加载/执行报错先怀疑路径与 SFTP 同步，别急于改 LLM 生成的算法逻辑。
5. **核迭代数看 `Learning iteration X/Y` 或裁决报告"当前迭代数"**，不要把日志行号当迭代数（曾因此误判）。

---

## 5. 配置与调试

- **调试开关**：`DEBUG_SKIP_PRE_TRAIN`（跳过 URDF 诊断/配置/落盘/同步，直接起训；当前有意保持开启）。
- **`cloud/` 本地镜像须手动回传**：`train.py`/`play_eval.py`/`cli_args.py`/`velocity_env_cfg.py`/`mdp/*`。其中 **`play_eval.py` 已插桩**（实测 `achieved_metrics`），**改后必须手动上传**回云端 `scripts/reinforcement_learning/rsl_rl/play_eval.py` 才生效。
- **主要 config 键**（`configs/unified.json` 的 `phase2.*`，详见 `schemas/config.py`）：
  - 续训/预算：`resume_from_best`、`iter_budget_cap_enabled`、`iter_budget_safety_ratio`
  - 硬闸(碎步家族)：`metric_gate_enabled`、`robot_hip_height`、`gate_min_fwd_speed_ratio`、`gate_max_resets`、`gate_max_contact_freq_hz`、`gate_min_swing_time_s`、`gate_min_stride_norm`、`gate_max_bounce_ratio`、`gate_terrain_overrides`
  - 硬闸(碎步以外，默认 `None`=仅观测，标定后填值开闸)：`gate_min_swing_clearance_m`、`gate_max_foot_slip_ratio`、`gate_max_foot_touchdown_cv`、`gate_max_diag_pair_diff`、`gate_max_p95_touchdown_grf_bw`
  - 验收快照：`accept_metrics_enabled`、`accept_min_terrain_levels`、`accept_max_error_vel_xy`、`accept_max_error_vel_yaw`
- **关键状态键**（`schemas/state.py`）：`phase2.train.resume_source`（续训源指针）、`phase2.train.iter_seconds`（预热每步耗时）、`phase2.checkpoint.history` / `phase2.best.checkpoint`（历史 / 冠军，含 `num_video_passed`）。

---

## 6. 演进记录（changelog）

- **2026-06-08**：补最大缺口——最优 `.pt` 跨轮留存交付；warm-start 续训打通（修正 `--resume` 为 store_true）；视频评估升级为 `achieved_metrics` 数值硬闸 + 按地形分档 + 失败标签→旋钮；训练验收快照与早停解耦。
- **2026-06-11**：一整轮真机审计定位"技能不跨轮累积"（续训源恒冻结首轮 `model_3400`、最优的超时轮被丢弃）。修 7 项：续训源解耦(`RESUME_SOURCE`) / 超时轮抢救 / 统一冠军排序(`_champion_rank_key`) / max_iter 时间预算截断 / 续训防发散(edit_limits) / 失败原因不串轮 / 版本号自增写回。
- **2026-06-16**：第二轮真机审计。① **冠军排序键纠偏**——06-11 的键把 `terrain_levels` 排在视频分前、被碎步刷高致劣轮(round1=30)顶替优轮(round0=41.25)、越接力越差；改为视频证据优先、`terrain_levels` 兜底，record 加 `num_video_passed`。② **掉线优雅退出**（`Tee` 吞错 + `os._exit`）。③ **评估系统泛化**——`play_eval` 新增抬脚/打滑/跛行/砸地 4 维（metrics-only）、硬闸重构为声明式规则表 `_gate_rules`、config 新增 5 个阈值（默认 `None`）。
