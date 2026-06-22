# Taili_Quad Agent 系统测试进度 (TESTING_PROGRESS.md)

*最后更新：2026-06-16*

记录 `taili_quad → robot_lab` 自动化系统的端到端测试状态与变更。

## 1. 进度概览

端到端全链路已跑通：自动开机 → 代码生成 → 云端训练 → 日志早停 → Qwen 视频评估 → 自动 revise。Phase 1 全通过；Phase 2 主链路已多轮跑通。后续增强功能**代码已落地、未真机验证**（关键前置：插桩后的 `cloud/play_eval.py` 须先回传云端）。

| 模块 | 状态 | 说明 |
|---|---|---|
| Phase 1（开机/轮询/SSH/Handoff） | ✅ | 全通过 |
| Phase 2 主链路（URDF诊断→配置→落盘→云端发布→训练轮询→日志裁判→视频裁判→revise） | ✅ | 闭环已跑通多轮 |
| 最终归档 `ArchiveTailiOutputsStepAgent` | 🟡 | 已扩展(最优 pt/历史/验收快照)，待端到端确认 |
| warm-start 续训 / 续训源解耦 / 超时抢救 / 最优 .pt 留存 / 冠军排序 / max_iter 截断 | 🟡 | 代码落地，待真机验证（见 §3 清单） |
| 视频数值硬闸·碎步家族 | 🟡 | 离线已验证可拦截(3.229Hz>3.0)，待真机分地形标定 |
| 碎步以外 4 维检测器(clearance/slip/asymmetry/impact) | 🟡 | metrics-only：已实测喂 VLM/修参，硬闸默认关，待训出好策略标定（见 §3.2） |
| 掉线优雅退出 | 🟡 | 待真机复现验证 |

---

## 2. 变更记录

- **2026-06-08**：补最大缺口——最优 `.pt` 跨轮留存交付(`logs/taili_best/`)；warm-start 续训打通（修正 `--resume` 为 store_true 裸旗标 + grep 自检）；视频评估从纯观感升级为 `achieved_metrics` 数值硬闸（否决碎步假通过）+ 按地形分档 + 失败标签→奖励旋钮；训练验收快照与早停解耦。
- **2026-06-11**：一整轮真机审计（round0+3 revise），定位"技能不跨轮累积"——续训源恒冻结首轮 `model_3400`、最优的超时轮(terrain 5.95)被丢弃。修 7 项：续训源解耦 `RESUME_SOURCE` / 超时轮抢救 / 统一冠军排序 `_champion_rank_key` / max_iter 时间预算截断 / 续训防发散 edit_limits / 失败原因不串轮 / 版本号自增写回。新增键 `iter_budget_*`、`resume_source`、`iter_seconds`。
- **2026-06-16**：第二轮真机审计（round0+2 revise，`logs/06-15_18-10.log`）。
  - **A 冠军排序键纠偏**：06-11 的键把 `terrain_levels` 排在视频分前，被碎步刷高致劣轮(round1=30)顶替优轮(round0=41.25)、续训源/best 也被劣轮覆盖、越接力越差。改为视频证据(通过地形数/综合分)优先、`terrain_levels` 兜底，record 加 `num_video_passed`(无视频轮 -1)。
  - **B 掉线优雅退出**：服务器欠费关机致 SSH 断后 `main.py` 不退、刷屏 `Exception ignored in`。修：`Tee` 写入吞错断级联 + `main()` try/finally `os._exit`。
  - **C 评估系统泛化**：`play_eval` 新增 抬脚高度/拖地、触地打滑、跛行不对称、落地砸地 4 维实测（全 **metrics-only**）；硬闸重构为声明式规则表 `_gate_rules`；config 新增 5 个阈值（默认 `None`=禁用）；VLM 提示词 + 旋钮映射补齐。已修对抗校验抓出的坑（足端用 robot 索引空间、质量窄兜底、样本门槛、warmup）。新增键见 §3.2。
  - 附带确认：06-11 版本号 bug 已修复（本轮日志 version 0→1→2 正确）。

---

## 3. 下一步计划

> 代码已落地但**未真机验证**，按下列顺序把 🟡 转 ✅。**第 0 步是前置。**

**0.【前置·必做】回传 `cloud/play_eval.py`**（含 2026-06-16 新增的 4 组实测）到云端 `/root/autodl-tmp/robot_lab/scripts/reinforcement_learning/rsl_rl/play_eval.py`。不回传则 `achieved_metrics` 全部新字段拿不到。

**1. 真机验证清单**（一次完整 revise 闭环里逐项核对）
- **续训接力**（最关键）：第 2+ 轮 warm-start 日志 `from_round/from_status` 指向上一轮更优 checkpoint，不再恒为 round0。
- **冠军排序(A)**：劣轮**不再顶替**续训源/best；`logs/taili_best/` 始终是迄今视频最好那版（看"刷新冠军 checkpoint"日志由 `num_video_passed`/综合分而非 terrain_levels 定胜负）。
- **超时抢救**：制造一次 `train_timeout`，确认绿色"已抢救…最新 checkpoint"且 `RESUME_SOURCE`/best 更新。
- **续训自检/防发散**：绿色"续训已确认"（非红色告警）；LLM 把 LR 降到 0.3~0.5x、未把权重 0→大值突变。
- **max_iter 截断**：LLM 设过大时训练命令被追加 `--max_iterations <可达值>`、自然 completed。
- **最优 .pt 落盘**：`logs/taili_best/` 出 `model_*.pt`+`policy.pt/onnx`+`params/*.yaml`+`BEST_MANIFEST.json`。
- **掉线退出(B)**：复现 SSH 断/欠费关机，确认 `main.py` 干净退出、不刷屏。
- **验收快照键名**：若训练结束的"验收指标快照"行出现 `None`，记下日志里的确切键名，回调 `_grade_training_acceptance` 的子串匹配。

**2. 标定数值硬闸阈值**（物理先验 → 真机校准）
- 碎步家族：拿"碎步策略 + 相对正常策略"各跑一次，对照真实 `achieved_metrics` 校准 `gate_max_contact_freq_hz` 等与 `gate_terrain_overrides`；重点看楼梯是否被误杀。

**3. 终极验收**：调大 `max_auto_iterations`(4~6)，跑完整飞轮（训练→续训→四地形评估→失败打标签→revise 对症改参→再训），把 taili_quad 推向四地形达标 + 验收达线，归档出最优 `.pt`。

### 3.2 碎步以外 4 维检测器：metrics-only → 标定开闸（后续主线提示）

4 个新检测器现在**只观测、不否决**（config 阈值全为 `None`）——阈值是四足运动学先验、未在本管线标定，直接当硬闸会误杀第一个好策略。推进顺序：

1. **先训练、先观测**（现在就能做）：回传新 `play_eval` 后正常跑闭环，新指标进 `eval_meta` 并喂 VLM/修参，但不否决。攒几轮看 `swing_clearance.min/mean`、`foot_slip_ratio`、`foot_touchdown_cv`、`impact.p95_touchdown_grf_bw` 真实范围。
2. **训出一版步态尚可的策略**（关键前提；或拿 Unitree B2 参考策略跑一遍当基准）。
3. **读其分布 → 填 config 开闸**：把对应 `gate_*`（现 `None`）填到"正常值约 0.4×"作下界/上界，填上即**自动开闸**（规则表无需改代码）。
4. **台阶特例**：clearance 本地基线在台阶上跨台阶失真，开 `gate_min_swing_clearance_m` 时务必在 `gate_terrain_overrides` 给 `stairs_up/stairs_down` 置 `None` 禁用（仅 flat/boxes 生效）。
5. **将来加新失败模式**：play_eval 加 1 量 + `_gate_rules` 加 1 行 + config 加 1 阈值，不动 `_metric_gate`。

> 新增配置键（metrics-only，默认 `None`）：`gate_min_swing_clearance_m`、`gate_max_foot_slip_ratio`、`gate_max_foot_touchdown_cv`、`gate_max_diag_pair_diff`、`gate_max_p95_touchdown_grf_bw`。

---

## 4. Backlog（可选，不阻塞主线）
- `stride_length_m` 改为只按前进段计算（现版本被转向/横移拉低、偏噪）。
- 验收快照在 HITL 提示里回显；加"未达验收线则不标 `succeeded`"可选开关。
- 视频 fps 标签问题（回放比真实快约 8%，cosmetic）。
- 碎步 warm-start 螺旋的策略层修法（低熵碎步策略 + 极小 LR 续训难逃局部最优，或需 fresh-restart/提熵；待与上层对齐后再动）。
