# 机械狗多 Agent 项目总结 (PROJECT_SUMMARY.md)

## 1. 项目定位与核心主线
这是一个基于 **Google ADK** 的专用多 Agent 闭环自动化系统，**只面向一条固定主线**：
- **本地输入**：`taili_quad/` (URDF 模型与 STL 资源)
- **云端环境**：AutoDL 实例上的 `robot_lab` 强化学习框架
- **最终目标**：LLM Agent 自主诊断 URDF、生成 PPO 算法与环境配置、同步到云端、驱动训练/自适应评估闭环，并在失败时自动迭代修复 (Revise) 或触发人工介入 (HITL)。

---

## 2. 核心架构与控制链路

### 2.1 顶层编排 (Phase 1 → Phase 2 Handoff)
- **唯一入口**：`run.py` 启动 `OrchestratorAgent`。
- **Phase 1**：`Phase1OrchestratorAgent` 自动控制 AutoDL 实例开机、探活，并将获取的 SSH 凭证 (`STATE_P1_SSH_*`) 无缝透传 (Handoff) 给 Phase 2。
- **Phase 2**：`TailiOrchestratorAgent` 承接 SSH 凭证，在同一 Session 内驱动完整的机器人部署与训练评估流。

### 2.2 状态协议与三态退出 (Unified State Protocol)
系统废弃了所有散乱的布尔退出标志，确立了以 `STATE_P2_TRAIN_STATUS` 为核心的**退出协议**：
1. `"running"`：远端训练执行中，Agent 进行增量日志提取。
2. `"early_stopped"`：日志裁判判定训练发散/崩溃，直接杀死远端进程，不渲染视频，标记评估失败并进入 Revise。
3. `"completed"`：训练自然达到最大步数或被日志裁判判定为已收敛。训练 Agent 自动触发 `play.py` 渲染评估视频，并流转至视频裁判进行终期判定。
4. `"play_failed"`：训练已完成但视频渲染命令报错，被及时拦截熔断，不执行视频评估，直接进入 Revise。

---

## 3. Phase2 核心 Agent 职责分工

| Agent 名称 | 职责定位 | 输入/输出协议 |
| :--- | :--- | :--- |
| **AnalyzeTailiUrdfStepAgent** | URDF 诊断专家 | 诊断机械狗结构的完整性与潜在训练风险。输出 `TailiUrdfAnalysisResult` JSON。 |
| **TailiConfigSynthesisAgent** | 配置生成大脑 | 在 `create/revise` 模式下生成 6 个核心 Python 配置代码。输出 `TailiConfigDraft` JSON。 |
| **GenerateTailiFilesStepAgent** | 本地代码落盘 | 将大模型生成的配置代码写入本地 `.taili_generated/` 目录。 |
| **PublishTailiWorkspaceStepAgent** | 确定性云端发布 | 递归扫描本地资产，在云端按原相对结构动态建树，并通过 SFTP 部署全部资产。 |
| **TrainTailiStepAgent** | 训练监控与早停控制 | 远端异步启动训练，提取增量日志，定期提取 checkpoint 指标，并在日志裁判建议早停时强行中止。最后渲染视频。 |
| **EvaluateTailiTrainingLogAgent** | 日志采样趋势裁判 | 接收 `metric_history` 指标趋势，进行单次无状态判断。输出 `TailiTrainingLogJudgeResult`。 |
| **EvaluateTailiVideoAgent** | 视频终审裁判 | 下载远端评估视频，进行最终视觉通过判定。输出 `TailiVideoJudgeResult`。 |
| **RepairTailiWorkflowStepAgent** | 故障自愈迭代控制器 | 收集上一轮失败原因，更新迭代轮次，为下一轮配置 Revise 做准备。 |
| **ArchiveTailiOutputsStepAgent** | 成果归档器 | 评估通过后，将最终配置路径与得分归档至 `STATE_P2_ARCHIVE_SUMMARY`。 |

---

## 4. 重大技术痛点攻克与优化记录

### 4.1 P0 级：增量日志拉取性能革命
- **痛点**：原 `remote_tail_log` 每次使用 Python 脚本读取全量日志并回传。在长时训练中（日志常达数十 MB），会导致高带宽消耗与 SSH 连接频繁超时崩溃。
- **解决**：改写为**字节偏移增量拉取机制**。新签名 `remote_tail_log(..., byte_offset)` 配合远端 `wc -c` 获取大小与 `tail -c +{offset+1}` 增量输出，每次仅拉取几百字节的新日志。同时完全移除了嵌入式 Python，避免了 `.bashrc` 输出的干扰。

### 4.2 P0 级：视频渲染超时防御
- **痛点**：视频渲染 `play.py` 通常耗时 2~5 分钟。原代码使用默认 of 60s SSH 超时，导致渲染步骤 100% 必然超时报错。
- **解决**：在配置中引入独立的 `play_timeout_seconds`（默认 600s），为视频录制提供充足的缓冲时间。

### 4.3 P1 级：检查点采样连续性修复
- **痛点**：原 `TrainTailiStepAgent` 在长时休眠唤醒后，仅处理新增日志正则捕获的最后一个 block (`blocks[-1]`)，跳过了中间产生的多个 checkpoint 指标，导致裁判接收的指标趋势产生“严重断带”。
- **解决**：重构为遍历追加所有 `idx > last_evaluated_iteration` 的 blocks，确保 `metric_history` 数据链 100% 连续。

### 4.4 架构级：上下文瘦身与物理隔离
- **痛点**：多轮 `revise` 时如果把全量历史代码塞入 prompt，会导致 Token 爆炸并干扰大模型生成。
- **解决**：`phase2.config.history` 仅保存极简的“修改说明与版本号”摘要。配置生成 Agent 在 `revise` 模式下直接通过 Python 读取本地 `.taili_generated/` 真实代码文件并喂给 LLM。成功实现上下文瘦身与精确代码复用的完美平衡。

### 4.5 规范级：100% 常量化与零硬编码
- 彻底清理了 `state.py` 中 14 个冗余废弃状态键，新增 13 个专用常量，并把 `taili_steps.py` 和 `taili_orchestrator.py` 中的所有硬编码 `"phase2.train.xxx"` 字符串替换为统一常量，消除了拼写隐患。

---

## 5. 重大架构设计决策与 create / revise 协议

### 5.1 为什么拒绝在 State 堆积历史代码（架构决策）
在早期版本中，系统尝试将历史生成的全量 Python 代码塞入 `ctx.session.state["phase2.config.history"]` 中以保留多轮记忆。这导致：
1. 随着迭代轮次增加，Session 存储大小呈指数级爆炸，严重浪费内存；
2. 每次状态同步和 API 调用都需要处理巨量文本，大模型经常产生幻觉或直接超出 Context Window。
**最终决策**：**物理隔离设计**。把 State 里的 `history` 改为只保留轻量级的“差异说明、版本号和改动原因”。Agent 在 `revise` 模式下，直接由 Python 代码在本地磁盘读取旧版本真实代码，组合成紧凑的 Prompt 传给 LLM。

### 5.2 `create` 与 `revise` 模式的明确契约

#### 5.2.1 `create` 模式
- **适用场景**：第一次部署 Taili 配置，本地尚未产生合规的历史代码。
- **核心输入**：
  - `reference_robot`：固定参考机器人（如 `unitree_b2`）的结构及参数模板
  - `taili_urdf`：自研机械狗 URDF 模型原始文本与诊断报告
  - `task_goal`：任务目标（如 velocity locomotion）
- **核心输出**：第一版 `TailiConfigDraft`（含 6 个核心配置代码）。

#### 5.2.2 `revise` 模式
- **适用场景**：前一轮训练失败、日志报错或被视频裁判判定为不通过。
- **核心输入**：
  - 本地磁盘读取的当前版本真实代码文件
  - `parent_version` 与轻量版 `history` 摘要
  - `failure_evidence`：训练中捕获的 stderr 报错、断点指标数据或视频裁判给出的故障细节
- **核心输出**：基于旧代码定向优化的新 `TailiConfigDraft`。

---

## 6. 避坑指南与生产环境排查路径 (核心资产)

### 6.1 ADK 状态同步陷阱（重大踩坑点）
- **现象**：在 Agent 代码中，如果仅通过 `ctx.session.state[KEY] = VALUE` 修改了内存中的状态，但在 Agent 退出或退出前没有显式调用编排器的 `await self._commit_state(ctx)` 或者 `append_event(..., state_delta=...)`，这些状态在多阶段序列化时会**彻底丢失**。
- **规避**：任何对状态键（如训练状态、评估结果）的重要变更，务必通过 `append_event` 或统一的步骤封装进行原子同步。

### 6.2 状态非法跳转与 Stage 控制陷阱
- **现象**：非编排器（Orchestrator）的普通 Step Agent 如果越权、大跨度地在代码里直接改写 `ctx.session.state[STATE_P2_STAGE]` 等阶段枚举，会导致顶层 Orchestrator 路由状态机逻辑错乱，产生死循环或直接漏步。
- **规避**：Step Agent 只负责在自己的职责内将 `STATE_P2_STAGE` 改为对应的当前运行阶段，所有的后续跳转决策（成功、失败、 revise、HITL）必须由编排器 `taili_orchestrator.py` 做集中路由，严禁 Step Agent 越权修改最终成败状态。

### 6.3 远端训练失败排查路径
当远端训练启动失败或未正常轮询到指标时，按以下优先级排查：
1. **优先排查同步原因**：检查本地生成的配置文件是否成功同步到云端 `robot_lab`。路径不对、文件名拼写错误或云端权限问题是 90% 训练报错的根源。
2. **检查 `phase2.train.command`**：通过 SSH 手动执行该命令，确认在云端是否能正常 import 对应模块。
3. **检查 Checkpoint 通配符匹配**：确认云端产生的日志文件是否能正常被 Train Agent 的 `remote_tail_log` 捕获。
4. **排除 `play.py` 超时**：确保 `play_timeout_seconds` 大于 300s，渲染不能使用 60s 默认超时。

### 6.4 路径怀疑优先原则
在多 Agent 系统调试中，一旦出现加载或执行报错，**永远优先怀疑路径与同步问题，不要急于修改大模型生成的算法逻辑**。95% 以上的问题都是由于 SFTP 上传没有对齐云端 `robot_lab` 的预期目录结构造成的。

---

## 7. 快速调试与测试说明
为方便断点调试，编排器中集成了两个类级测试开关：
1. **`DEBUG_SKIP_PRE_TRAIN`** (`bool`)：设为 `True` 将跳过 URDF 分析、配置生成、文件落盘与云端同步，**直接从训练启动开始运行**（适用于云端配置已准备就绪的场景）。
2. **`DEBUG_STOP_BEFORE_VIDEO`** (`bool`)：设为 `True` 将在训练与日志评估通过、视频渲染完毕后暂停，**在终端以高亮美化格式打印全量 `phase2.*` 状态字典**，然后立即返回，方便开发者在进入视频 LLM 评估前对状态进行终期校验。
