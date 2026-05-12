# 机械狗多 Agent 项目总结

## 0. 一句话总览

这是一个基于 Google ADK 的 `taili_quad -> robot_lab` 专用多 Agent 闭环系统：先由总编排器完成 Phase 1 的 AutoDL 开机与 SSH 连通，再由 Phase 2 在同一会话里完成 Taili 配置生成、云端同步、训练、日志/视频评估、失败修订与归档。

## 0.1 当前已落地的主链路

- 唯一 CLI 入口：`run.py`
- 总编排器：`src/robot_agent/agents/orchestrator.py` 作为唯一顶层入口，串联 Phase 1 与 Phase 2，并在 Phase 1 成功后把 `STATE_P1_*` 里的 SSH 结果直接 handoff 给 Phase 2
- Phase 1：`src/robot_agent/agents/phase1_orchestrator.py` 负责 AutoDL 自动开机、状态轮询、snapshot 提取、SSH 连通探活
- Phase 2：`src/robot_agent/agents/taili_orchestrator.py` 负责 Taili 任务接入、URDF 分析、LLM 配置生成、云端同步、训练、日志评估、视频评估、失败后 revise、归档
- 统一配置：`configs/unified.example.json` 作为唯一配置样例，结构分为 `phase1` / `phase2`
- 记忆：通过 `phase2.config.history`、`phase2.config.version`、`phase2.config.parent_version`、`phase2.config.last_changes`、`phase2.config.last_reason` 显式保存多轮修订历史
- 参考模板：首次生成固定参考 `unitree_b2` 模板路径
- 评估：已固定为“训练中按动态间隔检查日志；若异常则 early stop；若全部完成则播放视频并做最终判定”的单一路径
- Phase 2 发布链路已拆分为“本地生成发布文件”与“云端同步发布”两个步骤，职责更清晰
- SSH 复用：Phase 1 得到的 SSH 信息会直接 handoff 给 Phase 2，避免重复维护无用 SSH 字段

## 0.2 后续维护规则

- 以后每次对项目做了有功能价值的改动，都应同步更新本文件，保证下一次切会话时可快速恢复上下文。
- 更新时重点补充：新增/修改的流程、状态键、Agent 职责、远端路径、已知风险、下一步工作。
- 目前已补充中文注释的关键源码文件包括：`src/robot_agent/schemas/config.py`、`src/robot_agent/schemas/state.py`、`src/robot_agent/agents/orchestrator.py`、`src/robot_agent/agents/phase1_orchestrator.py`、`src/robot_agent/agents/taili_orchestrator.py`、`src/robot_agent/agents/taili_steps.py`、`run.py`，后续如果这些文件再发生功能变化，也应同步更新本摘要。

## 1. 项目定位

这是一个基于 **Google ADK** 的分阶段多 Agent 自动化系统，目标是让 LLM Agent 自主完成自研机械狗 `taili_quad` 的接入、配置生成、云端同步、训练评估与迭代修复。

当前项目不是通用机器人平台，而是**只面向一条固定主线**：

- 本地输入：`taili_quad/`
- 云端执行框架：`robot_lab/`
- 云端最终落点：固定的 `robot_lab` 目录结构
- 目标：自动生成配置文件并驱动训练/评估闭环

当前已经形成专用 Taili 系统骨架：
- `TailiCloudConfig`：固定云端路径、固定任务名、固定训练/播放命令模板、评估所需指标与 Phase-1 handoff 后的远端连接信息
- `taili_steps.py`：专用步骤 Agent 集合
- `taili_orchestrator.py`：Phase 2 专用编排器
- `phase1_orchestrator.py`：Phase 1 专用编排器
- `orchestrator.py`：总编排器
- `run.py`：唯一 CLI 入口
- `configs/unified.example.json`：统一配置样例

这套系统的核心约束已经明确：
- 只服务于 `taili_quad` 这一条固定链路
- 不做多机器人泛化
- 所有最终生成物必须同步到云端 `robot_lab` 的固定路径

---

## 2. 已完成与正在推进的能力

### 2.1 Phase 1（已完成、可用）

目标：把 AutoDL 实例启动并验证 SSH 可用。

已实装流程：
- `power_on`：开机
- 轮询状态直到 `running`
- 拉取 snapshot 获取 SSH 信息
- 执行 SSH 连通性探测
- 步骤级重试
- 最终收敛 `done / failed`

关键实现要求：
- 关键状态变化必须通过 `append_event + state_delta` 提交
- 避免只修改 `ctx.session.state` 但最终 state 不落库的问题

### 2.2 总编排器（已落地）

目标：把 Phase 1 和 Phase 2 串成一个统一的闭环。

已实装职责：
- 先执行 Phase 1
- 在 Phase 1 成功后提取 SSH 结果
- 把 SSH 信息 handoff 给 Phase 2
- 再执行 Phase 2
- 统一输出最终 session state

### 2.3 Taili 专用系统（当前主线，已落地到可运行骨架）

目标：围绕 `taili_quad -> robot_lab` 完成以下闭环：
1. 读取本地机器人模型与资源
2. 由 LLM 对 URDF 做结构化诊断（`valid / risk / issues / summary / recommendation`）
3. 由 LLM 生成 `robot_lab` 资产层、任务层、训练配置草案（严格 JSON）
4. 同步到云端固定路径
5. 在云端启动训练，按固定间隔轮询离散 checkpoint 输出
6. 由日志评估 Agent 判断是否 early stop
7. 训练完整结束后自动执行 `play.py`，并下载云端视频交给视频评估 Agent
8. 根据视频判定结果自动迭代或进入 HITL
9. 归档结果摘要

当前已形成的角色分工：
- `AnalyzeTailiUrdfStepAgent`：LLM URDF 结构化诊断 Agent（严格 JSON 输出）
- `TailiConfigSynthesisAgent`：LLM 配置生成 Agent（支持 `create / revise`，严格 JSON 输出）
- `TrainTailiStepAgent`：训练执行 Agent，负责启动训练、按固定间隔轮询离散 checkpoint 输出，并在需要时立刻 early stop
- `EvaluateTailiTrainingLogAgent`：日志评估 Agent，只看单个离散 checkpoint 的原始输出，判断是否 early stop
- `EvaluateTailiVideoAgent`：视频评估 Agent，负责下载云端视频并做最终视频判定
- `TailiOrchestratorAgent`：Phase 2 总编排器
- `OrchestratorAgent`：Phase 1 + Phase 2 总编排器
- `TailiCloudTool`：远端同步/扫描/渲染工具
- Phase 2 已拆为“本地生成发布文件”与“云端发布”两个独立步骤，职责边界更清楚

当前这三个 LLM Agent 的形式已经基本符合 ADK 官方推荐的 `LlmAgent` 思路：
- `instruction` 明确约束职责与输出格式；
- `output_schema` 强制输出严格 JSON；
- `input_schema` / 结构化上下文把模型需要看的事实整理清楚；
- `generate_content_config` 通过较低温度提升稳定性。

为了方便后续维护，我把它们的架构边界进一步收敛成了“输入 / 输出 / 职责 / 兜底”四层说明（已同步到 `README.md`）：
- `AnalyzeTailiUrdfStepAgent`：只负责 URDF 诊断，不改文件；
- `TailiConfigSynthesisAgent`：只负责配置草案生成，不同步、不训练；
- `TrainTailiStepAgent`：只负责训练执行与早停控制，不做最终视频裁决；
- `EvaluateTailiTrainingLogAgent`：只负责单个离散 checkpoint 的日志判定；
- `EvaluateTailiVideoAgent`：只负责视频下载与最终视频裁决。

但也要注意：它们目前仍然保留了少量 Python 侧的兜底逻辑，用于证据整理、状态落库和失败保护；也就是说，它们已经很像官方推荐的“LLM 思考型 Agent”，但还不是完全纯粹的“只靠模型推理、完全不含业务兜底”的最小实现。

当前已明确的评估证据：
- 训练过程中按离散 checkpoint 块采样得到的原始日志输出
- 训练完成后的最终视频判定
- checkpoint 路径与训练产物定位信息

当前已明确的配置记忆能力：
- `phase2.config.mode`
- `phase2.config.version`
- `phase2.config.parent_version`
- `phase2.config.reference_robot`
- `phase2.config.history`
- `phase2.config.last_changes`
- `phase2.config.last_reason`

---

## 3. 当前工程结构

```text
D:\robot_agent
├─ configs/
│  └─ unified.example.json
├─ src/
│  └─ robot_agent/
│     ├─ __init__.py
│     ├─ agents/
│     │  ├─ __init__.py
│     │  ├─ orchestrator.py
│     │  ├─ phase1_orchestrator.py
│     │  ├─ phase1_steps.py
│     │  ├─ taili_orchestrator.py
│     │  └─ taili_steps.py
│     ├─ schemas/
│     │  ├─ config.py
│     │  └─ state.py
│     └─ tools/
│        ├─ __init__.py
│        ├─ autodl_api.py
│        ├─ ssh_client.py
│        └─ taili_cloud.py
├─ run.py
├─ requirements.txt
├─ pyproject.toml
├─ PROJECT_SUMMARY.md
└─ .cursor/rules/
   ├─ robot-agent-core.mdc
   ├─ robot-agent-commands.mdc
   └─ robot-agent-taili-cloud.mdc
```

---

## 4. 已确认的目录与框架事实

### 4.1 `taili_quad/`

当前已确认：
- 机器人 URDF 位于 `taili_quad/urdf/robot.urdf`
- URDF 引用了 `taili_quad/meshes/` 下的多个 STL 资源
- 机器人是标准四足结构，包含：
  - `base_link`
  - `FL / FR / RL / RR` 四条腿
  - 每条腿包含 hip / thigh / calf / foot

### 4.2 `robot_lab/`

当前已确认：
- `source/robot_lab/robot_lab/assets/` 是资产定义层
- `source/robot_lab/robot_lab/tasks/` 是任务与训练配置层
- `tasks/manager_based/locomotion/velocity/config/quadruped/` 中已有大量机器人模板
- 典型结构包括：
  - `__init__.py`：gym 注册
  - `rough_env_cfg.py`：主环境模板
  - `flat_env_cfg.py`：平地派生模板
  - `agents/`：训练算法配置

### 4.3 当前专用 Taili 系统

当前代码中已经保留的专用模块：
- `src/robot_agent/schemas/config.py` 中的 `TailiCloudConfig`
- `src/robot_agent/schemas/config.py` 中的 `TailiConfigDraft`
- `src/robot_agent/schemas/config.py` 中的 `TailiJudgeResult`
- `src/robot_agent/agents/orchestrator.py`
- `src/robot_agent/agents/phase1_orchestrator.py`
- `src/robot_agent/agents/taili_orchestrator.py`
- `src/robot_agent/agents/taili_steps.py`
- `src/robot_agent/tools/taili_cloud.py`
- `run.py`
- `configs/unified.example.json`

这套专用系统只面向 `taili_quad -> robot_lab` 单一固定链路，不保留通用 Phase2 泛化入口。

---

## 5. 当前已经明确的系统设计方向

### 5.1 这不是“脚本式流程”

这个项目的核心不是让一个脚本完成所有事情，而是让多个 LLM Agent 协作：
- 分析目录
- 识别机器人结构
- 由 LLM 生成配置草案
- 同步到云端
- 验证注册与训练入口
- 根据结果迭代修复

### 5.2 这不是通用框架

当前明确只服务于：
- 自研机械狗 `taili_quad`
- 固定云端 `robot_lab`
- 固定同步路径
- 固定训练闭环

这意味着设计应优先保证稳定性、可审计性和可恢复性，而不是追求通用抽象。

### 5.3 配置生成与评估的 Agent 设计原则

配置生成 Agent：
- 首次运行使用 `create` 模式
- 失败后使用 `revise` 模式
- 参考固定模板机器人（例如 Unitree）+ Taili URDF + 任务目标
- 输出结构化 `TailiConfigDraft`
- 通过 `phase2.config.history` 记住多轮变化

评估 Judge Agent：
- 中间阶段只看日志摘要与关键指标趋势
- 训练完成后再看视频做最终判断
- 输出 `TailiJudgeResult`
- 结果进入 `phase2.eval.score_card`
- 失败时驱动配置进入 `revise`

---

## 6. 关键约束与规范

### 6.1 ADK 状态写入规范（非常重要）

- 关键状态变更必须通过 `append_event + EventActions(state_delta=...)` 提交
- 不能只在内存中改 `ctx.session.state` 后依赖其作为最终状态

### 6.2 枚举规范

- 阶段枚举统一使用 `Phase1Stage` / `Phase2Stage`
- 不要滥用 `.value`

### 6.3 状态命名规范

- 所有状态键集中在 `src/robot_agent/schemas/state.py`
- 使用命名空间：
  - `STATE_P1_*` 对应 `phase1.*`
  - `STATE_P2_*` 对应 `phase2.*`

### 6.4 工程协作规范

- 项目结构保持：`schemas / tools / agents / run.py`
- 中文注释必须完整，尤其是模块说明与关键分支原因
- 敏感信息不得写死在源码里
- 配置文件统一使用 JSON

### 6.5 云端同步规范

- 本地可以生成草稿
- 最终生效版本必须在云端 `robot_lab` 中
- 路径必须固定，不做通用机器人扩展

---

## 7. Taili 专用 Agent 方向（当前实现中）

- `IntakeTailiTaskStepAgent`：任务接入
- `AnalyzeTailiUrdfStepAgent`：URDF 体检
- `TailiConfigSynthesisAgent`：LLM 配置生成（create/revise）
- `GenerateTailiFilesStepAgent`：生成发布计划
- `PublishTailiToCloudStepAgent`：云端同步/发布
- `EvaluateTailiJudgeAgent`：日志检查 + 视频最终判定
- `RepairTailiWorkflowStepAgent`：迭代修复
- `ArchiveTailiOutputsStepAgent`：结果归档

---

## 8. 常用命令

### 本地开发
- 安装依赖：`pip install -e .`
- 运行统一流程：`python run.py --config configs/unified.example.json`

### 远端 `robot_lab` 训练常见形式
- `python scripts/reinforcement_learning/rsl_rl/train.py --task=<TASK_NAME> --headless`
- `python scripts/reinforcement_learning/rsl_rl/play.py --task=<TASK_NAME> --headless --video`

---

## 9. 注意事项与坑点

- 若出现“最终 state 未更新”，优先检查是否漏了 `append_event + state_delta`
- 若阶段状态异常跳转，检查是否有非编排器代码直接改了 `phase*.stage`
- Taili 训练失败时优先看：
  - `phase2.train.command`
  - `phase2.eval.checkpoint_path`
  - `phase2.eval.video_path`
  - `phase2.eval.score_card`
- 评估失败时优先检查：
  - 远端 `logs/` 是否存在有效 run
  - checkpoint 通配符是否匹配
  - `play.py` 是否能正常跑完
- 处于 `wait_human` 时，如果配置了 `hitl_response_text`，应自动恢复进入下一轮
- 路径问题优先怀疑同步问题，不要先怀疑算法本身

---

## 10. 下一步最合理的推进顺序

1. 按固定主链路继续完善 `TailiConfigSynthesisAgent` 的输入/输出契约
2. 让 `EvaluateTailiJudgeAgent` 的日志判定更稳、更可解释
3. 接入真实的 unitree 参考模板检索与差异化生成
4. 完善本地生成文件到云端固定路径的真实同步落盘
5. 接入真实训练/播放日志并输出更可信的评估结论
6. 再继续迭代 reward / observation / hyperparams

---

## 11. `TailiConfigSynthesisAgent` 的 `create / revise` 输入协议

这是后续最重要的一条配置生成规范，建议严格按这个协议继续推进。

### 11.1 `create` 模式
适用场景：
- 第一次生成 Taili 配置
- 还没有可用的 Taili 历史版本

输入内容：
- `reference_robot`：固定参考机器人，例如 unitree 某个四足模板
- `taili_urdf` 或 URDF 摘要
- `task_goal`：任务目标，如速度控制
- `fixed_cloud_paths`：云端固定落点
- `constraints`：只面向 `taili_quad -> robot_lab`

输出目标：
- 第一版 `TailiConfigDraft`
- 包含 `asset_code / task_init_code / task_cfg_code / reward / hyperparams`
- 同时生成 `assumptions / risk_flags`

### 11.2 `revise` 模式
适用场景：
- 训练失败
- 评估未通过
- 导入/注册报错
- Judge 给出需要修订的结论

输入内容：
- `current_draft`：当前版本配置
- `parent_version`：上一版版本号
- `history`：历史修改记录
- `failure_evidence`：checkpoint / video / stderr / 日志摘要
- `judge_result`：Judge 的结构化输出

输出目标：
- 仅做局部修订
- 返回新的 `TailiConfigDraft`
- 明确 `changed_fields`
- 明确 `change_reason`

### 11.3 记忆规则
- 不靠 LLM 自带长期记忆
- 所有历史通过 `phase2.config.history` 和版本字段回放
- 每次 revise 都要记录：
  - 改了什么
  - 为什么改
  - 父版本是谁
  - 这次失败证据是什么

### 11.4 切换规则
- 第一次运行默认 `create`
- 一旦 Judge 判定失败或 HITL 触发，就切到 `revise`
- 后续每轮都基于历史版本继续修订

---

## 12. 给后续会话的简短提示

如果要继续推进，可以直接说明：

> 继续做 `taili_quad -> robot_lab` 专用多 Agent 系统，只针对这一条固定机械狗链路，所有生成物最终同步到云端 `robot_lab` 固定路径，并且保留多轮配置历史以支持 create/revise。
