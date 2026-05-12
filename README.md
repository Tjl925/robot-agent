# robot_agent

`robot_agent` 是一个面向自研机械狗 `taili_quad` 的多 Agent 自动化系统。它基于 Google ADK，把从本地模型接入、配置生成、云端发布，到训练评估、视频验收与失败修订的流程收敛成一条固定闭环。

当前项目已经明确收敛为：
- **单入口**：`run.py`
- **单总编排器**：`src/robot_agent/agents/orchestrator.py`
- **单配置文件**：统一 `phase1` / `phase2` 的 JSON 配置
- **单业务主线**：本地 `taili_quad/` -> 云端 `robot_lab/`

## 这套系统做什么

这套系统的目标不是做一个通用机器人平台，而是稳定地服务于一条固定任务链路：

1. Phase 1 先把 AutoDL 实例拉起并确认 SSH 可用。
2. Phase 2 在同一会话里完成 Taili 任务接入、URDF 诊断、配置生成、云端发布、训练评估、失败修订与归档。
3. 所有关键过程都保留结构化状态，便于审计、回放和下一轮修订。

## 快速开始

1. 准备统一配置文件

```json
{
  "phase1": { /* AutoDL 开机与 SSH 探活配置 */ },
  "phase2": { /* Taili 闭环配置 */ }
}
```

建议直接复制 `configs/unified.example.json`，然后填写你的真实参数。

2. 运行系统

```bash
python run.py --config configs/unified.example.json
```

## 项目结构

- `run.py`：唯一 CLI 入口
- `src/robot_agent/agents/orchestrator.py`：总编排器，串联 Phase 1 + Phase 2
- `src/robot_agent/agents/phase1_orchestrator.py`：Phase 1 编排器
- `src/robot_agent/agents/taili_orchestrator.py`：Phase 2 编排器
- `src/robot_agent/agents/taili_steps.py`：Phase 2 的核心步骤 Agent 集合
- `configs/unified.example.json`：统一配置示例
- `PROJECT_SUMMARY.md`：项目架构与最新进展摘要

## 主流程概览

### Phase 1
Phase 1 负责 AutoDL 实例准备与连通性验证：
- 开机
- 等待实例进入 `running`
- 拉取 snapshot
- 提取 SSH 信息
- 执行 SSH 探活

### Phase 2
Phase 2 负责 Taili 到 robot_lab 的闭环：
- `AnalyzeTailiUrdfStepAgent` 对 URDF 做结构化诊断
- `TailiConfigSynthesisAgent` 生成严格 JSON 的配置草案
- `GenerateTailiFilesStepAgent` 在本地生成发布文件
- `PublishTailiWorkspaceStepAgent` 同步到云端并准备训练
- `TrainTailiStepAgent` 负责启动训练、轮询离散 checkpoint 输出，并在需要时立刻 early stop
- `EvaluateTailiTrainingLogAgent` 只分析单个 checkpoint 的原始训练输出，给出 early-stop 建议
- `EvaluateTailiVideoAgent` 在训练完成后自动拉取云端视频并做最终视频判定
- `RepairTailiWorkflowStepAgent` 根据失败证据推进 revise
- `ArchiveTailiOutputsStepAgent` 归档最终结果

### 评估策略
系统采用固定的两段式评估：
- 训练过程中按动态间隔检查日志文件里的离散 checkpoint 块，必要时由训练 agent 立刻 early stop
- 训练完整结束后自动执行 `play.py`，再下载云端视频交给视频评估 agent 做最终判定

## 设计原则

- **LLM 负责推理，代码负责编排**：模型输出结构化结果，代码负责校验、落状态和兜底。
- **严格 JSON 输出**：URDF 诊断、配置生成和评估判定都输出结构化 JSON，便于自动消费。
- **可追踪的版本历史**：多轮修订历史保存在 `phase2.config.history`，并配合 `phase2.config.version` / `phase2.config.version_id` / `phase2.config.parent_version` 记录版本演化。
- **职责边界清晰**：生成、发布、评估、归档分成独立步骤，方便维护和回放。
- **固定业务范围**：当前只服务于 `taili_quad -> robot_lab`，不做通用泛化。

## 核心 Agent 说明

### `AnalyzeTailiUrdfStepAgent`
负责读取本地 URDF 和上下文，输出 `TailiUrdfAnalysisResult`，用于判断机器人结构是否适合进入后续配置和训练闭环。

### `TailiConfigSynthesisAgent`
负责在 `create / revise` 模式下生成 `TailiConfigDraft`，包含资产代码、任务配置、reward、hyperparams 和修订说明。

### `GenerateTailiFilesStepAgent`
负责在本地生成发布文件，不做云端同步。

### `PublishTailiWorkspaceStepAgent`
负责把本地生成物和 URDF 同步到云端 `robot_lab`，并准备后续训练执行。

### `TrainTailiStepAgent`
负责启动训练、按固定间隔轮询训练日志、从新增日志中提取最新完整 checkpoint 块，并在日志评估 agent 建议 early stop 时立刻终止训练。

### `EvaluateTailiTrainingLogAgent`
负责只看单个离散 checkpoint 的原始输出，判断当前训练是否需要 early stop。

### `EvaluateTailiVideoAgent`
负责在训练完成后下载云端生成的视频文件，并基于视频证据判断是否真正通过。

## 运行时状态与记忆

系统使用 `ctx.session.state` 保存关键状态，重点包括：
- Phase 1 的实例与 SSH 信息
- Phase 2 的任务、URDF、配置、训练与评估状态
- `phase2.config.history`：配置历史
- `phase2.hitl.required` / `phase2.hitl.reason`：人工介入状态
- `phase2.train.log_input_payload` / `phase2.train.log_judge_result`：离散 checkpoint 评估输入与裁决结果
- `phase2.video.input_payload` / `phase2.video.judge_result`：视频下载与最终判定证据
- `phase2.archive.summary`：归档摘要

## 备注

- Phase 1 得到的 SSH 信息会直接 handoff 给 Phase 2。
- 当前系统只面向 `taili_quad -> robot_lab` 这条固定链路。
- 如需扩展，请优先更新 `PROJECT_SUMMARY.md` 和对应源码注释。
