# robot_agent

`robot_agent` 是一个面向自研四足机械狗 `taili_quad` 的多 Agent 闭环自动化配置与训练系统。它基于 Google ADK 开发，实现了从本地 URDF 模型诊断、合规配置生成、云端自动化发布同步，到远端长时训练监控、日志裁判趋势早停、视频验收与迭代自愈的完整业务闭环。

---

## 1. 核心设计原则
1. **单入口与单主线**：统一由 `run.py` 启动；专一服务于本地 `taili_quad/` -> 云端 `robot_lab/` 唯一主线。
2. **LLM 思考，代码执行**：LLM 仅负责复杂的逻辑推理与趋势诊断（输出严格 JSON），繁琐的流程流转、文件生成与 SFTP 同步由代码确定性处理。
3. **记忆瘦身与物理隔离**：多轮修订历史仅保存极简的文本摘要以杜绝 Token 爆炸。在修订模式 (`revise`) 下，Agent 直接从本地磁盘读取实际生成的 Python 代码喂给 LLM，保证上下文的 100% 精准与高性价比。
4. **100% 常量化管理**：所有跨 Agent 传输的状态键定义于 `state.py`，代码中无任何硬编码字符串键，彻底消除了拼写隐患。

---

## 2. 快速开始

### 2.1 基础配置
建议直接复制并修改配置模板：
```bash
cp configs/unified.example.json configs/unified.json
```
在 `configs/unified.json` 中配置你的 AutoDL API Token、Instance UUID 以及远端训练路径等信息。

### 2.2 启动流程
```bash
python run.py --config configs/unified.json
```

---

## 3. 核心 Agent 图谱

### 流程编排与控制
- **`OrchestratorAgent`** (总编排器)：串联 Phase 1 (AutoDL 开机与 SSH 探活) 与 Phase 2 (云端部署与训练自愈)。负责将 Phase 1 获取的 SSH 凭证无缝透传给 Phase 2。
- **`TailiOrchestratorAgent`** (Phase 2 编排器)：管理整个状态转移路由（URDF诊断 -> 配置生成 -> 云端发布 -> 训练监控 -> 视频评估 -> 迭代修复 -> 归档）。

### 部署与发布 Agent
- **`AnalyzeTailiUrdfStepAgent`**：读取本地机器人 URDF 模型，利用大模型做结构化诊断，输出可训练性风险报告。
- **`TailiConfigSynthesisAgent`**：在 `create` 或 `revise` 模式下，结合本地参考模板，利用大模型生成合规的 6 个核心 Python 配置草案。
- **`GenerateTailiFilesStepAgent`**：将大模型生成的合规代码结构化落盘于本地 `.taili_generated/` 目录下。
- **`PublishTailiWorkspaceStepAgent`**：递归扫描本地资源目录，并在云端按原相对结构动态创建目录树，通过 SFTP 安全部署所有资产文件至 `robot_lab`。

### 训练与裁决 Agent
- **`TrainTailiStepAgent`**：远端异步启动训练并记录 PID。利用 **byte-offset 增量日志拉取技术** 进行超低带宽探活，定期捕获连续的 Checkpoint 数据投喂给日志评估 Agent，在发散时强制杀死远端进程早停。训练结束后，使用配置的独立超时自动渲染评估视频。
- **`EvaluateTailiTrainingLogAgent`** (日志裁判)：对训练产生的离散 Checkpoint 指标历史进行单次无状态趋势判定，输出 `continue` / `stop_failed` (崩溃早停) / `stop_converged` (收敛早停)。
- **`EvaluateTailiVideoAgent`** (视频裁判)：下载云端渲染的 `play.py` 视频文件，并基于大模型对视频表现进行最终判定。

---

## 4. 项目结构
```text
D:\robot_agent
├─ configs/
│  └─ unified.example.json   # 统一配置文件模板
├─ src/
│  └─ robot_agent/
│     ├─ agents/
│     │  ├─ orchestrator.py        # 顶层总编排器
│     │  ├─ phase1_orchestrator.py # Phase 1 编排器
│     │  ├─ taili_orchestrator.py  # Phase 2 编排器
│     │  └─ taili_steps.py         # 核心步骤 Agent 集合
│     ├─ schemas/
│     │  ├─ config.py              # 类型化配置模型
│     │  └─ state.py               # 统一状态键常量定义
│     └─ tools/
│        ├─ ssh_client.py          # SSH 命令执行客户端
│        └─ taili_cloud.py         # SFTP 传输与增量日志核心工具
├─ run.py                    # 唯一执行入口
├─ requirements.txt          # 项目依赖
└─ PROJECT_SUMMARY.md        # 系统核心架构与痛点攻克记录
```
