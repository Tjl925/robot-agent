# robot_agent

`robot_agent` 是一个面向自研四足机械狗 `taili_quad` 的 **多 Agent 闭环自动化配置与训练系统**。基于 **Google ADK** 开发，实现了从本地 URDF 模型诊断、合规配置生成、云端自动化发布同步，到远端长时训练监控、日志裁判趋势早停、视频验收与迭代自愈的完整业务闭环。

---

## 1. 核心设计原则

1. **单入口与单主线**：统一由 `main.py` 启动；专一服务于本地 `taili_quad/` → 云端 `robot_lab/` 唯一主线。
2. **LLM 思考，代码执行**：LLM 仅负责复杂的逻辑推理与趋势诊断（输出严格 JSON），繁琐的流程流转、文件生成与 SFTP 同步由代码确定性处理。
3. **记忆瘦身与物理隔离**：多轮修订历史仅保存极简的文本摘要以杜绝 Token 爆炸。在修订模式 (`revise`) 下，Agent 直接从本地磁盘读取实际生成的 Python 代码喂给 LLM，保证上下文的 100% 精准与高性价比。
4. **100% 常量化管理**：所有跨 Agent 传输的状态键定义于 `state.py`，代码中无任何硬编码字符串键，彻底消除了拼写隐患。

---

## 2. 环境要求

| 依赖项 | 版本要求 |
|---|---|
| Python | ≥ 3.12.13 |
| google-adk | ≥ 2.1.0 |
| httpx | ≥ 0.28.1 |
| openai | ≥ 2.38.0 |
| paramiko | ≥ 5.0.0 |
| pydantic | ≥ 2.13.4 |
| python-dotenv | ≥ 1.2.2 |

---

## 3. 快速开始

### 3.1 安装依赖

```bash
# 推荐使用 uv 进行依赖管理
uv sync
```

### 3.2 环境变量

在项目根目录创建 `.env` 文件，填入以下敏感凭证：

```dotenv
AUTODL_TOKEN=your_autodl_api_token
GOOGLE_API_KEY=your_google_api_key
# ... 其他需要的 key
```

> **注意**：`.env` 已加入 `.gitignore`，不会被提交到版本库。`main.py` 启动时会通过 `python-dotenv` 自动加载。

### 3.3 配置文件

复制并修改配置模板：

```bash
cp configs/unified.example.json configs/unified.json
```

在 `configs/unified.json` 中配置你的 AutoDL Instance UUID、远端训练路径、指标评估规则、早停条件等。主要配置项包括：

| 配置项 | 说明 |
|---|---|
| `phase1.instance_uuid` | AutoDL 实例 UUID |
| `phase2.local_robot_root` | 本地机器人模型根目录（默认 `taili_quad`） |
| `phase2.cloud_robot_lab_root` | 云端 `robot_lab` 根路径 |
| `phase2.train_command_template` | 远端训练启动命令模板 |
| `phase2.play_command_template` | 远端视频渲染命令模板 |
| `phase2.play_timeout_seconds` | 视频渲染超时（默认 600s，勿使用默认 60s） |
| `phase2.max_auto_iterations` | 最大自动迭代轮次 |
| `phase2.max_training_minutes` | 单轮训练最大时长（分钟） |
| `phase2.eval_metric_specs` | 指标评估规格（名称、方向、权重） |
| `phase2.eval_early_stop_rules` | 早停规则定义 |

### 3.4 启动

```bash
python main.py --config configs/unified.json
```

---

## 4. 系统架构

### 4.1 两阶段编排

系统由唯一入口 `main.py` 启动 `OrchestratorAgent`，串联两大阶段：

```
Phase 1 (云端就绪)               Phase 2 (部署训练闭环)
┌──────────────────┐  Handoff   ┌───────────────────────────────────────────┐
│ AutoDL 自动开机  │──SSH凭证──→│ URDF诊断 → 配置生成 → 文件落盘 → 云端发布  │
│ 实例状态轮询     │            │     ↓                                     │
│ Snapshot 提取    │            │ 训练启动 → 增量日志监控 → 日志裁判          │
│ SSH 探活连通     │            │     ↓                                     │
└──────────────────┘            │ 视频渲染 → 视频裁判 → 归档/迭代修复        │
                                └───────────────────────────────────────────┘
```

### 4.2 三态退出协议

以 `STATE_P2_TRAIN_STATUS` 为核心的训练退出协议：

| 状态值 | 含义 | 后续行为 |
|---|---|---|
| `running` | 训练执行中 | Agent 持续增量日志拉取 |
| `early_stopped` | 日志裁判判定发散/崩溃 | 强制杀死远端进程，跳过视频，直接进入 Revise |
| `completed` | 训练正常完成或收敛 | 自动触发 `play.py` 渲染评估视频，流转至视频裁判 |
| `play_failed` | 训练已完成但视频渲染报错 | 拦截熔断，跳过视频评估，直接进入 Revise |

---

## 5. 核心 Agent 图谱

### 流程编排

| Agent | 职责 |
|---|---|
| **`OrchestratorAgent`** | 总编排器：串联 Phase 1 与 Phase 2，负责 SSH 凭证透传 |
| **`Phase1OrchestratorAgent`** | Phase 1 编排器：控制 AutoDL 实例开机、状态轮询、SSH 探活 |
| **`TailiOrchestratorAgent`** | Phase 2 编排器：管理完整的状态转移路由（诊断→归档/HITL） |

### 部署与发布

| Agent | 职责 |
|---|---|
| **`AnalyzeTailiUrdfStepAgent`** | URDF 诊断专家：读取本地 URDF，输出结构化风险报告 |
| **`TailiConfigSynthesisAgent`** | 配置生成大脑：在 `create/revise` 模式下生成 6 个核心 Python 配置代码 |
| **`GenerateTailiFilesStepAgent`** | 本地代码落盘：将 LLM 输出转化为 `.taili_generated/` 下的真实 Python 文件 |
| **`PublishTailiWorkspaceStepAgent`** | 云端发布：递归扫描本地资产，在云端按原相对结构动态建树，通过 SFTP 部署全部资产 |

### 训练与裁决

| Agent | 职责 |
|---|---|
| **`TrainTailiStepAgent`** | 远端异步启动训练，byte-offset 增量日志拉取，定期采样 Checkpoint 投喂裁判，发散时强制早停，训练后自动渲染视频 |
| **`EvaluateTailiTrainingLogAgent`** | 日志裁判：对 Checkpoint 指标趋势做单次无状态判定（`continue` / `stop_failed` / `stop_converged`） |
| **`EvaluateTailiVideoAgent`** | 视频裁判：下载远端评估视频，基于 LLM 做最终通过/不通过判定 |

### 迭代与归档

| Agent | 职责 |
|---|---|
| **`RepairTailiWorkflowStepAgent`** | 故障自愈：收集上一轮失败原因，更新迭代轮次，为下一轮 `revise` 做准备 |
| **`ArchiveTailiOutputsStepAgent`** | 成果归档：评估通过后，将最终配置路径与得分归档 |

---

## 6. 项目结构

```text
D:\robot_agent
├─ main.py                          # 唯一执行入口
├─ pyproject.toml                   # 项目元数据与依赖声明
├─ .env                             # 敏感凭证（不入库）
├─ configs/
│  └─ unified.example.json          # 统一配置文件模板
├─ reference/                       # 参考模板资源
│  ├─ data/Robots/                  # 参考机器人数据
│  └─ robot_lab/
│     ├─ assets/                    # 参考机器人资产模板
│     └─ tasks/                     # 参考任务配置模板
├─ taili_quad/                      # 自研机械狗模型资产（不入库）
│  ├─ urdf/                         # URDF 模型文件
│  ├─ meshes/                       # STL 3D 网格文件
│  └─ .taili_generated/             # LLM 生成的配置代码落盘目录
├─ src/
│  └─ robot_agent/
│     ├─ __init__.py                # 包根
│     ├─ agents/
│     │  ├─ __init__.py             # Agent 统一导出
│     │  ├─ orchestrator.py         # 顶层总编排器
│     │  ├─ phase1_orchestrator.py  # Phase 1 编排器
│     │  ├─ phase1_steps.py         # Phase 1 步骤 Agent 集合
│     │  ├─ taili_orchestrator.py   # Phase 2 编排器（核心路由）
│     │  └─ taili_steps.py          # Phase 2 步骤 Agent 集合（最大模块）
│     ├─ schemas/
│     │  ├─ __init__.py             # Schema 导出
│     │  ├─ config.py               # 类型化配置模型（Pydantic）
│     │  └─ state.py                # 统一状态键常量定义
│     └─ tools/
│        ├─ __init__.py             # 工具导出
│        ├─ autodl_api.py           # AutoDL REST API 调用
│        ├─ llm_client.py           # LLM 客户端封装
│        ├─ ssh_client.py           # SSH 命令执行客户端
│        └─ taili_cloud.py          # SFTP 传输与增量日志核心工具
├─ PROJECT_SUMMARY.md               # 系统核心架构与痛点攻克记录
└─ TESTING_PROGRESS.md              # 端到端功能测试进度
```

---

## 7. 关键技术亮点

### 7.1 byte-offset 增量日志拉取

原方案每次全量拉取日志（长时训练日志数十 MB），导致高带宽消耗与 SSH 超时。现采用 `remote_tail_log(..., byte_offset)` + 远端 `wc -c` / `tail -c +{offset+1}` 的增量拉取机制，每次仅传输几百字节新增日志。

### 7.2 Checkpoint 采样连续性保障

训练监控 Agent 会遍历追加所有 `idx > last_evaluated_iteration` 的 blocks，确保 `metric_history` 数据链 100% 连续，避免长时休眠唤醒后的"断带"问题。

### 7.3 `create` / `revise` 模式协议

- **`create`**：首次部署，基于参考机器人模板 + URDF 诊断报告生成第一版配置
- **`revise`**：训练失败后，从本地磁盘读取当前版本真实代码 + 轻量 `history` 摘要 + `failure_evidence`，定向优化生成新版配置

### 7.4 调试开关

编排器内置两个类级测试开关：

| 开关 | 作用 |
|---|---|
| `DEBUG_SKIP_PRE_TRAIN` | 跳过 URDF 分析、配置生成、文件落盘与云端同步，直接从训练启动开始（适用于云端已就绪的场景） |
| `DEBUG_STOP_BEFORE_VIDEO` | 训练完毕后暂停，在终端美化打印全量 `phase2.*` 状态字典，方便进入视频评估前的终期校验 |

---

## 8. 状态管理

所有跨 Agent 状态键集中定义于 `src/robot_agent/schemas/state.py`，按命名空间划分：

| 命名空间 | 覆盖领域 |
|---|---|
| `phase1.*` | 实例控制、SSH 凭证、重试计数 |
| `phase2.stage` / `phase2.status` | 阶段枚举与总体状态 |
| `phase2.urdf.*` | URDF 诊断结果（有效性、风险等级） |
| `phase2.config.*` | 配置模式、版本、历史摘要 |
| `phase2.train.*` | 训练 PID、日志路径、指标历史、裁判结果 |
| `phase2.play.*` | 视频渲染输出与错误捕获 |
| `phase2.eval.*` / `phase2.video.*` | 评估视频路径、通过判定、得分 |
| `phase2.iteration.*` | 迭代轮次控制 |
| `phase2.hitl.*` | 人工介入标志与响应 |
| `phase2.archive.*` | 最终归档摘要 |

---

## 9. 避坑指南

1. **ADK 状态同步陷阱**：仅通过 `ctx.session.state[KEY] = VALUE` 修改内存状态，在 Agent 退出时可能**丢失**。务必通过 `append_event` 或统一步骤封装进行原子同步。
2. **Stage 越权跳转**：Step Agent 严禁越权修改 `STATE_P2_STAGE` 的最终成败值，所有路由跳转必须由编排器 `taili_orchestrator.py` 集中决策。
3. **远端训练失败排查顺序**：路径与同步问题 → `train_command` 验证 → Checkpoint 通配符匹配 → `play_timeout_seconds` 排查（必须 > 300s）。
4. **路径怀疑优先原则**：出现加载或执行报错时，**永远优先怀疑路径与同步问题**，不要急于修改 LLM 生成的算法逻辑。

---

## 10. 相关文档

- [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) — 系统核心架构、重大技术痛点攻克与设计决策详细记录
- [TESTING_PROGRESS.md](TESTING_PROGRESS.md) — 端到端功能测试进度与里程碑
