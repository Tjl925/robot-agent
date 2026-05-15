from __future__ import annotations

"""会话状态定义（Phase-1 + Phase-2）。

本模块集中定义：
1) 阶段枚举（`Phase1Stage` / `Phase2Stage`）
2) `ctx.session.state` 中使用的统一 key 常量

设计目标：
- 避免在多个文件中硬编码字符串，降低拼写错误风险；
- 保持 phase 命名空间清晰，便于后续 phase3/phase4 扩展；
- 便于你后续做状态可视化、审计回放与故障排查。
"""

from enum import Enum


class Phase1Stage(str, Enum):
    """Phase-1 工作流阶段枚举。"""

    # 尚未执行任何步骤。
    INIT = "init"
    # 已发起 AutoDL 开机请求。
    POWER_ON = "power_on"
    # 正在等待实例进入 running。
    WAIT_RUNNING = "wait_running"
    # 已拉取实例快照并提取 SSH 信息。
    FETCH_SNAPSHOT = "fetch_snapshot"
    # 正在进行 SSH 连通性探测。
    SSH_CONNECT = "ssh_connect"
    # Phase-1 全部步骤成功结束。
    DONE = "done"
    # Phase-1 任意关键步骤失败。
    FAILED = "failed"


class Phase2Stage(str, Enum):
    """Phase-2 工作流阶段枚举。

    注意：
    - 本阶段按你的要求，不包含“训练环境准备”环节；
    - 默认假设远端训练环境已可直接使用（连上即可训练）。
    """

    # 尚未进入 Phase-2。
    INIT = "init"
    # 分析 URDF 结构。
    ANALYZE_URDF = "analyze_urdf"
    # 生成或修订配置。
    SYNTHESIZE_CONFIG = "synthesize_config"
    # 生成本地发布文件。
    GENERATE_FILES = "generate_files"
    # 发布到云端。
    PUBLISH_TO_CLOUD = "publish_to_cloud"
    # 在远端执行训练。
    RUN_TRAINING = "run_training"
    # 评估训练过程中的离散 checkpoint 输出。
    EVALUATE_TRAIN_LOG = "evaluate_train_log"
    # 评估训练完成后的视频证据。
    EVALUATE_VIDEO = "evaluate_video"
    # 收集 checkpoint / video / logs 并评估。
    EVALUATE_RESULT = "evaluate_result"
    # 进入调参 / 修订轮次。
    ITERATE_TUNING = "iterate_tuning"
    # 需要人工介入。
    WAIT_HUMAN = "wait_human"
    # 归档输出。
    ARCHIVE_OUTPUTS = "archive_outputs"
    # 完成。
    DONE = "done"
    # 失败。
    FAILED = "failed"


# ====== 统一状态键（phase1.* 命名空间） ======

# 当前阶段（取值见 Phase1Stage）。
STATE_P1_STAGE = "phase1.stage"
# AutoDL 实例状态（如 running）。
STATE_P1_STATUS = "phase1.status"
# 目标实例 UUID。
STATE_P1_INSTANCE_UUID = "phase1.instance_uuid"
# 当前总重试次数（跨步骤累计）。
STATE_P1_RETRY_COUNT = "phase1.retry_count"
# 最后一次失败原因。
STATE_P1_FAILURE_REASON = "phase1.failure_reason"
# 执行日志列表（字符串数组）。
STATE_P1_EVENTS = "phase1.events"

# SSH 主机地址。
STATE_P1_SSH_HOST = "phase1.ssh.host"
# SSH 端口。
STATE_P1_SSH_PORT = "phase1.ssh.port"
# SSH 用户名。
STATE_P1_SSH_USER = "phase1.ssh.user"
# SSH 密码（注意：生产环境建议脱敏或改为短时凭据）。
STATE_P1_SSH_PASSWORD = "phase1.ssh.password"
# AutoDL 返回的 ssh_command 原始串。
STATE_P1_SSH_COMMAND = "phase1.ssh.command"
# SSH 连通性探测是否成功。
STATE_P1_SSH_CONNECTED = "phase1.ssh.connected"


# ====== 统一状态键（phase2.* 命名空间） ======

# ------------- 控制域 -------------
# 当前阶段（取值见 Phase2Stage）。
STATE_P2_STAGE = "phase2.stage"
# 当前状态（pending/running/succeeded/failed 等）。
STATE_P2_STATUS = "phase2.status"
# 最后一次失败原因。
STATE_P2_FAILURE_REASON = "phase2.failure_reason"
# Phase-2 执行日志列表。
STATE_P2_EVENTS = "phase2.events"

# ------------- URDF 分析域 -------------
# URDF 是否有效。
STATE_P2_URDF_VALID = "phase2.urdf.valid"
# URDF 问题列表。
STATE_P2_URDF_ISSUES = "phase2.urdf.issues"
# 训练可行性风险等级。
STATE_P2_URDF_RISK = "phase2.urdf.risk"

# ------------- 配置域 -------------
# 当前配置模式：create / revise。
STATE_P2_CONFIG_MODE = "phase2.config.mode"
# 当前配置版本序号（纯整数）。
STATE_P2_CONFIG_VERSION = "phase2.config.version"
# 父版本号。
STATE_P2_CONFIG_PARENT_VERSION = "phase2.config.parent_version"
# 历史配置版本列表。
STATE_P2_CONFIG_HISTORY = "phase2.config.history"
# 任务模板名。
STATE_P2_CONFIG_TEMPLATE = "phase2.config.template_name"
# 配置生成后的 JSON 文本。
STATE_P2_CONFIG_TEXT = "phase2.config.generated_text"

# ------------- 训练与评估域 -------------
# 训练 run id。
STATE_P2_TRAIN_RUN_ID = "phase2.train.run_id"
# 训练状态。
STATE_P2_TRAIN_STATUS = "phase2.train.status"
# 训练命令字符串。
STATE_P2_TRAIN_COMMAND = "phase2.train.command"
# 训练标准输出。
STATE_P2_TRAIN_STDOUT = "phase2.train.stdout"
# 训练标准错误。
STATE_P2_TRAIN_STDERR = "phase2.train.stderr"
# 训练退出码。
STATE_P2_TRAIN_EXIT_CODE = "phase2.train.exit_code"
# 评估日志根目录。
STATE_P2_EVAL_LOG_ROOT = "phase2.eval.log_root"
# checkpoint 通配符。
STATE_P2_EVAL_CHECKPOINT_GLOB = "phase2.eval.checkpoint_glob"
# 视频文件名。
STATE_P2_EVAL_VIDEO_NAME = "phase2.eval.video_name"
# checkpoint 路径。
STATE_P2_EVAL_CHECKPOINT_PATH = "phase2.eval.checkpoint_path"
# 视频路径。
STATE_P2_EVAL_VIDEO_PATH = "phase2.eval.video_path"
# 是否通过。
STATE_P2_EVAL_PASSED = "phase2.eval.passed"
# 评估得分卡。
STATE_P2_EVAL_SCORE = "phase2.eval.score_card"
# 失败原因列表。
STATE_P2_EVAL_FAIL_REASONS = "phase2.eval.gate_failed_reasons"
# 训练总迭代次数。
STATE_P2_TRAIN_TOTAL_ITERATIONS = "phase2.train.total_iterations"
# 中间检查间隔（iterations）。
STATE_P2_EVAL_CHECK_INTERVAL = "phase2.eval.check_interval"
# 评估指标定义列表。
STATE_P2_EVAL_METRIC_SPECS = "phase2.eval.metric_specs"
# 早停规则定义列表。
STATE_P2_EVAL_EARLY_STOP_RULES = "phase2.eval.early_stop_rules"
# 最近一次训练检查的迭代点。
STATE_P2_EVAL_LAST_CHECK_ITER = "phase2.eval.last_check_iter"
# 最近一次训练检查摘要。
STATE_P2_EVAL_LAST_CHECK_SUMMARY = "phase2.eval.last_check_summary"
# 是否已触发早停。
STATE_P2_TRAIN_EARLY_STOPPED = "phase2.train.early_stopped"

# ------------- 迭代与 HITL 域 -------------
# 当前迭代轮数。
STATE_P2_ITER_ROUND = "phase2.iteration.current_round"
# 最大迭代轮数。
STATE_P2_ITER_MAX = "phase2.iteration.max_rounds"
# 是否需要人工介入。
STATE_P2_HITL_REQUIRED = "phase2.hitl.required"
# 人工介入原因。
STATE_P2_HITL_REASON = "phase2.hitl.reason"
# 人工反馈内容。
STATE_P2_HITL_RESPONSE = "phase2.hitl.response"
# 人工反馈是否已处理。
STATE_P2_HITL_RESOLVED = "phase2.hitl.resolved"

# ------------- 归档域 -------------
# 归档摘要。
STATE_P2_ARCHIVE_SUMMARY = "phase2.archive.summary"
# 归档是否完成。
STATE_P2_ARCHIVE_COMPLETED = "phase2.archive.completed"
