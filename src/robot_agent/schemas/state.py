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

    INIT = "init"
    POWER_ON = "power_on"
    WAIT_RUNNING = "wait_running"
    FETCH_SNAPSHOT = "fetch_snapshot"
    SSH_CONNECT = "ssh_connect"
    DONE = "done"
    FAILED = "failed"


class Phase2Stage(str, Enum):
    """Phase-2 工作流阶段枚举。"""

    INIT = "init"
    ANALYZE_URDF = "analyze_urdf"
    SYNTHESIZE_CONFIG = "synthesize_config"
    GENERATE_FILES = "generate_files"
    PUBLISH_TO_CLOUD = "publish_to_cloud"
    RUN_TRAINING = "run_training"
    EVALUATE_TRAIN_LOG = "evaluate_train_log"
    EVALUATE_VIDEO = "evaluate_video"
    ITERATE_TUNING = "iterate_tuning"
    WAIT_HUMAN = "wait_human"
    ARCHIVE_OUTPUTS = "archive_outputs"
    DONE = "done"
    FAILED = "failed"


# ====== 统一状态键（phase1.* 命名空间） ======

STATE_P1_STAGE = "phase1.stage"
STATE_P1_STATUS = "phase1.status"
STATE_P1_INSTANCE_UUID = "phase1.instance_uuid"
STATE_P1_RETRY_COUNT = "phase1.retry_count"
STATE_P1_FAILURE_REASON = "phase1.failure_reason"
STATE_P1_EVENTS = "phase1.events"

STATE_P1_SSH_HOST = "phase1.ssh.host"
STATE_P1_SSH_PORT = "phase1.ssh.port"
STATE_P1_SSH_USER = "phase1.ssh.user"
STATE_P1_SSH_PASSWORD = "phase1.ssh.password"
STATE_P1_SSH_COMMAND = "phase1.ssh.command"
STATE_P1_SSH_CONNECTED = "phase1.ssh.connected"


# ====== 统一状态键（phase2.* 命名空间） ======

# ------------- 控制域 -------------
STATE_P2_STAGE = "phase2.stage"
STATE_P2_STATUS = "phase2.status"
STATE_P2_FAILURE_REASON = "phase2.failure_reason"
STATE_P2_EVENTS = "phase2.events"

# ------------- URDF 分析域 -------------
STATE_P2_URDF_VALID = "phase2.urdf.valid"
STATE_P2_URDF_ISSUES = "phase2.urdf.issues"
STATE_P2_URDF_RISK = "phase2.urdf.risk"

# ------------- 配置域 -------------
STATE_P2_CONFIG_MODE = "phase2.config.mode"
STATE_P2_CONFIG_VERSION = "phase2.config.version"
STATE_P2_CONFIG_PARENT_VERSION = "phase2.config.parent_version"
STATE_P2_CONFIG_HISTORY = "phase2.config.history"
STATE_P2_CONFIG_TEMPLATE = "phase2.config.template_name"
STATE_P2_CONFIG_TEXT = "phase2.config.generated_text"

# ------------- 训练域 -------------
# 训练状态：running / early_stopped / completed。
STATE_P2_TRAIN_STATUS = "phase2.train.status"
STATE_P2_TRAIN_COMMAND = "phase2.train.command"
# 远端训练进程 PID。
STATE_P2_TRAIN_PID = "phase2.train.pid"
# 远端训练日志文件路径。
STATE_P2_TRAIN_LOG_PATH = "phase2.train.log_path"
# 全量指标历史（每次采样的 loss/reward 字典数组）。
STATE_P2_TRAIN_METRIC_HISTORY = "phase2.train.metric_history"
# 送入日志裁判的输入负载。
STATE_P2_TRAIN_LOG_INPUT = "phase2.train.log_input_payload"
# 日志裁判的裁决结果。
STATE_P2_TRAIN_LOG_JUDGE_RESULT = "phase2.train.log_judge_result"

# ------------- Play（视频渲染）域 -------------
STATE_P2_PLAY_STDOUT = "phase2.play.stdout"
STATE_P2_PLAY_STDERR = "phase2.play.stderr"
STATE_P2_PLAY_EXIT_CODE = "phase2.play.exit_code"
STATE_P2_PLAY_FAILED = "phase2.play.failed"

# ------------- 评估域 -------------
STATE_P2_EVAL_VIDEO_PATH = "phase2.eval.video_path"
STATE_P2_EVAL_VIDEO_REMOTE_PATH = "phase2.video.remote_path"
STATE_P2_EVAL_PASSED = "phase2.eval.passed"
STATE_P2_EVAL_SCORE = "phase2.eval.score_card"
STATE_P2_EVAL_FAIL_REASON = "phase2.eval.fail_reason"
STATE_P2_VIDEO_INPUT_PAYLOAD = "phase2.video.input_payload"
STATE_P2_VIDEO_JUDGE_RESULT = "phase2.video.judge_result"

# ------------- 迭代与 HITL 域 -------------
STATE_P2_ITER_ROUND = "phase2.iteration.current_round"
STATE_P2_ITER_MAX = "phase2.iteration.max_rounds"
STATE_P2_HITL_REQUIRED = "phase2.hitl.required"
STATE_P2_HITL_REASON = "phase2.hitl.reason"
STATE_P2_HITL_RESPONSE = "phase2.hitl.response"
STATE_P2_HITL_RESOLVED = "phase2.hitl.resolved"

# ------------- 归档域 -------------
STATE_P2_ARCHIVE_SUMMARY = "phase2.archive.summary"
STATE_P2_ARCHIVE_COMPLETED = "phase2.archive.completed"
