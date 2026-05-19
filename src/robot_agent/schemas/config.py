from __future__ import annotations

"""机器人 Agent 系统配置模块。

这个模块里放的是整个项目最核心的三类数据：
1. Phase-1 的 AutoDL 启动与 SSH 探活配置；
2. Phase-2 的 Taili 专用云端接入配置；
3. LLM Agent 的输入/输出契约。

设计目标很明确：
- 尽量把“路径、版本、历史、证据”这些关键信息都类型化；
- 让配置对象既能给代码用，也能给 LLM Agent 作为结构化上下文；
- 让后续做 create / revise 时有明确的状态来源，而不是散落在各个函数里。
"""

from typing import Literal

from pydantic import BaseModel, Field


class AutoDLConfig(BaseModel):
    """Phase-1：AutoDL 开机与 SSH 探活配置。

    这一组字段只负责“把远端实例拉起来并连上”，
    不关心机器人训练细节。
    """

    # AutoDL API 根地址。
    api_base: str = "https://www.autodl.art"
    # 访问 AutoDL 平台所需的 Token。
    token: str
    # 目标 AutoDL 实例 UUID。
    instance_uuid: str

    # 开机时使用的载荷类型，当前默认是 GPU 实例。
    power_on_payload: str = "gpu"
    # 轮询实例状态的时间间隔（秒）。
    poll_interval_seconds: int = Field(default=8, ge=1, description="轮询实例状态的间隔（秒）。")
    # 等待实例进入 running 状态的最长时间（秒）。
    boot_timeout_seconds: int = Field(default=420, ge=10, description="等待实例进入 running 的最长时间（秒）。")

    # SSH 探活命令的超时时间（秒）。
    ssh_timeout_seconds: int = Field(default=20, ge=1, description="SSH 探活连接超时时间（秒）。")
    # 用来验证 SSH 连通性的测试命令。
    ssh_test_command: str = "echo connected && hostname"
    # 是否严格检查 SSH HostKey。
    strict_host_key_check: bool = False

    # 单个步骤允许的最大重试次数。
    max_retries_per_step: int = Field(default=2, ge=0, description="每个步骤的最大自动重试次数。")
    # 每次重试之间的退避时间（秒）。
    retry_backoff_seconds: int = Field(default=3, ge=0, description="重试退避时长（秒）。")

    # Phase-1 的应用名，用于会话标识。
    app_name: str = "agents"
    # Phase-1 的用户标识。
    user_id: str = "local_user"
    # Phase-1 的会话标识。
    session_id: str = "agent_session"


class TailiCloudConfig(BaseModel):
    """Taili 专用云端接入配置。

    这里所有字段都围绕同一条固定链路：
    本地 `taili_quad/` -> 云端 `robot_lab/`。
    """

    # 本地机械狗模型根目录。
    local_robot_root: str = Field(default="taili_quad", description="本地机械狗模型根目录")
    # 本地机器人 URDF 所在子目录。
    local_robots_subdir: str = Field(default="urdf", description="本地机器人 URDF 子目录")
    # 云端 robot_lab 根目录（固定路径）。
    cloud_robot_lab_root: str = Field(default="/root/robot_lab", description="云端 robot_lab 根目录（固定路径）")
    # 云端资产文件固定落点。
    cloud_asset_path: str = Field(default="/root/robot_lab/source/robot_lab/robot_lab/assets/taili_quad.py", description="云端资产文件固定落点")
    # 云端任务目录固定落点。
    cloud_task_cfg_root: str = Field(default="/root/robot_lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/taili_quad", description="云端任务目录固定落点")

    # 远端命令超时时间（秒）。
    remote_timeout_seconds: int = Field(default=60, ge=1, description="云端命令超时时间（秒）")
    # 远端 SSH 主机（由 Phase-1 handoff 后写入）。
    remote_host: str | None = Field(default=None, description="由 Phase-1 提供的云端 SSH 主机")
    # 远端 SSH 端口（由 Phase-1 handoff 后写入）。
    remote_port: int | None = Field(default=None, ge=1, le=65535, description="由 Phase-1 提供的云端 SSH 端口")
    # 远端 SSH 用户名（由 Phase-1 handoff 后写入）。
    remote_user: str | None = Field(default=None, description="由 Phase-1 提供的云端 SSH 用户名")
    # 远端 SSH 密码（由 Phase-1 handoff 后写入）。
    remote_password: str | None = Field(default=None, description="由 Phase-1 提供的云端 SSH 密码")

    # 云端训练任务名。
    task_name: str = Field(default="RobotLab-Isaac-Velocity-Flat-Taili-Quad-v0", description="云端训练任务名（固定风格）")
    # 云端训练命令模板。
    train_command_template: str = Field(default="cd /root/autodl-tmp/robot_lab && python scripts/reinforcement_learning/rsl_rl/train.py --task={task_name} --headless", description="云端训练命令模板，支持变量: task_name")
    # 云端播放 / 验证命令模板。
    play_command_template: str = Field(default="cd /root/autodl-tmp/robot_lab && python scripts/reinforcement_learning/rsl_rl/play.py --task={task_name} --headless --video", description="云端播放/验证命令模板，支持变量: task_name")
    # 播放视频时的并行环境数，默认保持脚本默认值（通常是 64）。
    play_num_envs: int | None = Field(default=None, ge=1, description="播放视频时的并行环境数；None 表示不额外传参")
    # play.py 渲染视频的超时时间（秒）。渲染通常需要 2~5 分钟，不能用默认的 60s。
    play_timeout_seconds: int = Field(default=600, ge=30, description="play.py 渲染视频的超时时间（秒）")
    # 远端训练日志根目录。
    eval_log_root: str = Field(default="/root/robot_lab/logs/rsl_rl", description="云端训练日志根目录")
    # 总训练迭代次数，动态生成。
    max_training_iterations: int = Field(default=1000, ge=1, description="单轮训练总迭代次数，由配置生成阶段给出")
    # 中间检查间隔（自动推导或覆盖值）。
    eval_check_interval: int = Field(default=100, ge=1, description="训练中日志检查间隔（iterations）")
    # 每次轮询采样的 checkpoint 窗口大小。取本轮新增 blocks 的最后 N 个组成一个采样窗口。
    eval_sample_window_size: int = Field(default=5, ge=1, description="每次轮询采样的 checkpoint 数量")


    # 自动迭代上限。
    max_auto_iterations: int = Field(default=2, ge=0, description="评估失败后允许的自动迭代轮数上限（达到后触发人工介入）。")

    # 单轮训练最长分钟数。
    max_training_minutes: int = Field(default=240, ge=1, description="单轮训练允许的最长分钟数")
    # 人工介入文本。
    hitl_response_text: str | None = Field(default=None, description="人工介入文本。若不为空，则在触发 WAIT_HUMAN 后自动写入并继续执行。")

    # 训练日志中评估 Agent 必须认识的核心指标说明。
    eval_metric_specs: list[dict] = Field(default_factory=list, description="评估 Agent 使用的指标说明列表")
    # 训练日志中评估 Agent 用来早停的辅助规则摘要。
    eval_early_stop_rules: list[dict] = Field(default_factory=list, description="训练中 early stop 判定规则列表")

    # 应用名。
    app_name: str = "agents"
    # 用户标识。
    user_id: str = "local_user"
    # 会话标识。
    session_id: str = "agent_session"


class TailiUrdfAnalysisResult(BaseModel):
    """URDF 诊断结果。

    这个模型用于把 URDF 分析 Agent 的输出固定成结构化 JSON，
    这样后续的配置生成和评估都能稳定消费。
    """

    # URDF 是否可用于后续训练。
    valid: bool
    # 风险等级：low / medium / high。
    risk: str
    # 具体问题列表，必须使用中文。
    issues: list[str] = Field(default_factory=list)


class TailiConfigContext(BaseModel):
    """Taili 配置生成 Agent 的输入上下文。

    这个对象不是最终配置，而是把 create / revise 所需要的
    所有上下文一次性装好，避免 LLM 需要自己猜历史状态。
    """

    # 当前模式：create 或 revise。
    mode: str = Field(default="create", description="create / revise")
    # 当前版本号。
    version: int
    # 父版本号。
    parent_version: int | None = None
    # 当前任务目标。
    task_goal: str | None = None
    # 当前 URDF 文件路径。
    urdf_path: str | None = None
    # 当前 URDF 原文。
    urdf_text: str | None = None
    # 当前 URDF 风险等级。
    urdf_risk: str | None = None
    # 当前 URDF 问题列表。
    urdf_issues: list[str] = Field(default_factory=list)
    # 当前已经生成过的草案（用于 revise）。
    current_draft: dict | None = None
    # 历史版本列表。
    history: list[dict] = Field(default_factory=list)
    # 最近失败原因列表。
    failure_reasons: list[str] = Field(default_factory=list)
    # 失败摘要文本。
    failure_summary: str | None = None
    # 当前迭代轮数。
    iteration_round: int = 0
    # 允许的最大迭代轮数。
    max_iterations: int = 0
    # 参考模板文本字典。
    reference_templates: dict[str, str] = Field(default_factory=dict)


class TailiConfigDraft(BaseModel):
    """Taili 配置生成 Agent 的结构化输出。

    这就是后续真正要落盘、同步到云端、参与训练的配置草案。
    """

    # 生成模式：create 或 revise。
    mode: str = Field(default="create", description="create / revise")
    # 当前版本号。
    version: int = Field(default=1, description="当前版本号")
    # 父版本号。
    parent_version: int | None = Field(default=None, description="父版本号")
    # 任务名。
    task_name: str
    # 修改原因与思考过程。
    reasoning: str = Field(description="修改原因与思考过程")
    # 资产代码草案 (asset_code)
    asset_code: str
    # agents/__init__.py
    agents_init_code: str
    # agents/rsl_rl_ppo_cfg.py
    agents_ppo_cfg_code: str
    # 任务注册代码草案 (__init__.py)
    task_init_code: str
    # flat_env_cfg.py
    flat_env_cfg_code: str
    # rough_env_cfg.py
    rough_env_cfg_code: str




class TailiTrainingLogJudgeResult(BaseModel):
    """训练日志评估 Agent 的结构化输出。"""

    # 动作：继续 / 判定失败早停 / 判定收敛早停。
    action: Literal["continue", "stop_failed", "stop_converged"]
    # 评估依据的分数或证据。
    score: dict
    # 原因列表。
    reasons: list[str] = Field(default_factory=list)
    # 置信度。
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class TailiVideoJudgeResult(BaseModel):
    """视频评估 Agent 的结构化输出。"""

    # 是否通过。
    passed: bool
    # 评估得分卡 / 证据卡。
    score: dict
    # 不通过原因。
    reasons: list[str] = Field(default_factory=list)
    # 下一步动作。
    next_action: str = Field(default="revise")
    # 证据模式。
    evidence_mode: str = Field(default="play_video")
