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

from pydantic import BaseModel, Field, model_validator


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

    # 备用机器 SSH 主机地址。
    backup_ssh_host: str = "183.147.142.40"
    # 备用机器 SSH 端口。
    backup_ssh_port: int = 30113
    # 备用机器 SSH 用户名。
    backup_ssh_user: str = "root"
    # 备用机器 SSH 密码。
    backup_ssh_password: str = "zxy5ts34"

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


class WindowsRemoteConfig(BaseModel):
    """Windows 主机直连配置（remote_platform=windows 时启用）。

    与 Linux 链路（AutoDL 开机 -> 备用服务器）完全平行的另一条链路：
    - 不走 AutoDL 开机，直接 SSH 连这台带 3060 显卡的 Windows 主机；
    - 远端 shell 为 PowerShell 7（pwsh），命令执行层由 taili_cloud_windows 提供；
    - 路径与训练/评估命令模板默认从 robot_lab_root + conda_env 自动派生，
      只需填 ssh 连接信息和 robot_lab 根目录即可；需要时也可在 json 里显式覆盖。

    注意：路径统一用正斜杠（pwsh 与 OpenSSH SFTP 均接受），便于 posixpath 复用。
    """

    # —— SSH 连接信息 ——
    ssh_host: str = Field(default="100.81.254.37", description="Windows 主机 IP")
    ssh_port: int = Field(default=22, ge=1, le=65535, description="Windows 主机 SSH 端口")
    ssh_user: str = Field(default="xbtl", description="Windows 主机 SSH 用户名")
    ssh_password: str = Field(default="", description="Windows 主机 SSH 密码（可用环境变量 WINDOWS_SSH_PASSWORD 覆盖）")

    # —— robot_lab 根目录与 conda 环境（其余路径/命令默认由此派生）——
    robot_lab_root: str = Field(default="e:/tjl/robot_lab", description="Windows 主机上的 robot_lab 根目录（正斜杠）")
    conda_env: str = Field(default="tjl", description="训练所用 conda 环境名")

    # —— 以下为派生项：留空则由 robot_lab_root 自动补全；显式填写则原样使用 ——
    tmp_dir: str = Field(default="", description="远端临时目录，默认 <root>/tmp")
    asset_path: str = Field(default="", description="资产文件落点，默认 <root>/source/.../assets/taili_quad.py")
    task_cfg_root: str = Field(default="", description="任务目录落点，默认 <root>/source/.../quadruped/taili_quad")
    eval_log_root: str = Field(default="", description="训练日志根目录，默认 <root>/logs/rsl_rl")
    train_command_template: str = Field(default="", description="Windows 训练命令模板（pwsh），支持变量 task_name")
    play_eval_command_template: str = Field(default="", description="Windows 评估命令模板（pwsh），支持变量 task_name/eval_terrain")

    @model_validator(mode="after")
    def _derive_defaults(self) -> "WindowsRemoteConfig":
        root = str(self.robot_lab_root).replace("\\", "/").rstrip("/")
        self.robot_lab_root = root
        if not self.tmp_dir:
            self.tmp_dir = f"{root}/tmp"
        if not self.asset_path:
            self.asset_path = f"{root}/source/robot_lab/robot_lab/assets/taili_quad.py"
        if not self.task_cfg_root:
            self.task_cfg_root = f"{root}/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/taili_quad"
        if not self.eval_log_root:
            self.eval_log_root = f"{root}/logs/rsl_rl"
        # pwsh 训练/评估命令：先激活 conda 环境，再 cd 到 root 跑脚本。
        # 默认依赖 conda 已对 pwsh 初始化（conda init powershell）；若非交互会话找不到 conda，
        # 可在 json 里把模板改成带完整 conda 路径或 `& conda shell.powershell hook | iex` 前缀。
        if not self.train_command_template:
            self.train_command_template = (
                f"conda activate {self.conda_env}; cd '{root}'; "
                f"python scripts/reinforcement_learning/rsl_rl/train.py --task={{task_name}} --headless"
            )
        if not self.play_eval_command_template:
            self.play_eval_command_template = (
                f"conda activate {self.conda_env}; cd '{root}'; "
                f"python scripts/reinforcement_learning/rsl_rl/play_eval.py "
                f"--task={{task_name}} --headless --video --num_envs=1 --eval_terrain={{eval_terrain}}"
            )
        return self


class TailiCloudConfig(BaseModel):
    """Taili 专用云端接入配置。

    这里所有字段都围绕同一条固定链路：
    本地 `taili_quad/` -> 云端 `robot_lab/`。
    """

    # 远端平台：linux=沿用 AutoDL/备用服务器(bash/tmux)，windows=直连 Windows 主机(pwsh)。
    # 决定 taili_steps 选用哪套远端命令执行后端（taili_cloud / taili_cloud_windows）。
    remote_platform: Literal["linux", "windows"] = Field(default="linux", description="远端平台开关：linux / windows")

    # 本地机械狗模型根目录。
    local_robot_root: str = Field(default="taili_quad", description="本地机械狗模型根目录")
    # 本地机器人 URDF 所在子目录。
    local_robots_subdir: str = Field(default="urdf", description="本地机器人 URDF 子目录")
    # 云端 robot_lab 根目录（固定路径）。
    cloud_robot_lab_root: str = Field(default="/root/autodl-tmp/robot_lab", description="云端 robot_lab 根目录（固定路径）")
    # 云端临时目录落点（用于存放临时脚本、日志和状态退出码）。
    cloud_tmp_dir: str = Field(default="/root/autodl-tmp/robot_lab/tmp", description="云端临时目录落点（用于存放临时脚本、日志和状态退出码）")
    # 云端资产文件固定落点。
    cloud_asset_path: str = Field(default="/root/autodl-tmp/robot_lab/source/robot_lab/robot_lab/assets/taili_quad.py", description="云端资产文件固定落点")
    # 云端任务目录固定落点。
    cloud_task_cfg_root: str = Field(default="/root/autodl-tmp/robot_lab/source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/taili_quad", description="云端任务目录固定落点")

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
    task_name: str = Field(default="RobotLab-Isaac-Velocity-Rough-Taili-Quad-v0", description="云端训练任务名（固定风格）")
    # 云端训练命令模板。
    train_command_template: str = Field(default="cd /root/autodl-tmp/robot_lab && python scripts/reinforcement_learning/rsl_rl/train.py --task={task_name} --headless", description="云端训练命令模板，支持变量: task_name")
    # 评估时使用的地形列表
    play_eval_terrains: list[str] = Field(default=["flat", "boxes", "stairs_down", "stairs_up"], description="评估时使用的地形列表")
    # play_eval.py 整体渲染的超时时间（秒）
    play_eval_timeout_seconds: int = Field(default=900, ge=30, description="play_eval.py 渲染视频的超时时间（秒）")
    # 多地形评估命令模板
    play_eval_command_template: str = Field(
        default=(
            "cd /root/autodl-tmp/robot_lab && "
            "python scripts/reinforcement_learning/rsl_rl/play_eval.py "
            "--task={task_name} --headless --video --num_envs=1 "
            "--eval_terrain={eval_terrain}"
        ),
        description="多地形评估命令模板，支持变量: task_name, eval_terrain"
    )
    # 远端训练日志根目录。
    eval_log_root: str = Field(default="/root/robot_lab/logs/rsl_rl", description="云端训练日志根目录")
    # 中间检查间隔（自动推导或覆盖值）。
    eval_check_interval: int = Field(default=100, ge=1, description="训练中日志检查间隔（iterations）")
    # 每次轮询采样的 checkpoint 窗口大小。取本轮新增 blocks 的最后 N 个组成一个采样窗口。
    eval_sample_window_size: int = Field(default=5, ge=1, description="每次轮询采样的 checkpoint 数量")

    # 自动迭代上限。
    max_auto_iterations: int = Field(default=2, ge=0, description="评估失败后允许的自动迭代轮数上限（达到后触发人工介入）。")

    # 单轮训练最长分钟数。
    max_training_minutes: int = Field(default=240, ge=1, description="单轮训练允许的最长分钟数")
    # 是否在 revise 轮从历史最优 checkpoint 续训（warm-start），让迭代真正累积技能而非每轮从零重训。
    resume_from_best: bool = Field(default=True, description="revise 轮是否从历史最优 checkpoint 续训（warm-start）。True=续训，False=每轮从零训练。")
    # max_iterations 时间预算可达性截断：用上一轮实测每步耗时估算本轮在 max_training_minutes 内能跑到的步数，
    # 若 LLM 设的 max_iterations 超出可达范围，则用原生 --max_iterations 截断，让训练自然 completed 而非被墙钟杀死浪费一轮。
    iter_budget_cap_enabled: bool = Field(default=True, description="是否按时间预算截断 max_iterations，避免墙钟超时浪费整轮训练。")
    iter_budget_safety_ratio: float = Field(default=0.85, gt=0.0, le=1.0, description="可达步数安全系数（预留启动/落盘/视频渲染等开销）。")

    # ===== 视频评估·数值硬闸（基于 play_eval 写入 eval_meta 的 achieved_metrics）=====
    # 总开关：开启后，数值不达标可否决视频裁判的"通过"（只会让通过更难，绝不把失败救成通过）。
    metric_gate_enabled: bool = Field(default=True, description="是否启用 achieved_metrics 数值硬闸（可否决 VLM 误判通过）。")
    # 机器人站立髋高（米），用于把步幅归一化为 stride/hip_height 做跨尺寸判断。
    robot_hip_height: float = Field(default=0.53, ge=0.05, description="机器人名义站立髋高（米），用于步幅归一化。")
    # —— 任务完成类（与机器人尺寸无关，较硬）——
    gate_min_fwd_speed_ratio: float = Field(default=0.4, description="前进段实速/指令速 低于此值判失败（几乎没按指令前进）。")
    gate_max_resets: int = Field(default=0, ge=0, description="评估时长内允许的 reset 次数上限，超过判失败（基本=摔倒/出界）。")
    # —— 步态质量类（物理先验默认，provisional，建议据真机数据微调）——
    gate_max_contact_freq_hz: float = Field(default=3.0, gt=0, description="motion 段每足触地频率上限（Hz），超过判为碎步/高频踏步。")
    gate_min_swing_time_s: float = Field(default=0.13, gt=0, description="平均摆动时长下限（秒），过短判为碎步。")
    gate_min_stride_norm: float = Field(default=0.25, gt=0, description="归一化步幅(步幅/髋高)下限，过短判为碎步/拖步。")
    gate_max_bounce_ratio: float = Field(default=0.45, gt=0, description="motion 段 |vz|/前进速 上限，过大判为上下颠簸/跳。")
    # —— 碎步以外的失败模式检测（metrics-only：play_eval 已实测并喂 VLM/修参，但阈值 None=暂不接硬闸）——
    # 这些默认 None=仅观测不否决：阈值是四足运动学先验，未在本管线标定，直接当硬闸会误杀第一个好策略。
    # 标定流程：先让 play_eval 写出指标，用一个"已知良好"策略(或 B2 参考)跑一遍读真实分布，把阈值设到正常值的约 0.4x 再填数开闸。
    gate_min_swing_clearance_m: float | None = Field(default=None, description="摆动相足端最矮离地高度下限(米)，过低=拖地。None=不接硬闸仅观测。台阶地形 baseline 跨台阶跳变，启用时应在 overrides 里对 stairs 置 None 禁用。")
    gate_max_foot_slip_ratio: float | None = Field(default=None, description="支撑相足端对地水平速度>0.1m/s 的占比上限，过高=触地打滑/拖滑。None=不接硬闸仅观测。")
    gate_max_foot_touchdown_cv: float | None = Field(default=None, description="四足 motion 段触地次数变异系数上限，过大=跛行/瘸腿。None=不接硬闸仅观测。")
    gate_max_diag_pair_diff: float | None = Field(default=None, description="两对角线触地总数归一化差上限，过大=对角失衡(跛行指纹)。None=不接硬闸仅观测。")
    gate_max_p95_touchdown_grf_bw: float | None = Field(default=None, description="落地法向冲击力 p95(体重倍数)上限，过大=砸地/硬着陆。None=不接硬闸仅观测。")
    # 按地形覆盖部分阈值（缺省项继承上面的全局 gate_*）。键=地形名；值={短阈值名: 数值}，短名=gate_* 去掉前缀。
    # 例：楼梯天生步幅短、步频高、颠簸大，应放宽；平地不在表里则用全局（最严）。这些是 provisional 骨架值，请据真机分地形数据微调。
    # 值可为数值(覆盖阈值)或 None(在该地形禁用该规则，如 clearance 在台阶上 baseline 失真应禁用)。
    gate_terrain_overrides: dict[str, dict[str, float | None]] = Field(
        default_factory=lambda: {
            "boxes": {"max_contact_freq_hz": 3.6, "min_stride_norm": 0.18, "max_bounce_ratio": 0.55},
            "stairs_down": {"max_contact_freq_hz": 4.0, "min_swing_time_s": 0.10, "min_stride_norm": 0.13, "max_bounce_ratio": 0.60, "min_fwd_speed_ratio": 0.30},
            "stairs_up": {"max_contact_freq_hz": 4.0, "min_swing_time_s": 0.10, "min_stride_norm": 0.13, "max_bounce_ratio": 0.65, "min_fwd_speed_ratio": 0.30},
        },
        description="按地形覆盖硬闸阈值；缺省继承全局 gate_*；值为 None=在该地形禁用该规则。可覆盖键：min_fwd_speed_ratio/max_resets/max_contact_freq_hz/min_swing_time_s/min_stride_norm/max_bounce_ratio/min_swing_clearance_m/max_foot_slip_ratio/max_foot_touchdown_cv/max_diag_pair_diff/max_p95_touchdown_grf_bw。",
    )

    # ===== 训练级"最终验收"指标（仅供验收/参考，绝不参与训练早停决策）=====
    accept_metrics_enabled: bool = Field(default=True, description="训练完成后是否从训练日志抓取验收指标并对照目标（仅参考，不影响早停）。")
    accept_min_terrain_levels: float = Field(default=6.0, description="验收目标：Curriculum/terrain_levels 期望 >= 此值（已爬到高难度地形）。")
    accept_max_error_vel_xy: float = Field(default=0.4, description="验收目标：Metrics/base_velocity/error_vel_xy 期望 <= 此值。")
    accept_max_error_vel_yaw: float = Field(default=0.5, description="验收目标：Metrics/base_velocity/error_vel_yaw 期望 <= 此值。")
    # 人工介入文本。
    hitl_response_text: str | None = Field(default=None, description="人工介入文本。若不为空，则在触发 WAIT_HUMAN 后自动写入并继续执行。")

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
    asset_code: str | None = None
    # agents/__init__.py
    agents_init_code: str | None = None
    # agents/rsl_rl_ppo_cfg.py
    agents_ppo_cfg_code: str | None = None
    # 任务注册代码草案 (__init__.py)
    task_init_code: str | None = None
    # flat_env_cfg.py
    flat_env_cfg_code: str | None = None
    # rough_env_cfg.py
    rough_env_cfg_code: str | None = None




class TailiTrainingLogJudgeResult(BaseModel):
    """训练日志评估 Agent 的结构化输出。"""

    # 动作：继续 / 判定失败早停 / 判定收敛早停。
    action: Literal["continue", "stop_failed", "stop_converged"]
    # 评估依据的分数或证据摘要。
    score: dict
    # 判定理由与趋势分析。
    reason: str


class TailiVideoJudgeResult(BaseModel):
    """视频评估 Agent 的结构化输出。"""

    # 是否通过验收。
    passed: bool
    # 评估得分卡 / 证据卡。
    score: dict
    # 验收结论与综合评价（无论通过与否都要给出）。
    reason: str
