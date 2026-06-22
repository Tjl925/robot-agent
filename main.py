from __future__ import annotations

"""统一入口脚本
- 读取统一配置；
- 构建总编排器 `OrchestratorAgent`；
- 先执行 Phase-1，再执行 Phase-2；
- 最终输出完整 session state。
"""

import argparse
import asyncio
import json
import os
import re
import sys
import atexit
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# 匹配所有标准 ANSI/VT100 控制序列：CSI 参数序列（颜色/光标等）、OSC 序列、其他双字节 ESC 序列。
_ANSI_RE = re.compile(
    r"\x1b"
    r"(?:"
    r"\[[0-9;?]*[A-Za-z]"       # CSI 序列：\x1b[ ... 字母
    r"|\][^\x07\x1b]*(?:\x07|\x1b\\)"  # OSC 序列：\x1b] ... BEL 或 ST
    r"|[@-_]"                    # 双字节 ESC 序列：\x1b @-_
    r")"
)

def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


class Tee:
    """双向流分流器，支持将终端打印内容同步写入日志文件。

    终端输出保持原始 ANSI 颜色；写入日志文件时先剥去 ANSI 转义序列，保证日志可读。

    退出/GC 阶段写入容错：解释器关闭时残留对象的 finalizer 或后台线程仍可能调用 write/flush，
    若底层文件已关闭会抛 ValueError，再被 CPython 当作 "Exception ignored in" 打印 → 又走到这里
    的坏流 → 无限级联刷屏（服务器掉线后 main.py 卡死的直接症状）。故所有写入路径吞掉异常，
    关闭后置 closed 标志直接 no-op，彻底断开级联。
    """

    def __init__(self, file_handle, original_stream, is_owner: bool = False):
        self.file = file_handle
        self.original_stream = original_stream
        self.is_owner = is_owner
        self.closed = False

    def write(self, message):
        if self.closed:
            return
        try:
            self.original_stream.write(message)
        except Exception:
            pass
        try:
            self.file.write(_strip_ansi(message))
        except Exception:
            pass

    def flush(self):
        if self.closed:
            return
        try:
            self.original_stream.flush()
        except Exception:
            pass
        try:
            self.file.flush()
        except Exception:
            pass

    def close(self):
        self.closed = True
        if self.is_owner:
            try:
                self.file.close()
            except Exception:
                pass


def setup_console_logging():
    """初始化控制台终端日志同步保存到 logs 目录，采用单一文件描述符且防止多进程冲突。"""
    import os

    # 1. 检查环境变量以避免多进程/多模块导入时重复生成新的日志文件
    env_log_key = "ROBOT_AGENT_ACTIVE_LOG"
    if env_log_key in os.environ:
        return

    now_str = datetime.now().strftime("%m-%d_%H-%M")
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"{now_str}.log"

    # 将日志文件绝对路径写入环境变量，防止子进程或重复导入时再次创建
    os.environ[env_log_key] = str(log_file.resolve())

    # 在替换流之前，利用原生标准输出打印一行优雅提示给用户
    sys.stdout.write(f"【系统提示】终端打印内容已同步保存至本地日志: {log_file.as_posix()}\n\n")
    sys.stdout.flush()

    # 2. 共享同一个文件句柄以避免写冲突或重复打开清空文件
    log_file_handle = open(log_file, "w", encoding="utf-8", buffering=1)

    # 替换标准输出与错误输出流，仅让 stdout 做为 owner 负责最终句柄关闭
    sys.stdout = Tee(log_file_handle, sys.stdout, is_owner=True)
    sys.stderr = Tee(log_file_handle, sys.stderr, is_owner=False)

    def cleanup():
        if isinstance(sys.stdout, Tee):
            sys.stdout.close()

    atexit.register(cleanup)

load_dotenv()

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from robot_agent.agents import OrchestratorAgent
from robot_agent.schemas.config import AutoDLConfig, TailiCloudConfig, WindowsRemoteConfig
from robot_agent.schemas.state import (
    STATE_P1_EVENTS,
    STATE_P1_INSTANCE_UUID,
    STATE_P1_SSH_CONNECTED,
    STATE_P1_STAGE,
    STATE_P2_EVENTS,
    STATE_P2_HITL_REQUIRED,
    STATE_P2_ITER_MAX,
    STATE_P2_ITER_ROUND,
    STATE_P2_STAGE,
    STATE_P2_STATUS,
    STATE_P1_RETRY_COUNT,
    Phase1Stage,
    Phase2Stage,
)


def load_config(path: Path) -> tuple[AutoDLConfig, TailiCloudConfig, str, WindowsRemoteConfig]:
    import os
    raw = json.loads(path.read_text(encoding="utf-8"))

    # 优先从环境变量中读取 AutoDL Token，确保安全性
    phase1 = raw.get("phase1", {})
    env_token = os.getenv("AUTODL_TOKEN")
    if env_token:
        phase1["token"] = env_token

    auto_cfg = AutoDLConfig(**phase1)

    # 顶层开关：linux（AutoDL/备用服务器）或 windows（直连 Windows 主机）
    remote_platform = str(raw.get("remote_platform", "linux")).lower()
    windows_cfg = WindowsRemoteConfig(**raw.get("windows_remote", {}))
    # Windows 主机密码亦支持环境变量覆盖（与 AUTODL_TOKEN 一致的安全习惯）
    env_win_pw = os.getenv("WINDOWS_SSH_PASSWORD")
    if env_win_pw:
        windows_cfg.ssh_password = env_win_pw

    taili_cfg = TailiCloudConfig(**raw["phase2"])
    taili_cfg.remote_platform = "windows" if remote_platform == "windows" else "linux"
    return auto_cfg, taili_cfg, remote_platform, windows_cfg


async def run_all(auto_cfg: AutoDLConfig, taili_cfg: TailiCloudConfig, remote_platform: str, windows_cfg: WindowsRemoteConfig) -> dict:
    root_agent = OrchestratorAgent(auto_cfg=auto_cfg, taili_cfg=taili_cfg, remote_platform=remote_platform, windows_cfg=windows_cfg)
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=taili_cfg.app_name,
        user_id=taili_cfg.user_id,
        session_id=taili_cfg.session_id,
        state={
            STATE_P1_STAGE: Phase1Stage.INIT,
            STATE_P2_STAGE: Phase2Stage.INIT,
            STATE_P2_STATUS: "pending",
            STATE_P1_EVENTS: [],
            STATE_P2_EVENTS: [],
            STATE_P1_RETRY_COUNT: 0,
            STATE_P2_ITER_ROUND: 0,
            STATE_P2_ITER_MAX: taili_cfg.max_auto_iterations,
            STATE_P2_HITL_REQUIRED: False,
            STATE_P1_SSH_CONNECTED: False,
            STATE_P1_INSTANCE_UUID: auto_cfg.instance_uuid,
        },
    )

    runner = Runner(agent=root_agent, app_name=taili_cfg.app_name, session_service=session_service)
    kickoff = types.Content(role="user", parts=[types.Part(text="Run full phase1 + phase2 workflow")])

    async for event in runner.run_async(
        user_id=taili_cfg.user_id,
        session_id=taili_cfg.session_id,
        new_message=kickoff,
    ):
        if event.is_final_response() and event.content and event.content.parts:
            print(event.content.parts[0].text)

    final_session = await session_service.get_session(
        app_name=taili_cfg.app_name,
        user_id=taili_cfg.user_id,
        session_id=taili_cfg.session_id,
    )
    return dict(final_session.state if final_session else {})


def format_final_state(state: dict) -> dict:
    """对最终输出的状态进行过滤与精简，防止超大字段打爆终端，支持用户后续自由增删。"""
    omit_keys = {
        "phase2.train.metric_history",
        "phase2.train.log_input_payload",
        "phase2.config.generated_text",
        "phase2.video.input_payload",
        "phase2.play.stdout",
        "phase2.eval.score_card",
        "phase2.video.judge_summary",
    }
    
    formatted = {}
    for k, v in sorted(state.items()):
        if k in omit_keys:
            if isinstance(v, list):
                formatted[k] = f"<list of length {len(v)} omitted>"
            elif isinstance(v, dict):
                formatted[k] = f"<dict keys {list(v.keys())} omitted>"
            elif isinstance(v, str):
                formatted[k] = f"<str of length {len(v)} omitted>"
            else:
                formatted[k] = f"<{type(v).__name__} omitted>"
        else:
            formatted[k] = v
    return formatted


def main() -> None:
    setup_console_logging()
    parser = argparse.ArgumentParser(description="Unified Phase1 + Phase2 orchestrator")
    parser.add_argument("--config", required=True, help="Path to unified config json")
    args = parser.parse_args()

    exit_code = 0
    try:
        auto_cfg, taili_cfg, remote_platform, windows_cfg = load_config(Path(args.config))
        final_state = asyncio.run(run_all(auto_cfg, taili_cfg, remote_platform, windows_cfg))

        # 优雅过滤并格式化输出
        clean_state = format_final_state(final_state)
        print(json.dumps(clean_state, ensure_ascii=False, indent=2, default=str))
    except Exception:
        import traceback
        traceback.print_exc()
        exit_code = 1
    finally:
        # 硬保证退出：服务器掉线时 paramiko Transport 等非守护线程可能残留，普通 return 会让进程
        # 一直挂着（并随 Tee 刷屏 Exception ignored in）。先把日志刷盘，再用 os._exit 强制终止，
        # 绕过会被残留线程拖住的正常解释器关闭流程。
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        os._exit(exit_code)


if __name__ == "__main__":
    main()
