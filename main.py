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
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from robot_agent.agents import OrchestratorAgent
from robot_agent.schemas.config import AutoDLConfig, TailiCloudConfig
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


def load_config(path: Path) -> tuple[AutoDLConfig, TailiCloudConfig]:
    import os
    raw = json.loads(path.read_text(encoding="utf-8"))
    
    # 优先从环境变量中读取 AutoDL Token，确保安全性
    phase1 = raw.get("phase1", {})
    env_token = os.getenv("AUTODL_TOKEN")
    if env_token:
        phase1["token"] = env_token
        
    auto_cfg = AutoDLConfig(**phase1)
    taili_cfg = TailiCloudConfig(**raw["phase2"])
    return auto_cfg, taili_cfg


async def run_all(auto_cfg: AutoDLConfig, taili_cfg: TailiCloudConfig) -> dict:
    root_agent = OrchestratorAgent(auto_cfg=auto_cfg, taili_cfg=taili_cfg)
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
        "phase2.config.generated_text",
        "phase2.video.input_payload",
        "phase2.video.judge_result",
        "phase2.play.stdout",
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
    parser = argparse.ArgumentParser(description="Unified Phase1 + Phase2 orchestrator")
    parser.add_argument("--config", required=True, help="Path to unified config json")
    args = parser.parse_args()

    auto_cfg, taili_cfg = load_config(Path(args.config))
    final_state = asyncio.run(run_all(auto_cfg, taili_cfg))
    
    # 优雅过滤并格式化输出
    clean_state = format_final_state(final_state)
    print(json.dumps(clean_state, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
