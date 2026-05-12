from __future__ import annotations

"""AutoDL API 工具层封装。

职责：
- 统一管理 HTTP 请求细节（鉴权、超时、成功码校验）；
- 提供 Phase-1 所需最小接口：开机、查状态、查快照、等待 running；
- 对外抛出明确异常类型，便于 orchestrator 执行重试策略。

说明：
- 本文件是“工具层”，不包含 ADK 语义；
- 后续 phase 可复用同一个客户端继续扩展关机/释放/实例列表等接口。
"""

import time
from typing import Any, Dict

import httpx


class AutoDLApiError(RuntimeError):
    """AutoDL 返回业务失败时抛出的异常。"""


class AutoDLClient:
    """AutoDL Pro 实例 API 客户端（Phase-1 精简版）。"""

    def __init__(self, api_base: str, token: str, timeout: float = 30.0) -> None:
        """初始化 API 客户端。

        参数：
        - `api_base`: API 服务器基础地址，默认 `https://www.autodl.art`
        - `token`: 开发者 Token（放在 Authorization header）
        - `timeout`: 每个 HTTP 请求超时
        """

        self.api_base = api_base.rstrip("/")
        self.headers = {
            "Authorization": token,
            "Content-Type": "application/json",
        }
        self.timeout = timeout

    def _request(self, method: str, path: str, json_body: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """发送单次请求，并校验 AutoDL 返回码。

        约定：
        - HTTP 成功后，还需要检查业务字段 `code == "Success"`；
        - 若失败，统一抛 `AutoDLApiError`，便于上层统一处理。
        """

        url = f"{self.api_base}{path}"
        with httpx.Client(timeout=self.timeout) as client:
            if method.upper() == "GET":
                # 文档示例对 GET 有 body 写法，但很多网关更兼容 query 参数方式。
                resp = client.request(method, url, headers=self.headers, params=json_body)
            else:
                resp = client.request(method, url, headers=self.headers, json=json_body)

        # HTTP 层错误（如 4xx/5xx）
        resp.raise_for_status()

        # 业务层错误（code != Success）
        data = resp.json()
        if data.get("code") != "Success":
            raise AutoDLApiError(f"AutoDL API failed: {data.get('code')} - {data.get('msg')}")
        return data

    def power_on_instance(self, instance_uuid: str, payload: str = "gpu") -> None:
        """调用开机接口。

        对应官方接口：
        POST /api/v1/adl_dev/dev/instance/pro/power_on
        """

        body = {
            "instance_uuid": instance_uuid,
            "payload": payload,
        }
        self._request("POST", "/api/v1/adl_dev/dev/instance/pro/power_on", json_body=body)

    def get_instance_status(self, instance_uuid: str) -> str:
        """获取实例当前状态。

        对应官方接口：
        GET /api/v1/adl_dev/dev/instance/pro/status
        """

        body = {"instance_uuid": instance_uuid}
        data = self._request("GET", "/api/v1/adl_dev/dev/instance/pro/status", json_body=body)
        return str(data.get("data", ""))

    def get_instance_snapshot(self, instance_uuid: str) -> Dict[str, Any]:
        """获取实例快照详情（含 SSH 信息字段）。

        对应官方接口：
        GET /api/v1/adl_dev/dev/instance/pro/snapshot
        """

        body = {"instance_uuid": instance_uuid}
        data = self._request("GET", "/api/v1/adl_dev/dev/instance/pro/snapshot", json_body=body)
        return data.get("data", {})

    def wait_until_running(self, instance_uuid: str, timeout_seconds: int, poll_interval_seconds: int) -> str:
        """轮询等待实例达到 `running`。

        参数：
        - `timeout_seconds`: 最长等待时间；
        - `poll_interval_seconds`: 轮询间隔。

        超时行为：
        - 超时则抛 `TimeoutError`，交由上层 orchestrator 决定是否重试。
        """

        started_at = time.time()
        while True:
            status = self.get_instance_status(instance_uuid)
            if status == "running":
                return status
            if time.time() - started_at > timeout_seconds:
                raise TimeoutError(f"Instance {instance_uuid} not running after {timeout_seconds}s")
            time.sleep(poll_interval_seconds)
