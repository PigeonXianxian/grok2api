"""xAI API 客户端 — 通过 api.x.ai 调用 Grok 模型

基于 CPA 逆向的 API 协议，使用 OAuth Bearer Token 认证。
支持：
  - 聊天 (SSE streaming via /responses)
  - 图片生成 (/images/generations)
  - 视频生成 (/videos/generations, /videos/extensions, /videos/edits)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, AsyncGenerator

import httpx
import orjson

from app.platform.config.snapshot import get_config
from app.platform.errors import UpstreamError
from app.platform.logging.logger import logger
from app.dataplane.reverse.protocol.xai_oauth import (
    TokenStorage,
    DEFAULT_API_BASE,
    refresh_access_token,
)

# ── 端点 ──────────────────────────────────────────────────────────────────

RESPONSES = "/responses"
IMAGES_GENERATIONS = "/images/generations"
IMAGES_EDITS = "/images/edits"
VIDEOS_GENERATIONS = "/videos/generations"
VIDEOS_EDITS = "/videos/edits"
VIDEOS_EXTENSIONS = "/videos/extensions"
VIDEOS = "/videos"  # GET /videos/{request_id}


# ── 请求模型 ──────────────────────────────────────────────────────────────


@dataclass(slots=True)
class XAIClient:
    """xAI API 客户端，封装认证和请求逻辑。"""

    token_storage: TokenStorage
    base_url: str = DEFAULT_API_BASE
    _http: httpx.AsyncClient | None = None

    def _build_headers(self, stream: bool = False) -> dict[str, str]:
        """构建请求 headers。"""
        return {
            "Authorization": f"Bearer {self.token_storage.access_token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
            "Connection": "Keep-Alive",
        }

    async def _ensure_token(self) -> None:
        """检查并刷新 token（如需要）。"""
        if self.token_storage.needs_refresh:
            logger.info("xai token needs refresh, refreshing...")
            token_data = await refresh_access_token(
                self.token_storage.refresh_token,
                self.token_storage.token_endpoint,
            )
            self.token_storage.access_token = token_data.access_token
            if token_data.refresh_token:
                self.token_storage.refresh_token = token_data.refresh_token
            self.token_storage.expire = token_data.expire
            self.token_storage.expires_in = token_data.expires_in
            self.token_storage.last_refresh = token_data.expire

    async def chat_stream(
        self,
        payload: dict[str, Any],
        *,
        timeout_s: float = 120.0,
        session_id: str = "",
    ) -> AsyncGenerator[str, None]:
        """POST /responses — SSE 流式聊天。"""
        await self._ensure_token()

        url = f"{self.base_url.rstrip('/')}{RESPONSES}"
        headers = self._build_headers(stream=True)
        if session_id:
            headers["x-grok-conv-id"] = session_id

        async with self._get_client() as client:
            try:
                async with client.stream(
                    "POST", url,
                    headers=headers,
                    content=orjson.dumps(payload),
                    timeout=timeout_s,
                ) as resp:
                    if resp.status_code != 200:
                        body = (await resp.aread()).decode("utf-8", "replace")[:500]
                        raise UpstreamError(
                            f"xAI chat returned {resp.status_code}",
                            status=resp.status_code,
                            body=body,
                        )
                    async for line in resp.aiter_lines():
                        yield line
            except UpstreamError:
                raise
            except Exception as exc:
                raise UpstreamError(f"xAI chat transport error: {exc}") from exc

    async def chat(
        self,
        payload: dict[str, Any],
        *,
        timeout_s: float = 120.0,
        session_id: str = "",
    ) -> dict[str, Any]:
        """POST /responses — 非流式聊天。"""
        await self._ensure_token()

        url = f"{self.base_url.rstrip('/')}{RESPONSES}"
        headers = self._build_headers(stream=False)
        if session_id:
            headers["x-grok-conv-id"] = session_id

        async with self._get_client() as client:
            try:
                resp = await client.post(
                    url,
                    headers=headers,
                    content=orjson.dumps(payload),
                    timeout=timeout_s,
                )
                if resp.status_code != 200:
                    body = resp.content.decode("utf-8", "replace")[:500]
                    raise UpstreamError(
                        f"xAI chat returned {resp.status_code}",
                        status=resp.status_code,
                        body=body,
                    )
                return resp.json()
            except UpstreamError:
                raise
            except Exception as exc:
                raise UpstreamError(f"xAI chat transport error: {exc}") from exc

    async def generate_images(
        self,
        payload: dict[str, Any],
        *,
        timeout_s: float = 300.0,
    ) -> dict[str, Any]:
        """POST /images/generations — 图片生成。"""
        await self._ensure_token()

        url = f"{self.base_url.rstrip('/')}{IMAGES_GENERATIONS}"
        headers = self._build_headers(stream=False)

        async with self._get_client() as client:
            try:
                resp = await client.post(
                    url,
                    headers=headers,
                    content=orjson.dumps(payload),
                    timeout=timeout_s,
                )
                if resp.status_code != 200:
                    body = resp.content.decode("utf-8", "replace")[:500]
                    raise UpstreamError(
                        f"xAI images returned {resp.status_code}",
                        status=resp.status_code,
                        body=body,
                    )
                return resp.json()
            except UpstreamError:
                raise
            except Exception as exc:
                raise UpstreamError(f"xAI images transport error: {exc}") from exc

    async def generate_video(
        self,
        payload: dict[str, Any],
        *,
        timeout_s: float = 600.0,
    ) -> dict[str, Any]:
        """POST /videos/generations — 视频生成。"""
        await self._ensure_token()

        url = f"{self.base_url.rstrip('/')}{VIDEOS_GENERATIONS}"
        headers = self._build_headers(stream=False)
        if key := payload.pop("idempotency_key", None):
            headers["x-idempotency-key"] = str(key)

        async with self._get_client() as client:
            try:
                resp = await client.post(
                    url,
                    headers=headers,
                    content=orjson.dumps(payload),
                    timeout=timeout_s,
                )
                if resp.status_code not in (200, 201, 202):
                    body = resp.content.decode("utf-8", "replace")[:500]
                    raise UpstreamError(
                        f"xAI video returned {resp.status_code}",
                        status=resp.status_code,
                        body=body,
                    )
                return resp.json()
            except UpstreamError:
                raise
            except Exception as exc:
                raise UpstreamError(f"xAI video transport error: {exc}") from exc

    async def get_video_status(
        self,
        request_id: str,
        *,
        timeout_s: float = 60.0,
    ) -> dict[str, Any]:
        """GET /videos/{request_id} — 查询视频状态。"""
        await self._ensure_token()

        url = f"{self.base_url.rstrip('/')}{VIDEOS}/{request_id}"
        headers = self._build_headers(stream=False)

        async with self._get_client() as client:
            try:
                resp = await client.get(url, headers=headers, timeout=timeout_s)
                if resp.status_code != 200:
                    body = resp.content.decode("utf-8", "replace")[:500]
                    raise UpstreamError(
                        f"xAI video status returned {resp.status_code}",
                        status=resp.status_code,
                        body=body,
                    )
                return resp.json()
            except UpstreamError:
                raise
            except Exception as exc:
                raise UpstreamError(f"xAI video status error: {exc}") from exc

    def _get_client(self) -> httpx.AsyncClient:
        if self._http is None:
            cfg = get_config()
            proxy_url = cfg.get_str("proxy.egress.proxy_url", "")
            verify = not cfg.get_bool("proxy.egress.skip_ssl_verify", False)
            kwargs: dict[str, Any] = {"timeout": httpx.Timeout(120.0), "verify": verify}
            if proxy_url:
                kwargs["proxy"] = proxy_url
            self._http = httpx.AsyncClient(**kwargs)
        return self._http

    async def close(self) -> None:
        if self._http:
            await self._http.aclose()
            self._http = None


__all__ = [
    "XAIClient",
    "RESPONSES", "IMAGES_GENERATIONS", "IMAGES_EDITS",
    "VIDEOS_GENERATIONS", "VIDEOS_EDITS", "VIDEOS_EXTENSIONS", "VIDEOS",
]
