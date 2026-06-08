"""xAI OAuth 2.0 + PKCE 认证模块

基于 CPA (CLIProxyAPI) 逆向的 Grok CLI OAuth 协议：
  - OIDC 发现: https://auth.x.ai/.well-known/openid-configuration
  - Client ID: b1a00492-073a-47ea-816f-4c329264a828
  - Scope: openid profile email offline_access grok-cli:access api:access
  - 回调: http://127.0.0.1:56121/callback

Token 存储、自动刷新、过期管理。
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx

from app.platform.logging.logger import logger

# ── OAuth 常量 ────────────────────────────────────────────────────────────

ISSUER = "https://auth.x.ai"
DISCOVERY_URL = f"{ISSUER}/.well-known/openid-configuration"
CLIENT_ID = "b1a00492-073a-47ea-816f-4c329264a828"
SCOPE = "openid profile email offline_access grok-cli:access api:access"
REDIRECT_HOST = "127.0.0.1"
CALLBACK_PORT = 56121
REDIRECT_PATH = "/callback"
REDIRECT_URI = f"http://{REDIRECT_HOST}:{CALLBACK_PORT}{REDIRECT_PATH}"
DEFAULT_API_BASE = "https://api.x.ai/v1"
REFRESH_LEAD_SECONDS = 300  # 5 分钟提前量

# ── 数据模型 ──────────────────────────────────────────────────────────────


@dataclass(slots=True)
class PKCECodes:
    """PKCE verifier / challenge 对。"""
    code_verifier: str
    code_challenge: str


@dataclass(slots=True)
class Discovery:
    """OIDC 发现返回的端点。"""
    authorization_endpoint: str
    token_endpoint: str


@dataclass(slots=True)
class TokenData:
    """OAuth token 数据。"""
    access_token: str
    refresh_token: str = ""
    id_token: str = ""
    token_type: str = "Bearer"
    expires_in: int = 0
    expire: str = ""           # ISO 8601 过期时间
    email: str = ""
    subject: str = ""


@dataclass(slots=True)
class AuthBundle:
    """完整的认证包，包含 token 和元数据。"""
    token_data: TokenData
    last_refresh: str = ""     # ISO 8601
    base_url: str = DEFAULT_API_BASE
    redirect_uri: str = REDIRECT_URI
    token_endpoint: str = ""


@dataclass(slots=True)
class TokenStorage:
    """持久化的 token 存储格式。"""
    type: str = "xai"
    access_token: str = ""
    refresh_token: str = ""
    id_token: str = ""
    token_type: str = "Bearer"
    expires_in: int = 0
    expire: str = ""
    last_refresh: str = ""
    email: str = ""
    subject: str = ""
    base_url: str = DEFAULT_API_BASE
    redirect_uri: str = REDIRECT_URI
    token_endpoint: str = ""
    auth_kind: str = "oauth"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "id_token": self.id_token,
            "token_type": self.token_type,
            "expires_in": self.expires_in,
            "expire": self.expire,
            "last_refresh": self.last_refresh,
            "email": self.email,
            "subject": self.subject,
            "base_url": self.base_url,
            "redirect_uri": self.redirect_uri,
            "token_endpoint": self.token_endpoint,
            "auth_kind": self.auth_kind,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TokenStorage:
        return cls(
            type=data.get("type", "xai"),
            access_token=data.get("access_token", ""),
            refresh_token=data.get("refresh_token", ""),
            id_token=data.get("id_token", ""),
            token_type=data.get("token_type", "Bearer"),
            expires_in=data.get("expires_in", 0),
            expire=data.get("expire", ""),
            last_refresh=data.get("last_refresh", ""),
            email=data.get("email", ""),
            subject=data.get("subject", ""),
            base_url=data.get("base_url", DEFAULT_API_BASE),
            redirect_uri=data.get("redirect_uri", REDIRECT_URI),
            token_endpoint=data.get("token_endpoint", ""),
            auth_kind=data.get("auth_kind", "oauth"),
        )

    @property
    def is_expired(self) -> bool:
        """检查 token 是否已过期（考虑 5 分钟提前量）。"""
        if not self.expire:
            return False
        try:
            expire_time = _parse_iso8601(self.expire)
            return time.time() >= (expire_time - REFRESH_LEAD_SECONDS)
        except Exception:
            return False

    @property
    def needs_refresh(self) -> bool:
        return self.is_expired and bool(self.refresh_token)


# ── PKCE ──────────────────────────────────────────────────────────────────


def generate_pkce_codes() -> PKCECodes:
    """生成 PKCE code_verifier / code_challenge 对。"""
    # CPA 使用 96 字节随机数
    raw = secrets.token_bytes(96)
    verifier = base64.urlsafe_b64encode(raw).rstrip(b"=").decode()
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return PKCECodes(code_verifier=verifier, code_challenge=challenge)


# ── OIDC 发现 ─────────────────────────────────────────────────────────────


async def discover_oidc(http_client: httpx.AsyncClient | None = None) -> Discovery:
    """从 OIDC 发现端点获取 OAuth 端点。"""
    client = http_client or httpx.AsyncClient(timeout=15)
    try:
        resp = await client.get(DISCOVERY_URL, headers={"Accept": "application/json"})
        resp.raise_for_status()
        data = resp.json()
        auth_endpoint = data.get("authorization_endpoint", "")
        token_endpoint = data.get("token_endpoint", "")
        if not auth_endpoint or not token_endpoint:
            raise ValueError(f"OIDC discovery returned incomplete data: {data}")
        return Discovery(
            authorization_endpoint=auth_endpoint,
            token_endpoint=token_endpoint,
        )
    finally:
        if http_client is None:
            await client.aclose()


# ── 授权 URL 构建 ─────────────────────────────────────────────────────────


def build_authorize_url(
    authorization_endpoint: str,
    code_challenge: str,
    state: str,
    nonce: str,
) -> str:
    """构建浏览器授权 URL。"""
    params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPE,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": state,
        "nonce": nonce,
        "plan": "generic",
        "referrer": "cli-proxy-api",
    }
    return f"{authorization_endpoint}?{urlencode(params)}"


# ── Token 交换 ────────────────────────────────────────────────────────────


async def exchange_code_for_tokens(
    code: str,
    pkce: PKCECodes,
    token_endpoint: str = "",
    http_client: httpx.AsyncClient | None = None,
) -> AuthBundle:
    """用授权码交换 access_token + refresh_token。"""
    if not token_endpoint:
        discovery = await discover_oidc(http_client)
        token_endpoint = discovery.token_endpoint

    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": REDIRECT_URI,
        "client_id": CLIENT_ID,
        "code_verifier": pkce.code_verifier,
    }

    token_data = await _post_token(token_endpoint, data, http_client)
    return AuthBundle(
        token_data=token_data,
        last_refresh=_now_iso(),
        base_url=DEFAULT_API_BASE,
        redirect_uri=REDIRECT_URI,
        token_endpoint=token_endpoint,
    )


async def refresh_access_token(
    refresh_token: str,
    token_endpoint: str = "",
    http_client: httpx.AsyncClient | None = None,
) -> TokenData:
    """用 refresh_token 刷新 access_token。"""
    if not token_endpoint:
        discovery = await discover_oidc(http_client)
        token_endpoint = discovery.token_endpoint

    data = {
        "grant_type": "refresh_token",
        "client_id": CLIENT_ID,
        "refresh_token": refresh_token,
    }

    return await _post_token(token_endpoint, data, http_client)


# ── 内部辅助 ──────────────────────────────────────────────────────────────


async def _post_token(
    token_endpoint: str,
    form_data: dict[str, str],
    http_client: httpx.AsyncClient | None = None,
) -> TokenData:
    """POST token endpoint，返回 TokenData。"""
    client = http_client or httpx.AsyncClient(timeout=30)
    try:
        resp = await client.post(
            token_endpoint,
            data=form_data,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            },
        )
        resp.raise_for_status()
        payload = resp.json()

        access_token = payload.get("access_token", "")
        if not access_token:
            raise ValueError("Token response missing access_token")

        email, subject = _decode_jwt_identity(payload.get("id_token", ""))

        return TokenData(
            access_token=access_token,
            refresh_token=payload.get("refresh_token", ""),
            id_token=payload.get("id_token", ""),
            token_type=payload.get("token_type", "Bearer"),
            expires_in=payload.get("expires_in", 0),
            expire=_expire_from_now(payload.get("expires_in", 0)),
            email=email,
            subject=subject,
        )
    finally:
        if http_client is None:
            await client.aclose()


def _decode_jwt_identity(id_token: str) -> tuple[str, str]:
    """从 JWT id_token 中提取 email 和 subject。"""
    if not id_token:
        return "", ""
    try:
        parts = id_token.split(".")
        if len(parts) < 2:
            return "", ""
        payload_b64 = parts[1]
        payload_b64 += "=" * (4 - len(payload_b64) % 4 % 4)
        raw = base64.urlsafe_b64decode(payload_b64)
        claims = json.loads(raw)
        return claims.get("email", ""), claims.get("sub", "")
    except Exception:
        return "", ""


def _expire_from_now(expires_in: int) -> str:
    """根据 expires_in 秒数计算 ISO 8601 过期时间。"""
    if expires_in <= 0:
        return ""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + expires_in))


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _parse_iso8601(value: str) -> float:
    """解析 ISO 8601 字符串为 Unix 时间戳。"""
    import datetime
    # 尝试多种格式
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            dt = datetime.datetime.strptime(value, fmt)
            if dt.tzinfo:
                return dt.timestamp()
            return dt.replace(tzinfo=datetime.timezone.utc).timestamp()
        except ValueError:
            continue
    # 回退：假设 UTC
    return 0.0


# ── Token 文件存储 ────────────────────────────────────────────────────────


def save_token_to_file(storage: TokenStorage, filepath: Path) -> None:
    """将 token 保存到 JSON 文件。"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(storage.to_dict(), f, indent=2, ensure_ascii=False)
    logger.info("xai oauth token saved: path={}", filepath)


def load_token_from_file(filepath: Path) -> TokenStorage | None:
    """从 JSON 文件加载 token。"""
    if not filepath.exists():
        return None
    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        return TokenStorage.from_dict(data)
    except Exception as exc:
        logger.warning("xai oauth token load failed: path={} error={}", filepath, exc)
        return None


def credential_filename(email: str, subject: str = "") -> str:
    """生成 token 文件名，如 xai-user@example.com.json"""
    if email:
        safe = _sanitize_filename(email)
        return f"xai-{safe}.json"
    if subject:
        safe = _sanitize_filename(subject)
        return f"xai-{safe}.json"
    return f"xai-{int(time.time() * 1000)}.json"


def _sanitize_filename(value: str) -> str:
    """清理文件名中的非法字符。"""
    result = []
    for ch in value:
        if ch.isalnum() or ch in "@._-":
            result.append(ch)
        else:
            result.append("-")
    return "".join(result).strip("-")


# ── 导出 ──────────────────────────────────────────────────────────────────

__all__ = [
    "PKCECodes", "Discovery", "TokenData", "AuthBundle", "TokenStorage",
    "generate_pkce_codes", "discover_oidc", "build_authorize_url",
    "exchange_code_for_tokens", "refresh_access_token",
    "save_token_to_file", "load_token_from_file", "credential_filename",
    "CLIENT_ID", "SCOPE", "REDIRECT_URI", "CALLBACK_PORT", "DEFAULT_API_BASE",
    "ISSUER", "DISCOVERY_URL",
]
