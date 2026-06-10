"""xAI OAuth token management — integrated with account system, auto-refresh and monthly quota.

Tokens are stored in AccountRecord.ext fields, deduplicated via account_id.
Monthly OAuth token refresh and quota sync.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from app.platform.logging.logger import logger
from app.dataplane.reverse.protocol.xai_oauth import (
    TokenStorage,
    refresh_access_token,
    load_token_from_file,
    DEFAULT_API_BASE,
)
from app.control.account.cli_quota import (
    EXT_CLI_QUOTA_MONTHLY,
    EXT_CLI_QUOTA_USED,
    EXT_CLI_QUOTA_RESET,
    init_cli_quota,
    cli_quota_summary,
)

# ext 字段中的 key
EXT_KEY_TYPE = "xai_oauth_type"
EXT_KEY_ACCESS_TOKEN = "xai_oauth_access_token"
EXT_KEY_REFRESH_TOKEN = "xai_oauth_refresh_token"
EXT_KEY_EXPIRE = "xai_oauth_expire"
EXT_KEY_EMAIL = "xai_oauth_email"
EXT_KEY_BASE_URL = "xai_oauth_base_url"
EXT_KEY_TOKEN_ENDPOINT = "xai_oauth_token_endpoint"
EXT_KEY_LAST_REFRESH = "xai_oauth_last_refresh"
EXT_KEY_QUOTA_MONTHLY = "xai_oauth_quota_monthly"
EXT_KEY_QUOTA_USED = "xai_oauth_quota_used"
EXT_KEY_QUOTA_RESET = "xai_oauth_quota_reset"


async def import_oauth_token(
    token_storage: TokenStorage,
    pool: str = "super",
) -> dict[str, Any]:
    """Create AccountRecord ext data from TokenStorage.

    Registers a new xAI OAuth token in the account system with
    monthly CLI quota initialisation.
    """
    ext = {
        EXT_KEY_TYPE: "xai_oauth",
        EXT_KEY_ACCESS_TOKEN: token_storage.access_token,
        EXT_KEY_REFRESH_TOKEN: token_storage.refresh_token,
        EXT_KEY_EXPIRE: token_storage.expire,
        EXT_KEY_EMAIL: token_storage.email,
        EXT_KEY_BASE_URL: token_storage.base_url or DEFAULT_API_BASE,
        EXT_KEY_TOKEN_ENDPOINT: token_storage.token_endpoint,
        EXT_KEY_LAST_REFRESH: token_storage.last_refresh,
        EXT_KEY_QUOTA_MONTHLY: 0,
        EXT_KEY_QUOTA_USED: 0,
        EXT_KEY_QUOTA_RESET: "",
    }
    ext = init_cli_quota(ext)
    return ext


def ext_to_token_storage(ext: dict[str, Any]) -> TokenStorage | None:
    """从 AccountRecord.ext 还原 TokenStorage。"""
    if ext.get(EXT_KEY_TYPE) != "xai_oauth":
        return None
    if not ext.get(EXT_KEY_ACCESS_TOKEN):
        return None
    return TokenStorage(
        type="xai",
        access_token=ext.get(EXT_KEY_ACCESS_TOKEN, ""),
        refresh_token=ext.get(EXT_KEY_REFRESH_TOKEN, ""),
        expire=ext.get(EXT_KEY_EXPIRE, ""),
        email=ext.get(EXT_KEY_EMAIL, ""),
        base_url=ext.get(EXT_KEY_BASE_URL, DEFAULT_API_BASE),
        token_endpoint=ext.get(EXT_KEY_TOKEN_ENDPOINT, ""),
        last_refresh=ext.get(EXT_KEY_LAST_REFRESH, ""),
        auth_kind="oauth",
    )


def update_ext_after_refresh(ext: dict[str, Any], token_data: Any) -> dict[str, Any]:
    """刷新 token 后更新 ext 字段。"""
    ext[EXT_KEY_ACCESS_TOKEN] = token_data.access_token
    if token_data.refresh_token:
        ext[EXT_KEY_REFRESH_TOKEN] = token_data.refresh_token
    ext[EXT_KEY_EXPIRE] = token_data.expire
    ext[EXT_KEY_LAST_REFRESH] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
    )
    return ext


def is_oauth_token(ext: dict[str, Any]) -> bool:
    """判断 ext 是否为 xAI OAuth token。"""
    return ext.get(EXT_KEY_TYPE) == "xai_oauth"


def get_oauth_access_token(ext: dict[str, Any]) -> str:
    """从 ext 中提取 access_token。"""
    return ext.get(EXT_KEY_ACCESS_TOKEN, "")


def get_oauth_identity(ext: dict[str, Any]) -> str:
    """从 ext 中提取身份标识（email）。"""
    return ext.get(EXT_KEY_EMAIL, "")


# ── 文件导入 ──────────────────────────────────────────────────────────────


async def import_oauth_from_file(
    filepath: Path,
    pool: str = "super",
) -> dict[str, Any] | None:
    """从 JSON 文件导入 xAI OAuth token 到 ext 格式。"""
    storage = load_token_from_file(filepath)
    if storage is None:
        return None
    if storage.needs_refresh:
        logger.info("imported oauth token needs refresh, refreshing...")
        try:
            token_data = await refresh_access_token(
                storage.refresh_token,
                storage.token_endpoint,
            )
            storage.access_token = token_data.access_token
            if token_data.refresh_token:
                storage.refresh_token = token_data.refresh_token
            storage.expire = token_data.expire
            storage.last_refresh = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            )
        except Exception as exc:
            logger.warning("oauth token refresh on import failed: {}", exc)
    return await import_oauth_token(storage, pool)


# ── 导出 ──────────────────────────────────────────────────────────────────

__all__ = [
    "import_oauth_token", "ext_to_token_storage", "update_ext_after_refresh",
    "is_oauth_token", "get_oauth_access_token", "get_oauth_identity",
    "import_oauth_from_file",
    "EXT_KEY_TYPE", "EXT_KEY_ACCESS_TOKEN", "EXT_KEY_REFRESH_TOKEN",
    "EXT_KEY_EXPIRE", "EXT_KEY_EMAIL", "EXT_KEY_BASE_URL",
    "EXT_KEY_TOKEN_ENDPOINT", "EXT_KEY_LAST_REFRESH",
    "EXT_KEY_QUOTA_MONTHLY", "EXT_KEY_QUOTA_USED", "EXT_KEY_QUOTA_RESET",
    "EXT_CLI_QUOTA_MONTHLY", "EXT_CLI_QUOTA_USED", "EXT_CLI_QUOTA_RESET",
    "cli_quota_summary",
]
