"""Grok CLI 路由 — grokcli/ 模型走 xAI OAuth API

将 grokcli/ 前缀的模型请求路由到 api.x.ai 客户端，
使用 OAuth Bearer Token 认证，绕过 Cloudflare 反爬。
包含月度 CLI 额度显示和调度优化。
"""
from __future__ import annotations

from typing import Any

from app.dataplane.reverse.protocol.xai_oauth import TokenStorage
from app.dataplane.reverse.protocol.xai_api import XAIClient
from app.control.account.xai_oauth_store import (
    ext_to_token_storage,
    is_oauth_token,
)
from app.control.account.cli_quota import cli_quota_summary
from app.platform.errors import RateLimitError

GROKCLI_PREFIX = "grokcli/"


def is_grokcli_model(model_name: str) -> bool:
    """判断模型是否为 Grok CLI 路由。"""
    return model_name.startswith(GROKCLI_PREFIX)


def strip_grokcli_prefix(model_name: str) -> str:
    """去掉 grokcli/ 前缀，拿到原始 xAI 模型名。"""
    if model_name.startswith(GROKCLI_PREFIX):
        return model_name[len(GROKCLI_PREFIX):]
    return model_name


def select_grokcli_account(exts: dict[str, Any], token: str) -> TokenStorage | None:
    """从 account 的 ext 中提取 OAuth TokenStorage。"""
    if is_oauth_token(exts):
        storage = ext_to_token_storage(exts)
        if storage and storage.access_token:
            return storage
    return None


def get_grokcli_client(exts: dict[str, Any], token: str) -> XAIClient | None:
    """从账号数据创建 xAI API 客户端。"""
    storage = select_grokcli_account(exts, token)
    if storage is None:
        return None
    return XAIClient(token_storage=storage)


async def acquire_grokcli_account() -> tuple[XAIClient, str, str]:
    """获取一个可用的 Grok CLI 账号（OAuth token）。

    从 super/heavy 池中预留账号，token 字段即为 OAuth access_token。

    Returns:
        (XAIClient, 原始token, email标识)

    Raises:
        RateLimitError: 没有可用的 OAuth 账号
    """
    from app.dataplane.account import _directory as _acct_dir

    if _acct_dir is None:
        raise RateLimitError("Account directory not initialised")

    lease = await _acct_dir.reserve_any(pool_candidates=(1, 2), now_s_override=None)
    if lease is None:
        raise RateLimitError("No available Grok CLI (OAuth) accounts")

    ext = await _lookup_account_ext(lease.token)
    if ext is None or not is_oauth_token(ext):
        await _acct_dir.release(lease)
        raise RateLimitError("Account exists but has no valid OAuth token")

    client = get_grokcli_client(ext, lease.token)
    if client is None:
        await _acct_dir.release(lease)
        raise RateLimitError("Account has OAuth token but failed to create client")

    return client, lease.token, ext.get("xai_oauth_email", "unknown")


_repo_cache: Any = None


async def _get_repo() -> Any:
    """延迟获取 AccountRepository 单例。"""
    global _repo_cache
    if _repo_cache is not None:
        return _repo_cache
    from app.control.account.backends.factory import create_repository
    _repo_cache = create_repository()
    return _repo_cache


async def _lookup_account_ext(token: str) -> dict[str, Any] | None:
    """通过 token 从 repository 查找账号的 ext 字段。"""
    try:
        repo = await _get_repo()
        records = await repo.get_accounts([token])
        if records:
            return records[0].ext
    except Exception:
        pass
    return None


async def release_grokcli_account(token: str, success: bool = True) -> None:
    """释放 Grok CLI 账号。月度额度由后台调度器周期性刷新。"""
    from app.dataplane.account import _directory as _acct_dir
    from app.control.account.enums import FeedbackKind

    if _acct_dir is None:
        return

    if success:
        await _acct_dir.feedback(token, FeedbackKind.SUCCESS, 0)
    else:
        await _acct_dir.feedback(token, FeedbackKind.SERVER_ERROR, 0)


def get_grokcli_quota_info(ext: dict[str, Any]) -> dict[str, Any]:
    """获取 Grok CLI 月度额度信息（供 admin 面板展示）。"""
    if not is_oauth_token(ext):
        return {}
    return cli_quota_summary(ext)


__all__ = [
    "is_grokcli_model", "strip_grokcli_prefix",
    "select_grokcli_account", "get_grokcli_client",
    "acquire_grokcli_account", "release_grokcli_account",
    "get_grokcli_quota_info",
    "GROKCLI_PREFIX",
]
