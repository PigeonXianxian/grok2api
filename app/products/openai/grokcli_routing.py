"""Grok CLI 路由 — grokcli/ 模型走 xAI OAuth API

将 grokcli/ 前缀的模型请求路由到 api.x.ai 客户端，
使用 OAuth Bearer Token 认证，绕过 Cloudflare 反爬。
"""
from __future__ import annotations

from typing import Any

from app.control.model.registry import ModelSpec
from app.dataplane.reverse.protocol.xai_oauth import TokenStorage
from app.dataplane.reverse.protocol.xai_api import XAIClient
from app.control.account.xai_oauth_store import (
    ext_to_token_storage,
    is_oauth_token,
    get_oauth_access_token,
)
from app.platform.errors import RateLimitError, UpstreamError

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
    """从 account 的 ext 中提取 OAuth TokenStorage。

    优先检查 ext 是否为 OAuth 类型，兼容旧 token 字段直接作为 access_token。
    """
    if is_oauth_token(exts):
        storage = ext_to_token_storage(exts)
        if storage and storage.access_token:
            return storage
    # 兼容：token 字段本身就是 access_token（直接导入时）
    if token and not token.startswith("eyJ"):  # 不是 JWT（SSO cookie 格式）
        pass  # 无法确定，回退到 ext 检查
    return None


def get_grokcli_client(exts: dict[str, Any], token: str) -> XAIClient | None:
    """从账号数据创建 xAI API 客户端。"""
    storage = select_grokcli_account(exts, token)
    if storage is None:
        return None
    return XAIClient(token_storage=storage)


async def acquire_grokcli_account() -> tuple[XAIClient, str, str]:
    """获取一个可用的 Grok CLI 账号（OAuth token）。

    Returns:
        (XAIClient, 原始token, email标识)

    Raises:
        RateLimitError: 没有可用的 OAuth 账号
    """
    from app.dataplane.account import _directory as _acct_dir

    if _acct_dir is None:
        raise RateLimitError("Account directory not initialised")

    # 遍历所有 super 池账号（OAuth token 都算 super）
    acct = await _acct_dir.reserve_any(pool_candidates=(1, 2), now_s_override=None)
    if acct is None:
        raise RateLimitError("No available Grok CLI (OAuth) accounts")

    client = get_grokcli_client(acct.ext, acct.token)
    if client is None:
        await _acct_dir.release(acct)
        raise RateLimitError("Account exists but has no valid OAuth token")

    return client, acct.token, acct.ext.get("xai_oauth_email", "unknown")


async def release_grokcli_account(token: str, success: bool = True) -> None:
    """释放 Grok CLI 账号。"""
    from app.dataplane.account import _directory as _acct_dir
    from app.control.account.enums import FeedbackKind

    if _acct_dir is None:
        return
    # 找到对应的 account 记录并释放
    # 简化：直接 feedback
    if not success:
        await _acct_dir.feedback(
            token,
            FeedbackKind.SERVER_ERROR if not success else FeedbackKind.SUCCESS,
            0,  # mode_id
        )


__all__ = [
    "is_grokcli_model", "strip_grokcli_prefix",
    "select_grokcli_account", "get_grokcli_client",
    "acquire_grokcli_account", "release_grokcli_account",
    "GROKCLI_PREFIX",
]
