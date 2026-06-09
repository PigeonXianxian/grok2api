"""xAI OAuth token 自动刷新钩子

在账户刷新流程中插入：检测 OAuth token 是否过期，自动用 refresh_token 续期。
不修改原有 refresh.py，作为独立钩子在需要时调用。
"""
from __future__ import annotations

import time as _time
from typing import TYPE_CHECKING

from app.platform.logging.logger import logger

if TYPE_CHECKING:
    from .models import AccountRecord
    from .repository import AccountRepository


async def ensure_oauth_token_fresh(
    record: "AccountRecord",
    repo: "AccountRepository",
) -> bool:
    """检查并刷新 xAI OAuth token。返回 True 表示 token 有效（可能已刷新）。"""
    if record.ext.get("xai_oauth_type") != "xai_oauth":
        return True  # 不是 OAuth token，跳过

    refresh_token = record.ext.get("xai_oauth_refresh_token", "")
    expire_str = record.ext.get("xai_oauth_expire", "")

    if not refresh_token or not expire_str:
        return True  # 没有 refresh_token，无法刷新，保持原样

    from app.dataplane.reverse.protocol.xai_oauth import (
        _parse_iso8601,
        REFRESH_LEAD_SECONDS,
        refresh_access_token,
    )

    try:
        expire_ts = _parse_iso8601(expire_str)
    except Exception:
        return True  # 解析失败，跳过

    if _time.time() < (expire_ts - REFRESH_LEAD_SECONDS):
        return True  # 还没到刷新时间

    # 需要刷新
    logger.info(
        "refreshing xai oauth token: email={}",
        record.ext.get("xai_oauth_email", "?"),
    )
    try:
        token_endpoint = record.ext.get("xai_oauth_token_endpoint", "")
        token_data = await refresh_access_token(refresh_token, token_endpoint)

        from app.control.account.xai_oauth_store import update_ext_after_refresh
        from app.control.account.commands import AccountUpsert

        new_ext = update_ext_after_refresh(dict(record.ext), token_data)
        await repo.upsert(AccountUpsert(
            token=token_data.access_token,
            account_id=record.account_id,
            pool=record.pool,
            ext=new_ext,
        ))
        logger.info(
            "xai oauth token refreshed: email={}",
            record.ext.get("xai_oauth_email", "?"),
        )
        return True
    except Exception as exc:
        logger.warning(
            "xai oauth token refresh failed: email={} error={}",
            record.ext.get("xai_oauth_email", "?"), exc,
        )
        return False


async def refresh_oauth_tokens_batch(
    records: list["AccountRecord"],
    repo: "AccountRepository",
) -> tuple[int, int]:
    """批量刷新一批账号中的 OAuth token。返回 (成功数, 失败数)。"""
    ok = 0
    fail = 0
    for record in records:
        if record.ext.get("xai_oauth_type") != "xai_oauth":
            continue
        result = await ensure_oauth_token_fresh(record, repo)
        if result:
            ok += 1
        else:
            fail += 1
    return ok, fail


__all__ = ["ensure_oauth_token_fresh", "refresh_oauth_tokens_batch"]
