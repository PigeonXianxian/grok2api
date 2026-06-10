"""CLI quota admin endpoints — /admin/api/cli/*"""

from typing import TYPE_CHECKING

import orjson
from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response

from app.control.account.xai_oauth_store import is_oauth_token
from app.control.account.cli_quota import cli_quota_summary

if TYPE_CHECKING:
    from app.control.account.repository import AccountRepository

router = APIRouter(tags=["Admin - CLI Quota"])


def _get_repo(request: Request) -> "AccountRepository":
    return request.app.state.repository


def _json(data, status_code: int = 200) -> Response:
    return Response(
        content=orjson.dumps(data),
        media_type="application/json",
        status_code=status_code,
    )


@router.get("/cli/quota")
async def list_cli_quota(repo: "AccountRepository" = Depends(_get_repo)):
    """List all Grok CLI (OAuth) accounts with their monthly quota info."""
    from app.control.account.commands import ListAccountsQuery

    items: list = []
    page_num = 1
    while True:
        page = await repo.list_accounts(
            ListAccountsQuery(page=page_num, page_size=2000)
        )
        for record in page.items:
            if not is_oauth_token(record.ext):
                continue
            quota_info = cli_quota_summary(record.ext)
            items.append({
                "token": record.token[:8] + "..." + record.token[-8:] if len(record.token) > 20 else record.token,
                "account_id": record.account_id,
                "email": record.ext.get("xai_oauth_email", ""),
                "pool": record.pool,
                "status": record.status,
                "cli_quota": quota_info,
            })
        if page_num * 2000 >= page.total:
            break
        page_num += 1

    return _json({"accounts": items, "total": len(items)})


@router.get("/cli/quota/summary")
async def cli_quota_summary_endpoint(repo: "AccountRepository" = Depends(_get_repo)):
    """Aggregated CLI quota summary across all OAuth accounts."""
    from app.control.account.commands import ListAccountsQuery

    total_monthly = 0
    total_used = 0
    account_count = 0

    page_num = 1
    while True:
        page = await repo.list_accounts(
            ListAccountsQuery(page=page_num, page_size=2000)
        )
        for record in page.items:
            if not is_oauth_token(record.ext):
                continue
            account_count += 1
            qi = cli_quota_summary(record.ext)
            total_monthly += qi["monthly_total"]
            total_used += qi["monthly_used"]
        if page_num * 2000 >= page.total:
            break
        page_num += 1

    return _json({
        "account_count": account_count,
        "total_monthly_quota": total_monthly,
        "total_used": total_used,
        "total_remaining": max(0, total_monthly - total_used),
    })


__all__ = ["router"]
