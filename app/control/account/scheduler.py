"""Background scheduler for periodic account quota refresh.

Runs one independent loop per pool type (basic / super / heavy), each with
its own configurable interval read from:

    account.refresh.basic_interval_sec  (default 86400 — 24 h)
    account.refresh.super_interval_sec  (default  7200 —  2 h)
    account.refresh.heavy_interval_sec  (default  7200 —  2 h)

Plus a monthly CLI quota reset loop (runs every 3600 s — 1 h).
"""

import asyncio
import time

from app.platform.config.snapshot import get_config
from app.platform.logging.logger import logger
from .refresh import AccountRefreshService

# Pool → (config key, built-in default seconds)
_POOL_CONFIG: dict[str, tuple[str, int]] = {
    "basic": ("account.refresh.basic_interval_sec", 86_400),
    "super": ("account.refresh.super_interval_sec",  7_200),
    "heavy": ("account.refresh.heavy_interval_sec",  7_200),
}

_CLI_QUOTA_RESET_INTERVAL = 3600  # 1 hour


def _interval(pool: str) -> int:
    key, default = _POOL_CONFIG[pool]
    v = get_config(key, None)
    return int(v) if v is not None else default


class AccountRefreshScheduler:
    """Runs one refresh loop per pool type at pool-specific intervals.

    Lifecycle:  ``start()`` → loops run in background → ``stop()`` to cancel.
    """

    def __init__(self, refresh_service: AccountRefreshService) -> None:
        self._service = refresh_service
        self._tasks:  list[asyncio.Task] = []
        self._stop    = asyncio.Event()

    def bind_service(self, refresh_service: AccountRefreshService) -> None:
        """Update the refresh service used by the singleton scheduler."""
        self._service = refresh_service

    def is_running(self) -> bool:
        """Return True while any pool refresh loop is still active."""
        return any(not task.done() for task in self._tasks)

    def start(self) -> None:
        if self.is_running():
            return
        self._stop.clear()
        self._tasks = [
            asyncio.create_task(self._loop(pool), name=f"account-refresh-{pool}")
            for pool in _POOL_CONFIG
        ]
        self._tasks.append(
            asyncio.create_task(self._cli_quota_reset_loop(), name="cli-quota-reset")
        )
        intervals = {p: _interval(p) for p in _POOL_CONFIG}
        logger.info(
            "account refresh scheduler started: basic_interval_s={} super_interval_s={} heavy_interval_s={} cli_quota_interval_s={}",
            intervals["basic"], intervals["super"], intervals["heavy"], _CLI_QUOTA_RESET_INTERVAL,
        )

    def stop(self) -> None:
        was_running = self.is_running()
        self._stop.set()
        for t in self._tasks:
            if not t.done():
                t.cancel()
        self._tasks = []
        if was_running:
            logger.info("account refresh scheduler stopped")

    async def _loop(self, pool: str) -> None:
        while not self._stop.is_set():
            interval = _interval(pool)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=float(interval))
                break  # stop event fired
            except asyncio.TimeoutError:
                pass

            if self._stop.is_set():
                break

            try:
                result = await self._service.refresh_scheduled(pool=pool)
                logger.info(
                    "account refresh cycle completed: pool={} checked={} refreshed={} recovered={} failed={}",
                    pool, result.checked, result.refreshed, result.recovered, result.failed,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error(
                    "account refresh cycle failed: pool={} error_type={} error={}",
                    pool,
                    type(exc).__name__,
                    exc,
                )

    async def _cli_quota_reset_loop(self) -> None:
        """Periodic loop that checks and resets CLI monthly quotas via repository."""
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(
                    self._stop.wait(), timeout=float(_CLI_QUOTA_RESET_INTERVAL)
                )
                break
            except asyncio.TimeoutError:
                pass

            if self._stop.is_set():
                break

            try:
                await self._reset_cli_monthly_quotas()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error(
                    "cli quota reset loop failed: error_type={} error={}",
                    type(exc).__name__, exc,
                )

    async def _reset_cli_monthly_quotas(self) -> None:
        """Scan OAuth accounts via repository: reset expired monthly quotas and sync usage counts."""
        from app.control.account.xai_oauth_store import is_oauth_token
        from app.control.account.cli_quota import (
            check_and_reset_monthly,
            EXT_CLI_QUOTA_USED,
        )
        from app.control.account.commands import AccountPatch, ListAccountsQuery

        repo = self._service._repo
        patches: list[AccountPatch] = []
        reset_count = 0
        sync_count = 0

        page_num = 1
        while True:
            page = await repo.list_accounts(ListAccountsQuery(page=page_num, page_size=2000))
            for record in page.items:
                if not is_oauth_token(record.ext):
                    continue

                ext_changed = False

                if check_and_reset_monthly(record.ext):
                    ext_changed = True
                    reset_count += 1

                current_used = int(record.ext.get(EXT_CLI_QUOTA_USED, 0) or 0)
                call_count = record.usage_use_count or 0
                last_sync_count = int(record.ext.get("cli_quota_last_sync_usage", 0) or 0)
                if call_count > last_sync_count:
                    delta = call_count - last_sync_count
                    record.ext[EXT_CLI_QUOTA_USED] = current_used + delta
                    record.ext["cli_quota_last_sync_usage"] = call_count
                    ext_changed = True
                    sync_count += 1

                if ext_changed:
                    patches.append(AccountPatch(
                        token=record.token,
                        ext_merge=record.ext,
                    ))

            if page_num * 2000 >= page.total:
                break
            page_num += 1

        if patches:
            try:
                await repo.patch_accounts(patches)
            except Exception as exc:
                logger.error("cli quota sync patch failed: error={}", exc)

        if reset_count > 0 or sync_count > 0:
            logger.info(
                "cli quota sync: accounts_reset={} accounts_synced={}",
                reset_count, sync_count,
            )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_scheduler: AccountRefreshScheduler | None = None


def get_account_refresh_scheduler(
    refresh_service: AccountRefreshService,
) -> AccountRefreshScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = AccountRefreshScheduler(refresh_service)
    else:
        _scheduler.bind_service(refresh_service)
    return _scheduler


__all__ = ["AccountRefreshScheduler", "get_account_refresh_scheduler"]
