"""Grok CLI monthly quota management.

Monthly credits reset at 00:00 on the first day of each month.
Quota data is stored in AccountRecord.ext fields for OAuth (grokcli/) accounts.
"""

from __future__ import annotations

import time
from calendar import monthrange
from dataclasses import dataclass
from typing import Any

from app.platform.logging.logger import logger

# ext field keys for CLI monthly quota
EXT_CLI_QUOTA_MONTHLY = "cli_quota_monthly_total"
EXT_CLI_QUOTA_USED = "cli_quota_used"
EXT_CLI_QUOTA_RESET = "cli_quota_reset_at"

# Default monthly quota for CLI accounts (used when not synced from upstream)
DEFAULT_CLI_MONTHLY_QUOTA = 5000


@dataclass(slots=True)
class CLIMonthlyQuota:
    total: int = DEFAULT_CLI_MONTHLY_QUOTA
    used: int = 0
    reset_at: float = 0.0

    @property
    def remaining(self) -> int:
        return max(0, self.total - self.used)

    @property
    def is_exhausted(self) -> bool:
        return self.used >= self.total

    @property
    def reset_at_iso(self) -> str:
        if self.reset_at <= 0:
            return ""
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.reset_at))


def _compute_next_reset_ts() -> float:
    """Compute the timestamp (UTC) of next month's first day at 00:00."""
    now = time.time()
    gm = time.gmtime(now)
    year, month = gm.tm_year, gm.tm_mon

    if month == 12:
        next_year, next_month = year + 1, 1
    else:
        next_year, next_month = year, month + 1

    tm = time.struct_time((next_year, next_month, 1, 0, 0, 0, 0, 0, 0))
    return time.mktime(tm) - time.timezone  # convert local to UTC


def get_cli_quota(ext: dict[str, Any]) -> CLIMonthlyQuota:
    """Read CLI monthly quota from account ext fields."""
    total = int(ext.get(EXT_CLI_QUOTA_MONTHLY, 0) or 0)
    if total <= 0:
        total = DEFAULT_CLI_MONTHLY_QUOTA
    used = int(ext.get(EXT_CLI_QUOTA_USED, 0) or 0)
    reset_at = float(ext.get(EXT_CLI_QUOTA_RESET, 0) or 0)
    return CLIMonthlyQuota(total=total, used=used, reset_at=reset_at)


def init_cli_quota(ext: dict[str, Any], monthly_total: int = 0) -> dict[str, Any]:
    """Initialize CLI quota fields in ext for a new OAuth account."""
    total = monthly_total if monthly_total > 0 else DEFAULT_CLI_MONTHLY_QUOTA
    next_reset = _compute_next_reset_ts()
    ext[EXT_CLI_QUOTA_MONTHLY] = total
    ext[EXT_CLI_QUOTA_USED] = 0
    ext[EXT_CLI_QUOTA_RESET] = next_reset
    return ext


def check_and_reset_monthly(ext: dict[str, Any]) -> bool:
    """Check if monthly quota needs resetting. Returns True if reset occurred."""
    reset_at = float(ext.get(EXT_CLI_QUOTA_RESET, 0) or 0)
    now_ts = time.time()

    if reset_at <= 0:
        next_reset = _compute_next_reset_ts()
        ext[EXT_CLI_QUOTA_RESET] = next_reset
        ext[EXT_CLI_QUOTA_USED] = 0
        logger.info("cli monthly quota initialized: reset_at={}", time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(next_reset)))
        return True

    if now_ts >= reset_at:
        ext[EXT_CLI_QUOTA_USED] = 0
        next_reset = _compute_next_reset_ts()
        ext[EXT_CLI_QUOTA_RESET] = next_reset
        logger.info("cli monthly quota reset: next_reset={}", time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(next_reset)))
        return True

    return False


def deduct_cli_quota(ext: dict[str, Any], amount: int = 1) -> int:
    """Deduct from monthly quota. Returns new remaining count."""
    check_and_reset_monthly(ext)
    current_used = int(ext.get(EXT_CLI_QUOTA_USED, 0) or 0)
    current_used += amount
    ext[EXT_CLI_QUOTA_USED] = current_used
    total = int(ext.get(EXT_CLI_QUOTA_MONTHLY, DEFAULT_CLI_MONTHLY_QUOTA) or DEFAULT_CLI_MONTHLY_QUOTA)
    return max(0, total - current_used)


def has_cli_quota(ext: dict[str, Any]) -> bool:
    """Check if account has remaining CLI monthly quota."""
    check_and_reset_monthly(ext)
    used = int(ext.get(EXT_CLI_QUOTA_USED, 0) or 0)
    total = int(ext.get(EXT_CLI_QUOTA_MONTHLY, DEFAULT_CLI_MONTHLY_QUOTA) or DEFAULT_CLI_MONTHLY_QUOTA)
    return used < total


def cli_quota_summary(ext: dict[str, Any]) -> dict[str, Any]:
    """Return a human-readable summary of CLI quota state."""
    check_and_reset_monthly(ext)
    total = int(ext.get(EXT_CLI_QUOTA_MONTHLY, DEFAULT_CLI_MONTHLY_QUOTA) or DEFAULT_CLI_MONTHLY_QUOTA)
    used = int(ext.get(EXT_CLI_QUOTA_USED, 0) or 0)
    reset_at = float(ext.get(EXT_CLI_QUOTA_RESET, 0) or 0)
    return {
        "monthly_total": total,
        "monthly_used": used,
        "monthly_remaining": max(0, total - used),
        "reset_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(reset_at)) if reset_at > 0 else "",
    }


__all__ = [
    "CLIMonthlyQuota",
    "EXT_CLI_QUOTA_MONTHLY",
    "EXT_CLI_QUOTA_USED",
    "EXT_CLI_QUOTA_RESET",
    "DEFAULT_CLI_MONTHLY_QUOTA",
    "get_cli_quota",
    "init_cli_quota",
    "check_and_reset_monthly",
    "deduct_cli_quota",
    "has_cli_quota",
    "cli_quota_summary",
]
