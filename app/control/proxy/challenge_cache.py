"""Per-node Cloudflare challenge solution cache.

Caches successful CF challenge solutions keyed by (proxy_url, clearance_host)
with TTL-based expiration, LRU eviction, and disk persistence via SQLite.

Features:
  - Per-node granularity: each egress node caches its own CF cookies
  - TTL expiration: cookies expire after configurable lifetime
  - LRU eviction: caps memory usage
  - Disk persistence: survives server restarts
  - Cooldown tracking: exponential backoff per-node after challenge failure
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.platform.logging.logger import logger
from app.platform.paths import data_path

_DEFAULT_TTL_SECONDS = 1800  # 30 min
_DEFAULT_MAX_ENTRIES = 50
_DEFAULT_COOLDOWN_BASE = 10  # seconds
_DEFAULT_COOLDOWN_MAX = 600  # 10 min
_DEFAULT_COOLDOWN_MULTIPLIER = 2.0


@dataclass
class ChallengeCacheEntry:
    """Single cached CF challenge solution."""
    cookies: str = ""
    user_agent: str = ""
    created_at: float = 0.0
    ttl_seconds: int = _DEFAULT_TTL_SECONDS

    @property
    def is_expired(self) -> bool:
        if self.ttl_seconds <= 0:
            return False
        return time.time() - self.created_at > self.ttl_seconds


@dataclass
class CooldownEntry:
    """Per-node challenge cooldown state with exponential backoff."""
    consecutive_failures: int = 0
    cooldown_until: float = 0.0
    base_seconds: int = _DEFAULT_COOLDOWN_BASE
    max_seconds: int = _DEFAULT_COOLDOWN_MAX
    multiplier: float = _DEFAULT_COOLDOWN_MULTIPLIER

    @property
    def is_cooling(self) -> bool:
        return time.time() < self.cooldown_until

    def record_failure(self) -> float:
        self.consecutive_failures += 1
        delay = min(
            self.base_seconds * (self.multiplier ** (self.consecutive_failures - 1)),
            self.max_seconds,
        )
        self.cooldown_until = time.time() + delay
        return delay

    def record_success(self) -> None:
        self.consecutive_failures = 0
        self.cooldown_until = 0.0


def _make_key(proxy_url: str | None, clearance_host: str) -> str:
    return f"{proxy_url or 'direct'}@{clearance_host}"


class ChallengeCache:
    """Thread-safe per-node CF challenge solution cache with disk persistence."""

    def __init__(
        self,
        max_entries: int = _DEFAULT_MAX_ENTRIES,
        default_ttl: int = _DEFAULT_TTL_SECONDS,
        db_path: str | None = None,
    ) -> None:
        self._entries: OrderedDict[str, ChallengeCacheEntry] = OrderedDict()
        self._cooldowns: dict[str, CooldownEntry] = {}
        self._max_entries = max_entries
        self._default_ttl = default_ttl
        self._lock = threading.Lock()
        self._db_path = db_path or str(data_path() / "cf_challenge_cache.db")
        self._init_db()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(
        self, proxy_url: str | None, clearance_host: str
    ) -> ChallengeCacheEntry | None:
        key = _make_key(proxy_url, clearance_host)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.is_expired:
                del self._entries[key]
                self._remove_from_db(key)
                return None
            self._entries.move_to_end(key)
            return entry

    def put(
        self,
        proxy_url: str | None,
        clearance_host: str,
        cookies: str,
        user_agent: str = "",
        ttl_seconds: int | None = None,
    ) -> None:
        key = _make_key(proxy_url, clearance_host)
        entry = ChallengeCacheEntry(
            cookies=cookies,
            user_agent=user_agent,
            created_at=time.time(),
            ttl_seconds=ttl_seconds if ttl_seconds is not None else self._default_ttl,
        )
        with self._lock:
            if key in self._entries:
                del self._entries[key]
            elif len(self._entries) >= self._max_entries:
                oldest_key, _ = self._entries.popitem(last=False)
                self._remove_from_db(oldest_key)
            self._entries[key] = entry
            self._save_to_db(key, entry)

    def remove(self, proxy_url: str | None, clearance_host: str) -> None:
        key = _make_key(proxy_url, clearance_host)
        with self._lock:
            self._entries.pop(key, None)
            self._remove_from_db(key)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._clear_db()

    # ------------------------------------------------------------------
    # Cooldown API
    # ------------------------------------------------------------------

    def is_cooling(self, proxy_url: str | None, clearance_host: str) -> bool:
        key = _make_key(proxy_url, clearance_host)
        with self._lock:
            cd = self._cooldowns.get(key)
            if cd is None:
                return False
            return cd.is_cooling

    def record_challenge_failure(
        self, proxy_url: str | None, clearance_host: str
    ) -> float:
        """Record a challenge failure, returns cooldown delay in seconds."""
        key = _make_key(proxy_url, clearance_host)
        with self._lock:
            cd = self._cooldowns.get(key)
            if cd is None:
                cd = CooldownEntry()
                self._cooldowns[key] = cd
            delay = cd.record_failure()
            logger.debug(
                "cf challenge cooldown: key={} failures={} delay_s={:.1f}",
                key, cd.consecutive_failures, delay,
            )
            return delay

    def record_challenge_success(
        self, proxy_url: str | None, clearance_host: str
    ) -> None:
        """Reset cooldown after successful clearance solve."""
        key = _make_key(proxy_url, clearance_host)
        with self._lock:
            cd = self._cooldowns.get(key)
            if cd:
                cd.record_success()

    def cooldown_state(
        self, proxy_url: str | None, clearance_host: str
    ) -> dict[str, Any]:
        key = _make_key(proxy_url, clearance_host)
        with self._lock:
            cd = self._cooldowns.get(key)
            if cd is None:
                return {"cooling": False, "failures": 0, "delay_s": 0}
            return {
                "cooling": cd.is_cooling,
                "failures": cd.consecutive_failures,
                "delay_s": max(0, cd.cooldown_until - time.time()),
            }

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._entries)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "entries": len(self._entries),
                "cooldowns": len(self._cooldowns),
                "cooldown_items": [
                    {"key": k, "failures": c.consecutive_failures, "cooling": c.is_cooling}
                    for k, c in self._cooldowns.items()
                ],
            }

    # ------------------------------------------------------------------
    # SQLite persistence
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        try:
            os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
            conn = sqlite3.connect(self._db_path)
            conn.execute(
                """CREATE TABLE IF NOT EXISTS challenge_cache (
                    cache_key TEXT PRIMARY KEY,
                    cookies TEXT NOT NULL DEFAULT '',
                    user_agent TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL DEFAULT 0,
                    ttl_seconds INTEGER NOT NULL DEFAULT 1800
                )"""
            )
            conn.commit()
            self._load_from_db(conn)
            conn.close()
        except Exception as exc:
            logger.debug("cf challenge cache db init failed, using memory-only: error={}", exc)

    def _load_from_db(self, conn: sqlite3.Connection) -> None:
        try:
            rows = conn.execute(
                "SELECT cache_key, cookies, user_agent, created_at, ttl_seconds FROM challenge_cache"
            ).fetchall()
            loaded = 0
            for row in rows:
                key, cookies, ua, created, ttl = row
                entry = ChallengeCacheEntry(
                    cookies=cookies,
                    user_agent=ua,
                    created_at=created,
                    ttl_seconds=ttl,
                )
                if not entry.is_expired:
                    self._entries[key] = entry
                    loaded += 1
                else:
                    conn.execute("DELETE FROM challenge_cache WHERE cache_key = ?", (key,))
            if loaded:
                logger.debug("cf challenge cache loaded from disk: entries={}", loaded)
            conn.commit()
        except Exception as exc:
            logger.debug("cf challenge cache load from db failed: error={}", exc)

    def _save_to_db(self, key: str, entry: ChallengeCacheEntry) -> None:
        try:
            conn = sqlite3.connect(self._db_path, timeout=2)
            conn.execute(
                """INSERT OR REPLACE INTO challenge_cache
                   (cache_key, cookies, user_agent, created_at, ttl_seconds)
                   VALUES (?, ?, ?, ?, ?)""",
                (key, entry.cookies, entry.user_agent, entry.created_at, entry.ttl_seconds),
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

    def _remove_from_db(self, key: str) -> None:
        try:
            conn = sqlite3.connect(self._db_path, timeout=2)
            conn.execute("DELETE FROM challenge_cache WHERE cache_key = ?", (key,))
            conn.commit()
            conn.close()
        except Exception:
            pass

    def _clear_db(self) -> None:
        try:
            conn = sqlite3.connect(self._db_path, timeout=2)
            conn.execute("DELETE FROM challenge_cache")
            conn.commit()
            conn.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_challenge_cache: ChallengeCache | None = None


def get_challenge_cache() -> ChallengeCache:
    global _challenge_cache
    if _challenge_cache is None:
        _challenge_cache = ChallengeCache()
    return _challenge_cache


def reset_challenge_cache() -> None:
    global _challenge_cache
    _challenge_cache = None


__all__ = [
    "ChallengeCache", "ChallengeCacheEntry", "CooldownEntry",
    "get_challenge_cache", "reset_challenge_cache",
]
