import asyncio
import tempfile
import unittest
from pathlib import Path
from typing import Any

from app.platform.config.snapshot import ConfigSnapshot


class _Backend:
    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data

    async def load(self) -> dict[str, Any]:
        return self.data

    async def apply_patch(self, patch: dict[str, Any]) -> None:
        self.data.update(patch)

    async def version(self) -> object:
        return 1


class LegacyCacheConfigTests(unittest.TestCase):
    def test_legacy_storage_cache_limits_map_to_cache_local(self) -> None:
        cfg = asyncio.run(self._load({
            "storage": {
                "image_max_mb": 12,
                "video_max_mb": 34,
            },
        }))

        self.assertEqual(cfg.get_int("cache.local.image_max_mb"), 12)
        self.assertEqual(cfg.get_int("cache.local.video_max_mb"), 34)

    def test_cache_local_limits_win_over_legacy_storage(self) -> None:
        cfg = asyncio.run(self._load({
            "storage": {
                "image_max_mb": 12,
                "video_max_mb": 34,
            },
            "cache": {
                "local": {
                    "image_max_mb": 56,
                    "video_max_mb": 78,
                },
            },
        }))

        self.assertEqual(cfg.get_int("cache.local.image_max_mb"), 56)
        self.assertEqual(cfg.get_int("cache.local.video_max_mb"), 78)

    async def _load(self, overrides: dict[str, Any]) -> ConfigSnapshot:
        with tempfile.TemporaryDirectory() as tmpdir:
            defaults = Path(tmpdir) / "config.defaults.toml"
            defaults.write_text(
                "[cache.local]\nimage_max_mb = 0\nvideo_max_mb = 0\n",
                encoding="utf-8",
            )
            cfg = ConfigSnapshot(backend=_Backend(overrides))
            await cfg.load(defaults)
            return cfg


if __name__ == "__main__":
    unittest.main()
