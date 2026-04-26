import tempfile
import unittest
import os
from pathlib import Path
from unittest.mock import patch

from app.platform.storage.media_cache import LocalMediaCacheStore


class _StubConfig:
    def __init__(self, *, image_max_mb: int = 0, video_max_mb: int = 0) -> None:
        self._ints = {
            "cache.local.image_max_mb": image_max_mb,
            "cache.local.video_max_mb": video_max_mb,
        }

    def get_int(self, key: str, default: int = 0) -> int:
        return self._ints.get(key, default)


class MediaCacheLimitTests(unittest.TestCase):
    def test_save_image_prunes_oldest_file_when_type_limit_exceeded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_dir = root / "images"
            video_dir = root / "videos"
            image_dir.mkdir()
            video_dir.mkdir()

            store = LocalMediaCacheStore(
                config_provider=lambda: _StubConfig(image_max_mb=1)
            )
            with patch("app.platform.storage.media_cache.image_files_dir", return_value=image_dir):
                with patch("app.platform.storage.media_cache.video_files_dir", return_value=video_dir):
                    with patch(
                        "app.platform.storage.media_cache.local_media_cache_db_path",
                        return_value=root / "cache.db",
                    ):
                        with patch(
                            "app.platform.storage.media_cache.local_media_lock_path",
                            return_value=root / "cache.lock",
                        ):
                            store.save_image(b"a" * 800_000, "image/png", "old")
                            file_id = store.save_image(b"b" * 400_000, "image/png", "new")

            self.assertEqual(file_id, "new")
            self.assertTrue((image_dir / "new.png").exists())
            self.assertFalse((image_dir / "old.png").exists())

    def test_save_video_prunes_oldest_video_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_dir = root / "images"
            video_dir = root / "videos"
            image_dir.mkdir()
            video_dir.mkdir()

            image_path = image_dir / "keep.png"
            image_path.write_bytes(b"i" * 800_000)

            store = LocalMediaCacheStore(
                config_provider=lambda: _StubConfig(video_max_mb=1)
            )
            with patch("app.platform.storage.media_cache.image_files_dir", return_value=image_dir):
                with patch("app.platform.storage.media_cache.video_files_dir", return_value=video_dir):
                    with patch(
                        "app.platform.storage.media_cache.local_media_cache_db_path",
                        return_value=root / "cache.db",
                    ):
                        with patch(
                            "app.platform.storage.media_cache.local_media_lock_path",
                            return_value=root / "cache.lock",
                        ):
                            store.save_video(b"a" * 800_000, "old")
                            new_video = store.save_video(b"b" * 400_000, "new")

            self.assertTrue(new_video.exists())
            self.assertFalse((video_dir / "old.mp4").exists())
            self.assertTrue(image_path.exists())

    def test_stats_and_list_files_use_shared_cache_rules(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_dir = root / "images"
            video_dir = root / "videos"
            image_dir.mkdir()
            video_dir.mkdir()

            older = image_dir / "older.png"
            newer = image_dir / "newer.jpg"
            ignored = image_dir / "ignored.txt"
            older.write_bytes(b"a" * 128)
            newer.write_bytes(b"b" * 256)
            ignored.write_text("skip")
            os.utime(older, (100, 100))
            os.utime(newer, (200, 200))

            store = LocalMediaCacheStore(
                config_provider=lambda: _StubConfig(image_max_mb=1)
            )
            with patch("app.platform.storage.media_cache.image_files_dir", return_value=image_dir):
                with patch("app.platform.storage.media_cache.video_files_dir", return_value=video_dir):
                    stats = store.stats("image")
                    listing = store.list_files("image", page=1, page_size=10)

            self.assertEqual(stats["count"], 2)
            self.assertEqual(stats["size_bytes"], 384)
            self.assertEqual(stats["limit_mb"], 1)
            self.assertEqual([item["name"] for item in listing["items"]], ["newer.jpg", "older.png"])


if __name__ == "__main__":
    unittest.main()
