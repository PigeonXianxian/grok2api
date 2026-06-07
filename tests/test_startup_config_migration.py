import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.platform.startup import migration


class _Backend:
    def __init__(self, version: int = 0) -> None:
        self.version_calls = 0
        self.applied: list[dict] = []
        self._version = version

    async def version(self) -> int:
        self.version_calls += 1
        return self._version

    async def apply_patch(self, patch: dict) -> None:
        self.applied.append(patch)


class StartupConfigMigrationTests(unittest.TestCase):
    def test_account_storage_mysql_does_not_migrate_config_when_config_storage_unset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            defaults = root / "config.defaults.toml"
            user_config = root / "config.toml"
            defaults.write_text("[app]\napi_key = \"default\"\n", encoding="utf-8")
            user_config.write_text("[app]\napi_key = \"local\"\n", encoding="utf-8")
            backend = _Backend()

            with patch.dict(os.environ, {"ACCOUNT_STORAGE": "mysql"}, clear=True):
                with patch.object(migration, "_DEFAULTS_PATH", defaults):
                    with patch.object(migration, "_USER_CFG_PATH", user_config):
                        asyncio.run(migration._migrate_config(backend))

            self.assertEqual(backend.version_calls, 0)
            self.assertEqual(backend.applied, [])
            self.assertEqual(
                user_config.read_text(encoding="utf-8"),
                "[app]\napi_key = \"local\"\n",
            )

    def test_local_config_is_seeded_from_defaults_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            defaults = root / "config.defaults.toml"
            user_config = root / "config.toml"
            defaults.write_text("[app]\napi_key = \"default\"\n", encoding="utf-8")
            backend = _Backend()

            with patch.dict(os.environ, {"ACCOUNT_STORAGE": "postgresql"}, clear=True):
                with patch.object(migration, "_DEFAULTS_PATH", defaults):
                    with patch.object(migration, "_USER_CFG_PATH", user_config):
                        asyncio.run(migration._migrate_config(backend))

            self.assertTrue(user_config.exists())
            self.assertEqual(
                user_config.read_text(encoding="utf-8"),
                defaults.read_text(encoding="utf-8"),
            )
            self.assertEqual(backend.version_calls, 0)
            self.assertEqual(backend.applied, [])

    def test_explicit_remote_config_storage_still_migrates_local_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            defaults = root / "config.defaults.toml"
            user_config = root / "config.toml"
            defaults.write_text("[app]\napi_key = \"default\"\n", encoding="utf-8")
            user_config.write_text("[app]\napi_key = \"local\"\n", encoding="utf-8")
            backend = _Backend()

            with patch.dict(
                os.environ,
                {"ACCOUNT_STORAGE": "local", "CONFIG_STORAGE": "mysql"},
                clear=True,
            ):
                with patch.object(migration, "_DEFAULTS_PATH", defaults):
                    with patch.object(migration, "_USER_CFG_PATH", user_config):
                        asyncio.run(migration._migrate_config(backend))

        self.assertEqual(backend.version_calls, 1)
        self.assertEqual(backend.applied, [{"app": {"api_key": "local"}}])


if __name__ == "__main__":
    unittest.main()
