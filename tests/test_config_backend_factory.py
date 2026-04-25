import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.platform.config.backends.factory import (
    create_config_backend,
    get_config_backend_name,
)
from app.platform.config.backends.sql import SqlConfigBackend
from app.platform.config.backends.toml import TomlConfigBackend


class ConfigBackendFactoryTests(unittest.TestCase):
    def test_config_backend_defaults_to_local_when_account_storage_is_mysql(self) -> None:
        with patch.dict(os.environ, {"ACCOUNT_STORAGE": "mysql"}, clear=True):
            self.assertEqual(get_config_backend_name(), "local")

    def test_config_backend_defaults_to_local_when_account_storage_is_redis(self) -> None:
        with patch.dict(os.environ, {"ACCOUNT_STORAGE": "redis"}, clear=True):
            self.assertEqual(get_config_backend_name(), "local")

    def test_create_config_backend_uses_toml_for_explicit_local(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            with patch.dict(
                os.environ,
                {
                    "ACCOUNT_STORAGE": "mysql",
                    "CONFIG_STORAGE": "local",
                    "CONFIG_LOCAL_PATH": str(config_path),
                },
                clear=True,
            ):
                backend = create_config_backend()

        self.assertIsInstance(backend, TomlConfigBackend)

    def test_create_config_backend_uses_sql_only_when_config_storage_is_mysql(self) -> None:
        sentinel_engine = object()
        with patch.dict(
            os.environ,
            {
                "ACCOUNT_STORAGE": "local",
                "CONFIG_STORAGE": "mysql",
                "ACCOUNT_MYSQL_URL": "mysql://user:pass@example.com/db",
            },
            clear=True,
        ):
            with patch(
                "app.control.account.backends.sql.create_mysql_engine",
                return_value=sentinel_engine,
            ):
                backend = create_config_backend()

        self.assertIsInstance(backend, SqlConfigBackend)
        self.assertIs(backend._engine, sentinel_engine)
        self.assertEqual(backend._dialect, "mysql")


if __name__ == "__main__":
    unittest.main()
