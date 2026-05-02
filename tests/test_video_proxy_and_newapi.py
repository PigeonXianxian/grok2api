import asyncio
import hashlib
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.control.proxy import ProxyDirectory
from app.dataplane.reverse.transport import assets
from app.products.openai import video


class _ProxyConfig:
    def __init__(self, *, resource_proxy_url: str = "") -> None:
        self.resource_proxy_url = resource_proxy_url

    def get_str(self, key: str, default: str = "") -> str:
        values = {
            "proxy.egress.mode": "single_proxy",
            "proxy.clearance.mode": "none",
            "proxy.egress.proxy_url": "http://proxy.example:8080",
            "proxy.egress.resource_proxy_url": self.resource_proxy_url,
        }
        return values.get(key, default)

    def get_list(self, key: str, default=None):
        return default or []

    def get_int(self, key: str, default: int = 0) -> int:
        return default


class _AppConfig:
    def __init__(self, *, app_url: str = "") -> None:
        self.app_url = app_url

    def get_str(self, key: str, default: str = "") -> str:
        if key == "app.app_url":
            return self.app_url
        return default


class _AssetConfig:
    def get_float(self, key: str, default: float = 0.0) -> float:
        return default


class _VideoConfig:
    def get_float(self, key: str, default: float = 0.0) -> float:
        return default

    def get(self, key: str, default=None):
        return default


class _FakeProxyRuntime:
    def __init__(self) -> None:
        self.acquire_kwargs = []

    async def acquire(self, **kwargs):
        self.acquire_kwargs.append(kwargs)
        return SimpleNamespace(proxy_url=None)

    async def feedback(self, lease, result) -> None:
        return None


class _FakeAccountDirectory:
    def __init__(self) -> None:
        self.reserve_calls = []
        self.feedbacks = []

    async def reserve(self, **kwargs):
        self.reserve_calls.append(kwargs)
        return SimpleNamespace(token="tok-video")

    async def release(self, acct) -> None:
        return None

    async def feedback(self, token, kind, mode_id) -> None:
        self.feedbacks.append((token, kind, mode_id))


async def _empty_bytes_stream(*args, **kwargs):
    if False:
        yield b""


async def _noop_async(*args, **kwargs):
    return None


class VideoProxyAndNewApiTests(unittest.TestCase):
    def test_resource_download_is_direct_without_resource_proxy(self) -> None:
        async def _run():
            directory = ProxyDirectory()
            with patch("app.control.proxy.get_config", return_value=_ProxyConfig()):
                await directory.load()
                return await directory.acquire(resource=True)

        lease = asyncio.run(_run())

        self.assertIsNone(lease.proxy_url)

    def test_resource_download_uses_explicit_resource_proxy(self) -> None:
        async def _run():
            directory = ProxyDirectory()
            cfg = _ProxyConfig(resource_proxy_url="http://res-proxy.example:8080")
            with patch("app.control.proxy.get_config", return_value=cfg):
                await directory.load()
                return await directory.acquire(resource=True)

        lease = asyncio.run(_run())

        self.assertEqual(lease.proxy_url, "http://res-proxy.example:8080")

    def test_download_asset_acquires_resource_lease(self) -> None:
        async def _run():
            proxy = _FakeProxyRuntime()
            with patch.object(assets, "get_proxy_runtime", return_value=proxy):
                with patch.object(assets, "get_config", return_value=_AssetConfig()):
                    with patch.object(
                        assets, "get_bytes_stream", return_value=_empty_bytes_stream()
                    ):
                        await assets.download_asset(
                            "tok-test",
                            "https://assets.grok.com/users/u/file.mp4",
                        )
            return proxy.acquire_kwargs

        calls = asyncio.run(_run())

        self.assertEqual(calls[-1]["resource"], True)

    def test_completed_video_job_returns_newapi_metadata_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "video.mp4"
            path.write_bytes(b"fake-video")
            job = video._VideoJob(
                id="video_test",
                model="grok-imagine-video",
                prompt="hello",
                seconds="6",
                size="720x1280",
                quality="standard",
                created_at=int(time.time()),
                status="completed",
                progress=100,
                completed_at=int(time.time()),
                content_path=str(path),
                content_file_id="08b8238c88e9bbe8423c692cbd04ec52",
            )

            with patch.object(video, "get_config", return_value=_AppConfig(app_url="https://api.example.com")):
                payload = job.to_dict()

        self.assertEqual(
            payload["metadata"]["url"],
            "https://api.example.com/v1/files/video?id=08b8238c88e9bbe8423c692cbd04ec52",
        )

    def test_video_job_saves_with_sync_file_id_and_metadata_url(self) -> None:
        async def _run(tmpdir: str):
            artifact = video._VideoArtifact(
                video_url="https://assets.grok.com/users/u/generated.mp4",
                video_post_id="post_1",
                asset_id="asset_1",
                thumbnail_url="https://assets.grok.com/thumb.jpg",
            )
            raw = b"fake-video"
            saved_ids: list[str] = []
            job = video._VideoJob(
                id="video_async",
                model="grok-imagine-video",
                prompt="hello",
                seconds="6",
                size="720x1280",
                quality="standard",
                created_at=int(time.time()),
            )

            async def _fake_run_video_with_account(*, model, runner):
                return artifact, raw

            def _fake_save_video_bytes(raw_bytes: bytes, file_id: str) -> Path:
                saved_ids.append(file_id)
                path = Path(tmpdir) / f"{file_id}.mp4"
                path.write_bytes(raw_bytes)
                return path

            with patch.object(
                video, "_run_video_with_account", _fake_run_video_with_account
            ):
                with patch.object(video, "_save_video_bytes", _fake_save_video_bytes):
                    await video._run_video_job(
                        job,
                        size="720x1280",
                        resolution_name=None,
                        prompt="hello",
                        seconds=6,
                        preset=None,
                    )

            with patch.object(
                video,
                "get_config",
                return_value=_AppConfig(app_url="https://api.example.com"),
            ):
                payload = job.to_dict()

            return job, payload, saved_ids

        with tempfile.TemporaryDirectory() as tmpdir:
            job, payload, saved_ids = asyncio.run(_run(tmpdir))

        expected_file_id = hashlib.sha1(
            "https://assets.grok.com/users/u/generated.mp4".encode("utf-8")
        ).hexdigest()[:32]
        self.assertEqual(saved_ids, [expected_file_id])
        self.assertEqual(job.content_file_id, expected_file_id)
        self.assertTrue(job.content_path.endswith(f"{expected_file_id}.mp4"))
        self.assertEqual(
            payload["metadata"]["url"],
            f"https://api.example.com/v1/files/video?id={expected_file_id}",
        )

    def test_video_pool_candidates_exclude_basic(self) -> None:
        spec = SimpleNamespace(pool_candidates=lambda: (0, 1, 2))

        self.assertEqual(video._video_pool_candidates(spec), (1, 2))

    def test_video_account_runner_reserves_only_super_or_heavy(self) -> None:
        async def _run():
            directory = _FakeAccountDirectory()
            spec = SimpleNamespace(
                is_video=lambda: True,
                mode_id=0,
                pool_candidates=lambda: (0, 1, 2),
            )

            async def _runner(token: str, timeout_s: float) -> str:
                return token

            with patch("app.dataplane.account._directory", directory):
                with patch.object(video, "get_config", return_value=_VideoConfig()):
                    with patch.object(video, "resolve_model", return_value=spec):
                        with patch.object(video, "selection_max_retries", return_value=0):
                            with patch.object(video, "_quota_sync", _noop_async):
                                result = await video._run_video_with_account(
                                    model="grok-imagine-video",
                                    runner=_runner,
                                )
            return result, directory.reserve_calls

        result, calls = asyncio.run(_run())

        self.assertEqual(result, "tok-video")
        self.assertEqual(calls[0]["pool_candidates"], (1, 2))


if __name__ == "__main__":
    unittest.main()
