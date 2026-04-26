import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.control.account.enums import FeedbackKind
from app.platform.errors import UpstreamError
from app.products.anthropic import messages as anthropic_messages
from app.products.anthropic import router as anthropic_router
from app.products.openai import chat, responses
from app.products.openai import router as openai_router


class _FakeConfig:
    def get(self, key: str, default=None):
        return default

    def get_bool(self, key: str, default: bool = False) -> bool:
        return default

    def get_float(self, key: str, default: float = 0.0) -> float:
        return default

    def get_str(self, key: str, default: str = "") -> str:
        return default


class _FakeDirectory:
    def __init__(self) -> None:
        self.feedbacks: list[tuple[str, FeedbackKind, int]] = []

    async def release(self, acct) -> None:
        return None

    async def feedback(
        self,
        token: str,
        kind: FeedbackKind,
        selected_mode_id: int,
        now_s_val=None,
    ) -> None:
        self.feedbacks.append((token, kind, selected_mode_id))


class _FakeAdapter:
    def __init__(
        self,
        *,
        text: str = "",
        images: list[tuple[str, str]] | None = None,
        references: str = "",
        sources: list[dict] | None = None,
        thinking: str = "",
    ) -> None:
        self.text_buf = [text] if text else []
        self.thinking_buf = [thinking] if thinking else []
        self.image_urls = images or []
        self._references = references
        self._sources = sources

    def references_suffix(self) -> str:
        return self._references

    def search_sources_list(self):
        return self._sources


async def _empty_stream(**kwargs):
    yield "data: [DONE]"


async def _noop_async(*args, **kwargs):
    return None


async def _empty_generator():
    if False:
        yield ""


class EmptyResponseHandlingTests(unittest.TestCase):
    def test_visible_output_helper_ignores_thinking_only(self) -> None:
        self.assertFalse(
            chat._adapter_has_visible_output(_FakeAdapter(thinking="reasoning only"))
        )

    def test_visible_output_helper_accepts_text_tool_images_and_sources(self) -> None:
        self.assertTrue(chat._adapter_has_visible_output(_FakeAdapter(text="hello")))
        self.assertTrue(
            chat._adapter_has_visible_output(
                _FakeAdapter(images=[("https://example.com/a.png", "img-1")])
            )
        )
        self.assertTrue(
            chat._adapter_has_visible_output(
                _FakeAdapter(sources=[{"url": "https://example.com", "title": "x"}])
            )
        )
        self.assertTrue(
            chat._adapter_has_visible_output(_FakeAdapter(), has_tool_calls=True)
        )

    def test_empty_response_error_is_retryable_429(self) -> None:
        exc = chat._empty_upstream_response_error()

        self.assertEqual(exc.status, 429)
        self.assertEqual(exc.details["body"], chat.EMPTY_UPSTREAM_BODY)

    def test_stream_prime_turns_no_first_chunk_into_429(self) -> None:
        async def _run() -> None:
            await openai_router._prime_sse(_empty_generator())

        with self.assertRaises(UpstreamError) as ctx:
            asyncio.run(_run())

        self.assertEqual(ctx.exception.status, 429)

    def test_anthropic_stream_prime_turns_no_first_chunk_into_429(self) -> None:
        async def _run() -> None:
            await anthropic_router._prime_sse(_empty_generator())

        with self.assertRaises(UpstreamError) as ctx:
            asyncio.run(_run())

        self.assertEqual(ctx.exception.status, 429)

    def test_chat_non_stream_empty_response_raises_429_feedback(self) -> None:
        directory = _FakeDirectory()
        exc = self._run_chat_non_stream_empty(directory)

        self.assertEqual(exc.status, 429)
        self.assertEqual(exc.details["body"], chat.EMPTY_UPSTREAM_BODY)
        self.assertEqual(directory.feedbacks[-1][1], FeedbackKind.RATE_LIMITED)

    def test_chat_stream_empty_response_raises_429_before_first_chunk(self) -> None:
        directory = _FakeDirectory()

        async def _run() -> None:
            stream = await chat.completions(
                model="grok-test",
                messages=[{"role": "user", "content": "hello"}],
                stream=True,
            )
            await openai_router._prime_sse(stream)

        with self.assertRaises(UpstreamError) as ctx:
            with self._patch_common(chat, directory):
                asyncio.run(_run())

        self.assertEqual(ctx.exception.status, 429)
        self.assertEqual(directory.feedbacks[-1][1], FeedbackKind.RATE_LIMITED)

    def test_responses_non_stream_empty_response_raises_429_feedback(self) -> None:
        directory = _FakeDirectory()
        exc = self._run_responses_non_stream_empty(directory)

        self.assertEqual(exc.status, 429)
        self.assertEqual(exc.details["body"], chat.EMPTY_UPSTREAM_BODY)
        self.assertEqual(directory.feedbacks[-1][1], FeedbackKind.RATE_LIMITED)

    def test_anthropic_non_stream_empty_response_raises_429_feedback(self) -> None:
        directory = _FakeDirectory()
        exc = self._run_anthropic_non_stream_empty(directory)

        self.assertEqual(exc.status, 429)
        self.assertEqual(exc.details["body"], chat.EMPTY_UPSTREAM_BODY)
        self.assertEqual(directory.feedbacks[-1][1], FeedbackKind.RATE_LIMITED)

    def _run_chat_non_stream_empty(self, directory: _FakeDirectory) -> UpstreamError:
        async def _run() -> None:
            await chat.completions(
                model="grok-test",
                messages=[{"role": "user", "content": "hello"}],
                stream=False,
            )

        with self.assertRaises(UpstreamError) as ctx:
            with self._patch_common(chat, directory):
                asyncio.run(_run())
        return ctx.exception

    def _run_responses_non_stream_empty(self, directory: _FakeDirectory) -> UpstreamError:
        async def _run() -> None:
            await responses.create(
                model="grok-test",
                input_val="hello",
                instructions=None,
                stream=False,
                emit_think=True,
                temperature=0.8,
                top_p=0.95,
            )

        with self.assertRaises(UpstreamError) as ctx:
            with self._patch_common(responses, directory):
                asyncio.run(_run())
        return ctx.exception

    def _run_anthropic_non_stream_empty(
        self, directory: _FakeDirectory
    ) -> UpstreamError:
        async def _run() -> None:
            await anthropic_messages.create(
                model="grok-test",
                messages=[{"role": "user", "content": "hello"}],
                stream=False,
                emit_think=True,
                temperature=0.8,
                top_p=0.95,
            )

        with self.assertRaises(UpstreamError) as ctx:
            with self._patch_common(anthropic_messages, directory):
                asyncio.run(_run())
        return ctx.exception

    def _patch_common(self, module, directory: _FakeDirectory):
        import contextlib

        @contextlib.contextmanager
        def _ctx():
            with patch("app.dataplane.account._directory", directory):
                with patch.object(module, "get_config", return_value=_FakeConfig()):
                    with patch.object(
                        module,
                        "resolve_model",
                        return_value=SimpleNamespace(mode_id=0),
                    ):
                        with patch.object(module, "selection_max_retries", return_value=0):
                            with patch.object(
                                module,
                                "reserve_account",
                                return_value=(SimpleNamespace(token="tok-test"), 0),
                            ):
                                with patch.object(
                                    module, "_stream_chat", side_effect=_empty_stream
                                ):
                                    with patch.object(module, "_fail_sync", _noop_async):
                                        with patch.object(module, "_quota_sync", _noop_async):
                                            yield

        return _ctx()


if __name__ == "__main__":
    unittest.main()
