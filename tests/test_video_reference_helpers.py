import asyncio
import unittest

from app.platform.errors import ValidationError
from app.products.openai import router, video


def _message_with_references(count: int) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "生成一个参考图视频"},
                *[
                    {
                        "type": "image_url",
                        "image_url": {"url": f"https://example.com/ref-{idx}.png"},
                    }
                    for idx in range(count)
                ],
            ],
        }
    ]


class VideoReferenceHelperTests(unittest.TestCase):
    def test_chat_video_prompt_allows_five_references(self) -> None:
        prompt, refs = video._extract_video_prompt_and_reference(
            _message_with_references(5)
        )

        self.assertEqual(prompt, "生成一个参考图视频")
        self.assertEqual(
            refs,
            [
                {"image_url": f"https://example.com/ref-{idx}.png"}
                for idx in range(5)
            ],
        )

    def test_chat_video_prompt_rejects_more_than_five_references(self) -> None:
        with self.assertRaises(ValidationError):
            video._extract_video_prompt_and_reference(_message_with_references(6))

    def test_prepare_video_references_rejects_more_than_five_references(self) -> None:
        refs = [{"image_url": f"https://example.com/ref-{idx}.png"} for idx in range(6)]

        async def _run() -> None:
            await video._prepare_video_references("token", refs)

        with self.assertRaises(ValidationError):
            asyncio.run(_run())

    def test_videos_create_rejects_more_than_five_multipart_references(self) -> None:
        async def _run() -> None:
            await router.videos_create(
                model="grok-imagine-video",
                prompt="生成一个参考图视频",
                input_reference=[object() for _ in range(6)],  # type: ignore[list-item]
            )

        with self.assertRaises(ValidationError):
            asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
