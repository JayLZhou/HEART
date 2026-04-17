import asyncio
import unittest
from unittest.mock import patch

from Config.LLMConfig import LLMConfig
from Provider.OpenaiApi import OpenAILLM


class _FakeCompletions:
    def __init__(self, api_key: str, calls: list[str]):
        self.api_key = api_key
        self.calls = calls

    async def create(self, **kwargs):
        self.calls.append(self.api_key)
        msg = type("Msg", (), {"content": f"reply-from-{self.api_key}"})()
        choice = type("Choice", (), {"message": msg})()
        return type("Rsp", (), {"choices": [choice], "usage": None})()


class _FakeClient:
    def __init__(self, api_key: str, calls: list[str]):
        self.chat = type("Chat", (), {"completions": _FakeCompletions(api_key, calls)})()


class OpenAIKeyPoolTest(unittest.TestCase):
    def test_round_robin_across_key_pool(self):
        calls: list[str] = []

        def _fake_openai(**kwargs):
            return _FakeClient(kwargs["api_key"], calls)

        with patch("Provider.OpenaiApi.AsyncOpenAI", side_effect=_fake_openai), patch.object(
            OpenAILLM, "_load_api_key_pool", return_value=["k1", "k2", "k3"]
        ):
            llm = OpenAILLM(LLMConfig(api_key="fallback", model="demo-model", max_concurrent=9))
            for _ in range(5):
                asyncio.run(
                    llm._achat_completion(
                        [{"role": "user", "content": "hello"}],
                    )
                )

        self.assertEqual(calls, ["k1", "k2", "k3", "k1", "k2"])

    def test_disabled_slot_is_skipped(self):
        calls: list[str] = []

        def _fake_openai(**kwargs):
            return _FakeClient(kwargs["api_key"], calls)

        with patch("Provider.OpenaiApi.AsyncOpenAI", side_effect=_fake_openai), patch.object(
            OpenAILLM, "_load_api_key_pool", return_value=["k1", "k2", "k3"]
        ):
            llm = OpenAILLM(LLMConfig(api_key="fallback", model="demo-model", max_concurrent=9))
            llm._disable_client_slot(1, Exception("insufficient balance"))
            for _ in range(4):
                asyncio.run(
                    llm._achat_completion(
                        [{"role": "user", "content": "hello"}],
                    )
                )

        self.assertEqual(calls, ["k1", "k3", "k1", "k3"])

    def test_batch_slots_work_across_multiple_event_loops(self):
        calls: list[str] = []

        def _fake_openai(**kwargs):
            return _FakeClient(kwargs["api_key"], calls)

        async def _use_batch_slot(llm: OpenAILLM):
            slot = await llm.acquire_batch_slot()
            try:
                return await llm.acompletion_text_with_slot(
                    slot_idx=slot,
                    messages=[{"role": "user", "content": "hello"}],
                )
            finally:
                llm.release_batch_slot(slot)

        with patch("Provider.OpenaiApi.AsyncOpenAI", side_effect=_fake_openai), patch.object(
            OpenAILLM, "_load_api_key_pool", return_value=["k1", "k2"]
        ):
            llm = OpenAILLM(LLMConfig(api_key="fallback", model="demo-model", max_concurrent=4))
            first = asyncio.run(_use_batch_slot(llm))
            second = asyncio.run(_use_batch_slot(llm))

        self.assertIn(first, {"reply-from-k1", "reply-from-k2"})
        self.assertIn(second, {"reply-from-k1", "reply-from-k2"})
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
