import asyncio
import unittest

from Tuner.BOTuner.LGBO import LGBOSampler


class _FakeBatchLLM:
    def __init__(self):
        self.current_slot = 0
        self.slot_uses = []
        self.fail_once = {0}

    def batch_slot_count(self) -> int:
        return 2

    async def acquire_batch_slot(self) -> int:
        slot = self.current_slot
        self.current_slot = (self.current_slot + 1) % 2
        self.slot_uses.append(("acquire", slot))
        return slot

    def release_batch_slot(self, idx: int) -> None:
        self.slot_uses.append(("release", idx))

    async def acompletion_text_with_slot(self, *, slot_idx: int, messages, stream=False, **kwargs):
        if slot_idx in self.fail_once:
            self.fail_once.remove(slot_idx)
            raise RuntimeError(f"slot {slot_idx} batch failed")
        return f"ok-{slot_idx}-{messages[-1]['content']}"


class _WaveBatchLLM:
    def __init__(self):
        self.current_slot = 0
        self.active = 0
        self.max_active = 0

    def batch_slot_count(self) -> int:
        return 2

    async def acquire_batch_slot(self) -> int:
        slot = self.current_slot
        self.current_slot = (self.current_slot + 1) % 2
        return slot

    def release_batch_slot(self, idx: int) -> None:
        return None

    async def acompletion_text_with_slot(self, *, slot_idx: int, messages, stream=False, **kwargs):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        return f"slot-{slot_idx}-{messages[-1]['content']}"


class _AlwaysFailBatchLLM:
    def __init__(self):
        self.current_slot = 0
        self.slot_uses = []

    def batch_slot_count(self) -> int:
        return 7

    async def acquire_batch_slot(self) -> int:
        slot = self.current_slot
        self.current_slot = (self.current_slot + 1) % 7
        self.slot_uses.append(("acquire", slot))
        return slot

    def release_batch_slot(self, idx: int) -> None:
        self.slot_uses.append(("release", idx))

    async def acompletion_text_with_slot(self, *, slot_idx: int, messages, stream=False, **kwargs):
        raise RuntimeError(f"slot {slot_idx} failed")


class LGBOBatchFallbackTest(unittest.TestCase):
    def test_single_request_immediate_failover_to_next_key(self):
        sampler = object.__new__(LGBOSampler)
        llm = _FakeBatchLLM()

        result = asyncio.run(
            sampler._acall_prompt_with_immediate_failover(
                llm,
                "p1",
            )
        )

        self.assertEqual(result, "ok-1-p1")
        self.assertEqual(llm.slot_uses[:4], [("acquire", 0), ("release", 0), ("acquire", 1), ("release", 1)])

    def test_batch_waves_run_in_parallel(self):
        sampler = object.__new__(LGBOSampler)
        llm = _WaveBatchLLM()
        sampler._get_llm = lambda: llm

        results = asyncio.run(
            sampler._acall_llm_batch(
                [f"p{i}" for i in range(45)],
            )
        )

        self.assertEqual(len(results), 45)
        self.assertEqual(llm.max_active, 28)

    def test_prompt_gives_up_after_three_failed_keys(self):
        sampler = object.__new__(LGBOSampler)
        llm = _AlwaysFailBatchLLM()

        result = asyncio.run(
            sampler._acall_prompt_with_immediate_failover(
                llm,
                "p1",
            )
        )

        self.assertIsNone(result)
        acquires = [item for item in llm.slot_uses if item[0] == "acquire"]
        self.assertEqual(len(acquires), 3)


if __name__ == "__main__":
    unittest.main()
