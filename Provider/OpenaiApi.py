from __future__ import annotations

import numpy as np
import yaml
from typing import Optional, Union
import asyncio
import threading
from pathlib import Path

from openai import APIConnectionError, AsyncOpenAI, AsyncStream, PermissionDeniedError
from openai._base_client import AsyncHttpxClientWrapper
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from tenacity import (
    after_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from Config.LLMConfig import LLMConfig, LLMType
from Common.Constants import USE_CONFIG_TIMEOUT
from Common.Logger import log_llm_stream, logger
from Common.Constants import PROJECT_ROOT
from Provider.BaseLLM import BaseLLM
from Provider.LLMProviderRegister import register_provider
from Common.Utils import  log_and_reraise,prase_json_from_response
from Common.CostManager import CostManager
from Utils.Exceptions import handle_exception
from Utils.TokenCounter import (
    count_input_tokens,
    count_output_tokens,
    get_max_completion_tokens,
)


@register_provider(
    [
        LLMType.OPENAI,
        LLMType.FIREWORKS,
        LLMType.OPEN_LLM,
    ]
)
class OpenAILLM(BaseLLM):
    """Check https://platform.openai.com/examples for examples"""
    PER_KEY_CONCURRENCY = 4

    def __init__(self, config: LLMConfig):
        self.config = config
        self._slot_lock = threading.Lock()
        self._slot_cursor = 0
        self._request_semaphores: dict[int, asyncio.Semaphore] = {}
        self._slot_semaphores: dict[int, list[asyncio.Semaphore]] = {}
        self._loop_state_lock = threading.Lock()
        self._client_slots = []
        self._init_client()
        self.auto_max_tokens = False
        self.cost_manager: Optional[CostManager] = None
    def _init_client(self):
        """https://github.com/openai/openai-python#async-usage"""
        self.model = self.config.model  # Used in _calc_usage & _cons_kwargs
        self.pricing_plan = self.config.pricing_plan or self.model
        self._client_slots = self._build_client_slots()
        self.aclient = self._client_slots[0]["client"]

    def _make_client_kwargs(self, api_key: str | None = None) -> dict:
        kwargs = {"api_key": api_key or self.config.api_key, "base_url": self.config.base_url}

        # to use proxy, openai v1 needs http_client
        if proxy_params := self._get_proxy_params():
            kwargs["http_client"] = AsyncHttpxClientWrapper(**proxy_params)

        return kwargs

    def _build_client_slots(self) -> list[dict]:
        api_keys = self._load_api_key_pool() or [self.config.api_key]
        per_key_concurrency = self.PER_KEY_CONCURRENCY
        slots = []
        for api_key in api_keys:
            slots.append(
                {
                    "client": AsyncOpenAI(**self._make_client_kwargs(api_key=api_key)),
                    "per_key_concurrency": per_key_concurrency,
                    "disabled": False,
                }
            )
        return slots

    def _get_request_semaphore(self):
        loop = asyncio.get_running_loop()
        loop_id = id(loop)
        with self._loop_state_lock:
            semaphore = self._request_semaphores.get(loop_id)
            if semaphore is None:
                semaphore = asyncio.Semaphore(self.config.max_concurrent)
                self._request_semaphores[loop_id] = semaphore
            return semaphore

    def _get_slot_semaphores(self) -> list[asyncio.Semaphore]:
        loop = asyncio.get_running_loop()
        loop_id = id(loop)
        with self._loop_state_lock:
            semaphores = self._slot_semaphores.get(loop_id)
            if semaphores is None:
                semaphores = [
                    asyncio.Semaphore(int(slot["per_key_concurrency"]))
                    for slot in self._client_slots
                ]
                self._slot_semaphores[loop_id] = semaphores
            return semaphores

    def _load_api_key_pool(self) -> list[str]:
        path = PROJECT_ROOT / "api_keys.yaml"
        if not path.exists():
            return []
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            logger.warning(f"Failed to load api key pool from {path}: {exc}")
            return []

        raw_items = payload.get("api_keys", []) if isinstance(payload, dict) else []
        keys: list[str] = []
        for item in raw_items:
            if isinstance(item, str) and item:
                keys.append(item)
            elif isinstance(item, dict):
                for cand in ("api_key", "key", "value", "token"):
                    value = item.get(cand)
                    if value:
                        keys.append(str(value))
                        break

        deduped: list[str] = []
        seen = set()
        for key in keys:
            if key not in seen:
                seen.add(key)
                deduped.append(key)
        return deduped

    async def _acquire_client_slot(self) -> tuple[int, dict]:
        with self._slot_lock:
            if not self._client_slots:
                raise RuntimeError("No OpenAI-compatible API clients are configured.")
            for offset in range(len(self._client_slots)):
                idx = (self._slot_cursor + offset) % len(self._client_slots)
                slot = self._client_slots[idx]
                if not slot["disabled"]:
                    self._slot_cursor = (idx + 1) % len(self._client_slots)
                    break
            else:
                raise RuntimeError("All configured OpenAI-compatible API keys are disabled.")
        await self._get_slot_semaphores()[idx].acquire()
        return idx, slot

    async def acquire_batch_slot(self) -> int:
        idx, _ = await self._acquire_client_slot()
        return idx

    def release_batch_slot(self, idx: int) -> None:
        self._get_slot_semaphores()[idx].release()

    def batch_slot_count(self) -> int:
        return sum(0 if slot["disabled"] else 1 for slot in self._client_slots)

    def _disable_client_slot(self, idx: int, reason: Exception) -> None:
        slot = self._client_slots[idx]
        if slot["disabled"]:
            return
        text = str(reason).lower()
        if any(token in text for token in ("insufficient", "invalid", "unauthorized", "incorrect api key", "balance")):
            slot["disabled"] = True
            logger.warning("Disabled one OpenAI-compatible API key due to a permanent auth/balance failure.")

    def _get_proxy_params(self) -> dict:
        params = {}
        if self.config.proxy:
            params = {"proxies": self.config.proxy}
            if self.config.base_url:
                params["base_url"] = self.config.base_url

        return params

    async def _achat_completion_stream(
        self,
        messages: list[dict],
        timeout=USE_CONFIG_TIMEOUT,
        max_tokens = None,
        slot_idx: int | None = None,
    ) -> str:
        owns_slot = slot_idx is None
        if owns_slot:
            idx, slot = await self._acquire_client_slot()
        else:
            idx = slot_idx
            slot = self._client_slots[idx]
        try:
            response: AsyncStream[ChatCompletionChunk] = await slot["client"].chat.completions.create(
                **self._cons_kwargs(messages, timeout=self.get_timeout(timeout), max_tokens = max_tokens), stream=True
            )
            usage = None
            collected_messages = []
            has_finished = False
            async for chunk in response:
                chunk_message = chunk.choices[0].delta.content or "" if chunk.choices else ""
                finish_reason = (
                    chunk.choices[0].finish_reason if chunk.choices and hasattr(chunk.choices[0], "finish_reason") else None
                )
                log_llm_stream(chunk_message)
                collected_messages.append(chunk_message)
                chunk_has_usage = hasattr(chunk, "usage") and chunk.usage
                if has_finished:
                    if chunk_has_usage:
                        usage = CompletionUsage(**chunk.usage) if isinstance(chunk.usage, dict) else chunk.usage
                if finish_reason:
                    if chunk_has_usage:
                        if isinstance(chunk.usage, CompletionUsage):
                            usage = chunk.usage
                        else:
                            usage = CompletionUsage(**chunk.usage)
                    elif hasattr(chunk.choices[0], "usage"):
                        usage = CompletionUsage(**chunk.choices[0].usage)
                    has_finished = True

            log_llm_stream("\n")
            full_reply_content = "".join(collected_messages)
            if not usage:
                usage = self._calc_usage(messages, full_reply_content)

            self._update_costs(usage)
            return full_reply_content
        except PermissionDeniedError as exc:
            self._disable_client_slot(idx, exc)
            raise
        finally:
            if owns_slot:
                self._get_slot_semaphores()[idx].release()

    def _cons_kwargs(self, messages: list[dict], timeout=USE_CONFIG_TIMEOUT, max_tokens = None, **extra_kwargs) -> dict:
        kwargs = {
            "messages": messages,
            "max_tokens": self._get_max_tokens(messages),
            # "n": 1,  # Some services do not provide this parameter, such as mistral
            "stop": ["[/INST]", "<<SYS>>"] ,  # default it's None and gpt4-v can't have this one
            "temperature": self.config.temperature,
            "model": self.model,
            "timeout": self.get_timeout(timeout),
        }
        if "o1-" in self.model:
            # compatible to openai o1-series
            kwargs["temperature"] = 1
            kwargs.pop("max_tokens")
        if max_tokens != None:
            kwargs["max_tokens"] = max_tokens
        if extra_kwargs:
            kwargs.update(extra_kwargs)
        return kwargs

    async def _achat_completion(
        self,
        messages: list[dict],
        timeout=USE_CONFIG_TIMEOUT,
        max_tokens = None,
        slot_idx: int | None = None,
    ) -> ChatCompletion:
        owns_slot = slot_idx is None
        if owns_slot:
            idx, slot = await self._acquire_client_slot()
        else:
            idx = slot_idx
            slot = self._client_slots[idx]
        try:
            kwargs = self._cons_kwargs(messages, timeout=self.get_timeout(timeout), max_tokens=max_tokens)
            rsp: ChatCompletion = await slot["client"].chat.completions.create(**kwargs)
            self._update_costs(rsp.usage)
            return rsp
        except PermissionDeniedError as exc:
            self._disable_client_slot(idx, exc)
            raise
        finally:
            if owns_slot:
                self._get_slot_semaphores()[idx].release()

    async def acompletion(self, messages: list[dict], timeout=USE_CONFIG_TIMEOUT) -> ChatCompletion:
        return await self._achat_completion(messages, timeout=self.get_timeout(timeout))

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        after=after_log(logger, logger.level("WARNING").name),
        retry=retry_if_exception_type(Exception),
        retry_error_callback=log_and_reraise,
    )
    async def acompletion_text(self, messages: list[dict], stream=False, timeout=USE_CONFIG_TIMEOUT, max_tokens = None, format = "text") -> str:
        """when streaming, print each token in place."""
        if stream:
            return await self._achat_completion_stream(messages, timeout=timeout, max_tokens = max_tokens)

        rsp = await self._achat_completion(messages, timeout=self.get_timeout(timeout), max_tokens = max_tokens)

        rsp_text = self.get_choice_text(rsp)
        if format == "json":
            return prase_json_from_response(rsp_text)
        return rsp_text

    async def acompletion_text_with_slot(
        self,
        *,
        slot_idx: int,
        messages: list[dict],
        stream=False,
        timeout=USE_CONFIG_TIMEOUT,
        max_tokens=None,
        format="text",
    ) -> str:
        if stream:
            return await self._achat_completion_stream(
                messages,
                timeout=timeout,
                max_tokens=max_tokens,
                slot_idx=slot_idx,
            )

        rsp = await self._achat_completion(
            messages,
            timeout=self.get_timeout(timeout),
            max_tokens=max_tokens,
            slot_idx=slot_idx,
        )
        rsp_text = self.get_choice_text(rsp)
        if format == "json":
            return prase_json_from_response(rsp_text)
        return rsp_text


 

    def get_choice_text(self, rsp: ChatCompletion) -> str:
        """Required to provide the first text of choice"""
        return rsp.choices[0].message.content if rsp.choices else ""

    def _calc_usage(self, messages: list[dict], rsp: str) -> CompletionUsage:
        usage = CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        if not self.config.calc_usage:
            return usage

        try:
            usage.prompt_tokens = count_input_tokens(messages, self.pricing_plan)
            usage.completion_tokens = count_output_tokens(rsp, self.pricing_plan)
        except Exception as e:
            logger.warning(f"usage calculation failed: {e}")

        return usage

    def _get_max_tokens(self, messages: list[dict]):
        if not self.auto_max_tokens:
            return self.config.max_token
        # FIXME
        # https://community.openai.com/t/why-is-gpt-3-5-turbo-1106-max-tokens-limited-to-4096/494973/3
        return min(get_max_completion_tokens(messages, self.model, self.config.max_token), 4096)


   
    def get_maxtokens(self) -> int:
       return ['max_tokens']

    async def openai_embedding(self, text):
        response = await self.aclient.embeddings.create(
            model = model, input = text, encoding_format = "float"
        )
        return np.array([dp.embedding for dp in response.data])
