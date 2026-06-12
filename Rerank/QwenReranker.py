import copy
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from Rerank.BasicRerank import BaseRanking
from Rerank.Utils import get_device
from Schema.DocumentSchema import Document
from tqdm import tqdm


class QwenReranker(BaseRanking):
    """Qwen3 text reranker using yes/no relevance scoring.

    This follows the official Qwen3 reranker scoring pattern at a lightweight
    adapter level so it can fit the repo's current reranker interface.
    """

    SYSTEM_PROMPT = (
        'Judge whether the Document meets the requirements based on the Query '
        'and the Instruct provided. Note that the answer can only be "yes" or "no".'
    )

    DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"

    def __init__(self, method=None, model_name="Qwen/Qwen3-Reranker-0.6B", api_key=None, **kwargs):
        self.method = method
        self.model_name = model_name
        self.device = get_device(kwargs.get("device", "cuda"))
        self.batch_size = kwargs.get("batch_size", 8)
        self.max_length = kwargs.get("max_length", 4096)
        self.instruction = kwargs.get("instruction", self.DEFAULT_INSTRUCTION)

        torch_dtype = kwargs.get("torch_dtype")
        if torch_dtype is None:
            if str(self.device).startswith("cuda"):
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            padding_side="left",
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Workaround: transformers sets ALL_PARALLEL_STYLES=None when torch distributed
        # is unavailable, but Qwen3's tp_plan validation in post_init crashes on None.
        import transformers.modeling_utils as _mu
        if _mu.ALL_PARALLEL_STYLES is None:
            _mu.ALL_PARALLEL_STYLES = frozenset({'colwise', 'rowwise', 'local_colwise', 'local_rowwise'})
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

        self.true_token_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.false_token_id = self.tokenizer.convert_tokens_to_ids("no")

    def rank(self, documents: List[Document]) -> List[Document]:
        for document in tqdm(documents, desc="Reranking Documents"):
            contexts = [context.text for context in document.contexts]
            if not contexts:
                document.reorder_contexts = []
                continue

            scores = self._score_passages(document.question, contexts)
            ranked = []
            for context, score in zip(document.contexts, scores):
                ctx = copy.deepcopy(context)
                ctx.score = float(score)
                ranked.append(ctx)
            document.reorder_contexts = sorted(ranked, key=lambda item: item.score, reverse=True)
        return documents

    def _score_passages(self, query: str, passages: List[str]) -> List[float]:
        scores: List[float] = []
        for start in range(0, len(passages), self.batch_size):
            batch = passages[start : start + self.batch_size]
            messages = [
                [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {passage}",
                    },
                ]
                for passage in batch
            ]
            tokenized = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)
            tokenized = [tokens[: self.max_length - len(suffix_tokens)] + suffix_tokens for tokens in tokenized]
            model_inputs = self.tokenizer.pad(
                {"input_ids": tokenized},
                padding=True,
                return_tensors="pt",
                max_length=self.max_length,
            )
            model_inputs = {key: value.to(self.device) for key, value in model_inputs.items()}

            with torch.no_grad():
                logits = self.model(**model_inputs).logits[:, -1, :]
            true_scores = logits[:, self.true_token_id]
            false_scores = logits[:, self.false_token_id]
            pair = torch.stack([false_scores, true_scores], dim=1)
            probs = torch.softmax(pair, dim=1)[:, 1]
            scores.extend(probs.detach().cpu().tolist())
        return scores
