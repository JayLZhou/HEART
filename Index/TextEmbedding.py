from modelscope import AutoTokenizer, AutoModel
import openai
import ollama
import numpy as np
import torch
import torch.nn.functional as F
import math
import gc
from tqdm import tqdm
from typing import List
from Common.Logger import logger
from llama_index.core.embeddings import BaseEmbedding


class TextEmbedding(BaseEmbedding):
    """Custom embedding provider using ModelScope."""

    MM_EMBEDDER: bool = False

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-4B",
        backend: str = "local",
        device: list[str] = ["cuda:0"],
        max_length: int = 8192,
        api_base: str = None,
        batch_size: int = 16,
    ):
        super().__init__()
        # Use object.__setattr__ to bypass Pydantic field validation
        object.__setattr__(self, "model_name", model_name)
        object.__setattr__(self, "backend", backend.lower())
        object.__setattr__(self, "max_length", max_length)
        object.__setattr__(self, "batch_size", batch_size)
        
        if self.backend == "local":
            if device == "auto" or (isinstance(device, list) and len(device) > 0 and device[0] == "auto"):
                device_value = "cuda" if torch.cuda.is_available() else "cpu"
            elif isinstance(device, list) and len(device) > 0:
                device_value = device[0]  # Use first device from list
            else:
                device_value = device if isinstance(device, str) else "cuda" if torch.cuda.is_available() else "cpu"
            object.__setattr__(self, "device", device_value)

            logger.info(f"Using local backend (ModelScope) on device: {self.device}")
            logger.info(f"Loading model: {self.model_name}...")

            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, padding_side="left", trust_remote_code=True
                )
                self.model = AutoModel.from_pretrained(
                    self.model_name, trust_remote_code=True
                ).to(self.device)
                self.model.eval()
            except Exception as e:
                logger.error(f"Error loading model '{self.model_name}': {e}")
                raise e

            logger.info("Local model loaded successfully.")

        elif self.backend == "ollama":
            object.__setattr__(self, "device", "ollama_service")
        elif self.backend == "openai":
            object.__setattr__(self, "client", openai.OpenAI(api_key="empty", base_url=api_base))
        else:
            raise ValueError(
                f"Unsupported backend: '{self.backend}'. Choose 'local', 'ollama', or 'openai'."
            )
    def _get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text."""
        embeddings = self._get_text_embeddings([text])
        return embeddings[0]

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_text_embedding(query)

        
    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Async version of _get_text_embedding."""
        return self._get_text_embedding(text)
    def _last_token_pool(
        self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device),
                sequence_lengths,
            ]

    def _normalize(self, embeddings: torch.Tensor) -> np.ndarray:
        """Normalize the PyTorch tensor to L2 norm and convert to numpy array."""
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        return normalized_embeddings.cpu().numpy()

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        if not texts or not isinstance(texts, list):
            raise ValueError("Input 'texts' must be a non-empty list of strings.")

        if self.backend == "local":
            all_embeddings = []
            n = len(texts)
            if n <= self.batch_size:
                batch_dict = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                batch_dict.to(self.device)
                with torch.no_grad():
                    outputs = self.model(**batch_dict)
                embeddings_tensor = self._last_token_pool(
                    outputs.last_hidden_state, batch_dict["attention_mask"]
                )
                all_embeddings.append(embeddings_tensor.cpu())
                embeddings_tensor = torch.cat(all_embeddings, dim=0)
            else:
                num_batches = math.ceil(n / self.batch_size)
                for i in tqdm(
                    range(0, n, self.batch_size),
                    desc="Embedding batches",
                    total=num_batches,
                ):
                    batch_texts = texts[i : i + self.batch_size]
                    batch_dict = self.tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt",
                    )
                    batch_dict.to(self.device)
                    with torch.no_grad():
                        outputs = self.model(**batch_dict)
                    embeddings_tensor = self._last_token_pool(
                        outputs.last_hidden_state, batch_dict["attention_mask"]
                    )
                    all_embeddings.append(embeddings_tensor.cpu())
                embeddings_tensor = torch.cat(all_embeddings, dim=0)

        elif self.backend == "ollama":
            all_embeddings = []
            for text in texts:
                response = ollama.embeddings(model=self.model_name, prompt=text)
                all_embeddings.append(response["embedding"])
            embeddings_np = np.array(all_embeddings, dtype=np.float32)
            embeddings_tensor = torch.from_numpy(embeddings_np).to("cpu")

        elif self.backend == "openai":
            n = len(texts)
            all_embeddings = []
            num_batches = math.ceil(n / self.batch_size)
            for i in tqdm(
                range(0, n, self.batch_size), desc="Embedding texts", total=num_batches
            ):
                chunk = texts[i : i + self.batch_size]
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=chunk,
                )
            
                embeddings_from_this_batch = [item.embedding for item in response.data]
                all_embeddings.extend(embeddings_from_this_batch)

            embeddings_np = np.array(all_embeddings, dtype=np.float32)
            embeddings_tensor = torch.from_numpy(embeddings_np).to("cpu")

        # Normalize and convert to list of lists
        normalized = self._normalize(embeddings_tensor)
        return normalized.tolist()

    def compute_texts_sim(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        embeddings = self._get_text_embeddings([text1, text2])
        vec1 = np.array(embeddings[0])
        vec2 = np.array(embeddings[1])
        similarity = vec1 @ vec2
        return float(similarity)

    def close(self) -> None:
        """Close and release resources."""
        if hasattr(self, "model"):
            logger.info("Releasing embedding model resources...")
            del self.model
            if torch.cuda.is_available():
                logger.info("Embedder: Emptying CUDA cache.")
                torch.cuda.empty_cache()
            gc.collect()
            logger.info("Embedding model resources released.")



