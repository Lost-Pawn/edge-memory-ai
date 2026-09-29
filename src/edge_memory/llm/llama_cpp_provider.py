"""In-process local LLM via llama.cpp (no daemon, no network).

The GGUF model file must already exist on disk; it is never downloaded at runtime. Use
the ``download-llm`` CLI command to fetch one explicitly. If the model is missing or
llama-cpp-python is not installed, the provider degrades to :class:`NullLLM` so the rest
of the platform (retrieval, sync, policy) keeps working.
"""

from __future__ import annotations

from edge_memory.config import EdgeSettings
from edge_memory.logging import get_logger

log = get_logger(__name__)


class NullLLM:
    """Fallback used when no local model is available.

    Returns an extractive answer built from the retrieved context so the product still
    responds offline without a generative model.
    """

    @property
    def available(self) -> bool:
        return False

    def generate(
        self, prompt: str, *, system: str | None = None, max_tokens: int = 512
    ) -> str:
        return (
            "[no local LLM loaded] Returning retrieved context verbatim.\n\n" + prompt
        )


class LlamaCppProvider:
    """Loads a quantized GGUF model into the process on first use."""

    def __init__(self, settings: EdgeSettings) -> None:
        self._settings = settings
        self._llm = None
        self._failed = False

    @property
    def available(self) -> bool:
        return not self._failed and self._settings.llm_model_path.exists()

    def _model(self):  # type: ignore[no-untyped-def]
        if self._llm is None:
            if not self._settings.llm_model_path.exists():
                raise FileNotFoundError(self._settings.llm_model_path)
            from llama_cpp import Llama

            log.info("loading local LLM", path=str(self._settings.llm_model_path))
            self._llm = Llama(
                model_path=str(self._settings.llm_model_path),
                n_ctx=self._settings.llm_ctx,
                n_threads=self._settings.llm_threads,
                n_gpu_layers=self._settings.llm_gpu_layers,
                verbose=False,
            )
        return self._llm

    def generate(
        self, prompt: str, *, system: str | None = None, max_tokens: int = 512
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        result = self._model().create_chat_completion(
            messages=messages, max_tokens=max_tokens, temperature=0.2
        )
        return result["choices"][0]["message"]["content"].strip()


def build_llm(settings: EdgeSettings) -> LLMProvider:  # type: ignore[valid-type]
    """Return a llama.cpp provider if a model is present, else the null fallback."""
    if settings.llm_enabled and settings.llm_model_path.exists():
        try:
            provider = LlamaCppProvider(settings)
            return provider
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("llm init failed, using NullLLM", error=str(exc))
    return NullLLM()


def download_llm(settings: EdgeSettings) -> None:
    """Explicitly download a GGUF model from Hugging Face. Only called on purpose."""
    from huggingface_hub import hf_hub_download

    target_dir = settings.llm_model_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    log.info(
        "downloading LLM (explicit)",
        repo=settings.download_llm_repo,
        file=settings.download_llm_file,
    )
    path = hf_hub_download(
        repo_id=settings.download_llm_repo,
        filename=settings.download_llm_file,
        local_dir=str(target_dir),
    )
    log.info("LLM downloaded", path=path)
