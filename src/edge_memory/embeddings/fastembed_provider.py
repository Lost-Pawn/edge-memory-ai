"""FastEmbed-backed on-device embeddings.

Models are loaded lazily and, by default, with ``local_files_only=True`` so the running
application never reaches out to the network for weights. Downloading models is a
separate, explicit step (see :func:`download_models` / the ``download-models`` CLI).
"""

from __future__ import annotations

from pathlib import Path

from edge_memory.config import EdgeSettings
from edge_memory.embeddings.base import SparseVector
from edge_memory.exceptions import ModelNotFoundError
from edge_memory.logging import get_logger

log = get_logger(__name__)


class FastEmbedProvider:
    """Dense (CLIP text), sparse (BM25), and image (CLIP vision) embeddings."""

    def __init__(self, settings: EdgeSettings) -> None:
        self._settings = settings
        self._cache_dir = str(settings.models_dir)
        self._local_only = settings.local_files_only
        self._text = None
        self._sparse = None
        self._image = None

    @property
    def dense_dim(self) -> int:
        return self._settings.dense_dim

    # ── lazy loaders ────────────────────────────────────────────────────────
    def _text_model(self):  # type: ignore[no-untyped-def]
        if self._text is None:
            from fastembed import TextEmbedding

            log.info("loading dense text model", model=self._settings.dense_model)
            self._text = self._guard(
                lambda: TextEmbedding(
                    model_name=self._settings.dense_model,
                    cache_dir=self._cache_dir,
                    local_files_only=self._local_only,
                )
            )
        return self._text

    def _sparse_model(self):  # type: ignore[no-untyped-def]
        if self._sparse is None:
            from fastembed import SparseTextEmbedding

            log.info("loading sparse model", model=self._settings.sparse_model)
            # BM25 caches under a different (GCS) layout than the HF models, so the
            # strict offline lookup can miss it. BM25 has no neural weights, so a
            # non-strict fallback is safe and still offline once cached.
            self._sparse = self._guard(
                lambda: SparseTextEmbedding(
                    model_name=self._settings.sparse_model,
                    cache_dir=self._cache_dir,
                    local_files_only=self._local_only,
                ),
                fallback=lambda: SparseTextEmbedding(
                    model_name=self._settings.sparse_model,
                    cache_dir=self._cache_dir,
                ),
            )
        return self._sparse

    def _image_model(self):  # type: ignore[no-untyped-def]
        if self._image is None:
            from fastembed import ImageEmbedding

            log.info("loading image model", model=self._settings.image_model)
            self._image = self._guard(
                lambda: ImageEmbedding(
                    model_name=self._settings.image_model,
                    cache_dir=self._cache_dir,
                    local_files_only=self._local_only,
                )
            )
        return self._image

    @staticmethod
    def _guard(loader, fallback=None):  # type: ignore[no-untyped-def]
        try:
            return loader()
        except Exception as exc:  # fastembed raises when cache is empty in local mode
            if fallback is not None:
                try:
                    return fallback()
                except Exception:  # noqa: S110 - fall through to a clear error
                    pass
            raise ModelNotFoundError(
                "Embedding weights not found in the local cache. Run "
                "`make download-models` (with network) before going offline."
            ) from exc

    # ── inference ───────────────────────────────────────────────────────────
    def embed_text(self, texts: list[str]) -> list[list[float]]:
        return [vec.tolist() for vec in self._text_model().embed(texts)]

    def embed_sparse(self, texts: list[str]) -> list[SparseVector]:
        out: list[SparseVector] = []
        for sv in self._sparse_model().embed(texts):
            out.append(
                SparseVector(
                    indices=[int(i) for i in sv.indices],
                    values=[float(v) for v in sv.values],
                )
            )
        return out

    def embed_image(self, paths: list[Path]) -> list[list[float]]:
        return [vec.tolist() for vec in self._image_model().embed([str(p) for p in paths])]


def download_models(settings: EdgeSettings) -> None:
    """Explicitly download and cache FastEmbed models. Only called on purpose."""
    from fastembed import ImageEmbedding, SparseTextEmbedding, TextEmbedding

    settings.models_dir.mkdir(parents=True, exist_ok=True)
    cache = str(settings.models_dir)
    log.info("downloading embedding models (explicit)", cache_dir=cache)
    TextEmbedding(model_name=settings.dense_model, cache_dir=cache)
    SparseTextEmbedding(model_name=settings.sparse_model, cache_dir=cache)
    ImageEmbedding(model_name=settings.image_model, cache_dir=cache)
    log.info("embedding models cached")
