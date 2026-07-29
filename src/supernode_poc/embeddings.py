import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self._cache: dict[str, np.ndarray] = {}

    def embed(self, texts: list[str]) -> np.ndarray:
        missing = list(dict.fromkeys(text for text in texts if text not in self._cache))
        if missing:
            vectors = np.asarray(
                self.model.encode(missing, normalize_embeddings=True), dtype=np.float32
            )
            self._cache.update(zip(missing, vectors, strict=True))
        return np.stack([self._cache[text] for text in texts])
