import hashlib
import json
from pathlib import Path

class EvaluationCache:
    def __init__(self, cache_dir: str = ".eval_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _hash(self, *args) -> str:
        content = json.dumps(args, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, metric: str, *args):
        key = self._hash(metric, *args)
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            self.hits += 1
            with open(path) as f:
                return json.load(f)
        self.misses += 1
        return None

    def set(self, result: dict, metric: str, *args):
        key = self._hash(metric, *args)
        path = self.cache_dir / f"{key}.json"
        with open(path, "w") as f:
            json.dump(result, f)

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 2) if total > 0 else 0,
            "cached_files": len(list(self.cache_dir.glob("*.json")))
        }
