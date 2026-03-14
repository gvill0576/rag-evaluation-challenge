import json
from datetime import datetime
from pathlib import Path

class QualityTracker:
    def __init__(self, history_file: str = "data/quality_history.jsonl"):
        self.history_file = Path(history_file)
        self.history_file.parent.mkdir(exist_ok=True)

    def log_evaluation(self, evaluation: dict):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "metrics": evaluation.get("category_averages", {}),
            "pass_rate": evaluation.get("pass_rate", 0),
            "total": evaluation.get("total", 0),
            "passed": evaluation.get("passed", 0)
        }
        with open(self.history_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_history(self, days: int = 7) -> list:
        if not self.history_file.exists():
            return []
        cutoff = datetime.now().timestamp() - (days * 86400)
        entries = []
        with open(self.history_file) as f:
            for line in f:
                entry = json.loads(line)
                ts = datetime.fromisoformat(entry["timestamp"]).timestamp()
                if ts >= cutoff:
                    entries.append(entry)
        return entries

    def get_trend(self, metric: str, days: int = 7) -> dict:
        entries = self.get_history(days)
        if len(entries) < 2:
            return {"trend": "insufficient_data", "entries": len(entries)}
        values = [e["metrics"].get(metric, e.get("pass_rate", 0)) for e in entries]
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        avg_first = sum(first_half) / len(first_half)
        avg_second = sum(second_half) / len(second_half)
        change = avg_second - avg_first
        trend = "improving" if change > 0.5 else "degrading" if change < -0.5 else "stable"
        return {"trend": trend, "change": round(change, 2),
                "current": round(avg_second, 2), "previous": round(avg_first, 2)}

    def check_regression(self, current: dict, threshold: float = 1.0) -> dict:
        regressions = []
        for cat, score in current.get("category_averages", {}).items():
            trend = self.get_trend(cat)
            if trend["trend"] == "degrading" and abs(trend["change"]) > threshold:
                regressions.append({"category": cat, "current": score, "change": trend["change"]})
        return {"has_regression": len(regressions) > 0, "regressions": regressions}
