class CostTracker:
    PRICING = {
        "us.amazon.nova-lite-v1:0": {"input": 0.0002, "output": 0.0008},
        "us.amazon.nova-pro-v1:0": {"input": 0.001, "output": 0.004}
    }

    def __init__(self):
        self.total_cost = 0.0
        self.calls = 0

    def track(self, model: str, input_text: str, output_text: str) -> float:
        pricing = self.PRICING.get(model, {"input": 0.001, "output": 0.004})
        input_tokens = len(input_text) // 4
        output_tokens = len(output_text) // 4
        cost = (input_tokens / 1000) * pricing["input"] + \
               (output_tokens / 1000) * pricing["output"]
        self.total_cost += cost
        self.calls += 1
        return cost

    def report(self) -> dict:
        return {
            "total_cost": round(self.total_cost, 4),
            "total_calls": self.calls,
            "avg_per_call": round(self.total_cost / self.calls, 6) if self.calls else 0,
            "estimated_monthly": round(self.total_cost * 30, 2) if self.calls else 0
        }
