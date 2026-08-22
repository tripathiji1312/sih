"""Decision arbiter — engine fault vs sensor fault vs fallback."""
from backend.ml.models.sensor_trust import SensorTrustEvaluator

class DecisionArbiter:
    def __init__(self):
        self.trust_eval = SensorTrustEvaluator()

    def arbitrate(self, trust_scores: dict, residuals: dict) -> str:
        return self.trust_eval.arbitrate(trust_scores, residuals)
