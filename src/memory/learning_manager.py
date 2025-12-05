import os
import json
from typing import Dict, Any
from datetime import datetime

class LearningManager:
    def __init__(self, persist_dir: str):
        os.makedirs(persist_dir, exist_ok=True)
        self.path = os.path.join(persist_dir, "learning.json")
        self.data: Dict[str, Any] = {
            "synonym_weights": {},
            "source_type_weights": {"industry": 1.0, "academic": 1.0, "technical": 0.8, "general": 0.7, "tech_news": 0.6},
            "history": []
        }
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception:
                pass

    def _save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False)
        except Exception:
            pass

    def get_synonym_weights(self) -> Dict[str, float]:
        return self.data.get("synonym_weights", {})

    def update_after_run(self, metrics: Dict[str, Any]):
        diversity = metrics.get("source_diversity", 0)
        coverage = metrics.get("coverage_iterations", 0)
        closure = metrics.get("closure", 0)
        # 简单规则：提升行业与学术来源权重，降低纯技术来源权重
        if diversity < 5:
            self.data["source_type_weights"]["industry"] = min(2.0, self.data["source_type_weights"].get("industry", 1.0) + 0.1)
            self.data["source_type_weights"]["academic"] = min(2.0, self.data["source_type_weights"].get("academic", 1.0) + 0.05)
        if closure < 0.9:
            self.data["source_type_weights"]["general"] = min(1.5, self.data["source_type_weights"].get("general", 0.7) + 0.05)
        self.data["history"].append({"timestamp": datetime.now().isoformat(), "metrics": metrics})
        self._save()

