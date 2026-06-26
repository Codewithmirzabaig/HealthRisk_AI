"""
HealthRisk AI
Regulatory & Responsible AI Mapping
"""

from pathlib import Path
import json
from datetime import datetime


class RegulatoryMapping:

    def __init__(self):

        self.mapping = {
            "project": "HealthRisk AI",
            "version": "2.0",
            "generated": datetime.now().isoformat(),
            "principles": {}
        }

    def add_principle(
        self,
        principle,
        implementation,
        status="Implemented"
    ):

        self.mapping["principles"][principle] = {
            "implementation": implementation,
            "status": status
        }

    def save_json(self, output_dir="reports"):

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        path = output_dir / "regulatory_mapping.json"

        with open(path, "w") as f:
            json.dump(self.mapping, f, indent=4)

        return path

    def save_markdown(self, output_dir="reports"):

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        path = output_dir / "regulatory_mapping.md"

        md = f"""# Responsible AI Governance Mapping

Project: {self.mapping['project']}

Version: {self.mapping['version']}

Generated:
{self.mapping['generated']}

---

| Principle | Implementation | Status |
|-----------|----------------|--------|
"""

        for principle, value in self.mapping["principles"].items():

            md += (
                f"| {principle} | "
                f"{value['implementation']} | "
                f"{value['status']} |\n"
            )

        with open(path, "w", encoding="utf-8") as f:
            f.write(md)

        return path