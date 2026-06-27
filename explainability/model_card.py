"""
HealthRisk AI
Model Card Generator

Generates Markdown and JSON documentation
for trained machine learning models.
"""

from pathlib import Path
import json
from datetime import datetime


class ModelCard:

    def __init__(
        self,
        model_name,
        version,
        algorithm,
        author="MIRZA SHARIF BAIG",
    ):

        self.card = {
            "model_name": model_name,
            "version": version,
            "algorithm": algorithm,
            "author": author,
            "generated": datetime.now().isoformat(),
            "dataset": {},
            "performance": {},
            "features": [],
            "limitations": [],
            "ethical_considerations": [],
            "intended_use": "",
        }

    def set_dataset(
        self,
        name,
        records,
        description,
    ):

        self.card["dataset"] = {
            "name": name,
            "records": records,
            "description": description,
        }

    def set_performance(
        self,
        mae,
        rmse,
        r2,
    ):

        self.card["performance"] = {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        }

    def add_features(self, features):

        self.card["features"] = features

    def set_intended_use(self, text):

        self.card["intended_use"] = text

    def add_limitations(self, limitations):

        self.card["limitations"] = limitations

    def add_ethics(self, ethics):

        self.card["ethical_considerations"] = ethics

    def save_json(self, output_dir="reports"):

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        path = output_dir / "model_card.json"

        with open(path, "w") as f:
            json.dump(self.card, f, indent=4)

        return path

    def save_markdown(self, output_dir="reports"):

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        path = output_dir / "model_card.md"

        md = f"""# {self.card['model_name']}

## Version

{self.card['version']}

## Algorithm

{self.card['algorithm']}

## Author

{self.card['author']}

## Generated

{self.card['generated']}

## Dataset

- Name: {self.card['dataset']['name']}
- Records: {self.card['dataset']['records']}
- Description: {self.card['dataset']['description']}

## Performance

| Metric | Value |
|--------|------:|
| MAE | {self.card['performance']['MAE']} |
| RMSE | {self.card['performance']['RMSE']} |
| R² | {self.card['performance']['R2']} |

## Features

"""

        for feature in self.card["features"]:
            md += f"- {feature}\n"

        md += "\n## Intended Use\n\n"
        md += self.card["intended_use"]

        md += "\n\n## Limitations\n\n"

        for item in self.card["limitations"]:
            md += f"- {item}\n"

        md += "\n\n## Ethical Considerations\n\n"

        for item in self.card["ethical_considerations"]:
            md += f"- {item}\n"

        with open(path, "w", encoding="utf-8") as f:
            f.write(md)

        return path