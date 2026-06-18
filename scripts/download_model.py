"""Download the configured final model artifact from Hugging Face Hub."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download


def main() -> None:
    config_path = Path(os.environ.get("CHURN_CONFIG", "config.yaml"))
    with config_path.open("r") as file:
        config = yaml.safe_load(file)

    repo_id = os.environ.get("CHURN_MODEL_REPO", config["models"]["registry_repo"])
    model_path = Path(config["models"]["final_model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)

    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=model_path.name,
        repo_type=os.environ.get("CHURN_MODEL_REPO_TYPE", "model"),
    )
    shutil.copyfile(downloaded_path, model_path)
    print(f"Model downloaded from {repo_id} to {model_path}")


if __name__ == "__main__":
    main()
