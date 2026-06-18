"""Download the Telco customer churn dataset to the configured raw data path."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import yaml


DEFAULT_DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)


def main() -> None:
    config_path = Path(os.environ.get("CHURN_CONFIG", "config.yaml"))
    with config_path.open("r") as file:
        config = yaml.safe_load(file)

    output_path = Path(config["data"]["raw_data_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    url = os.environ.get("CHURN_DATA_URL", DEFAULT_DATA_URL)
    print(f"Downloading dataset from {url}")
    urllib.request.urlretrieve(url, output_path)
    print(f"Dataset saved to {output_path}")


if __name__ == "__main__":
    main()
