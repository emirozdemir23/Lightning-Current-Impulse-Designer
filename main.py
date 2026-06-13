import json
from pathlib import Path

from src.gui import launch_gui


def main():
    config_path = Path(__file__).parent / "config.json"
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    launch_gui(initial_config=config["impulse_generator"])


if __name__ == "__main__":
    main()
