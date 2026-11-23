import json
import os
from pathlib import Path

class Config:
    def __init__(self, config_path="config.json"):
        self.config_path = Path(config_path)
        self.settings = {
            "input_dir": "",
            "output_dir": "output",
            "output_filename": "Timelapse_output",
            "speed_multiplier": 4.0,
            "music_path": "",
            "music_volume": 0.5,
            "min_stationary_seconds": 3.0,
            "motion_threshold": 150000,
            "blur_plates": True,
            "enable_gpu": True
        }
        self.load()

    def load(self):
        if self.config_path.exists():
            try:
                with open(self.config_path, "r") as f:
                    loaded = json.load(f)
                    self.settings.update(loaded)
            except Exception as e:
                print(f"Error loading config: {e}")

    def save(self):
        try:
            with open(self.config_path, "w") as f:
                json.dump(self.settings, f, indent=4)
        except Exception as e:
            print(f"Error saving config: {e}")

    def get(self, key, default=None):
        return self.settings.get(key, default)

    def set(self, key, value, auto_save=False):
        self.settings[key] = value
        if auto_save:
            self.save()

app_config = Config()
