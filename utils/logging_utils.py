import os
import json

class JSONLogger:
    def __init__(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.file = open(filepath, "w", encoding="utf-8")

    def log(self, data: dict):
        self.file.write(json.dumps(data) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()