import json
import numbers


class JsonLogger:
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        self.file = open(self.path, 'a', buffering=1)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def log(self, data):
        scalars = {
            key: int(value) if isinstance(value, numbers.Integral) else float(value)
            for key, value in data.items()
            if isinstance(value, numbers.Number)
        }
        self.file.write(json.dumps(scalars) + '\n')
