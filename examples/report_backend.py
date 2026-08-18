import json

from scientific_computing_rocm import backend_report, choose_device

print(json.dumps(backend_report(), indent=2))
print(f"Selected device: {choose_device()}")
