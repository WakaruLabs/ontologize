import subprocess
import os

env = os.environ.copy()
env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
env["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".10"

result = subprocess.run(["uv", "run", "python", "decode_tags.py", "data/out/sonar/linear/multilingual"], capture_output=True, env=env)
print(result.stdout.decode('utf-8', errors='replace'))
print(result.stderr.decode('utf-8', errors='replace'))
