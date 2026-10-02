import subprocess
import os

env = os.environ.copy()
env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
env["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
env["CUDA_VISIBLE_DEVICES"] = ""

inputs = b"set 0 k_add 5,10\nrun The quick brown fox jumps over the lazy dog.\nq\n"

result = subprocess.run(["uv", "run", "python", "decode.py", "data/out/sonar/linear/multilingual"], 
                     input=inputs, capture_output=True, env=env)
print("STDOUT:", result.stdout.decode('utf-8', errors='replace'))
print("STDERR:", result.stderr.decode('utf-8', errors='replace'))
