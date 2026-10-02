import subprocess
import os
import pty
import select

env = os.environ.copy()
env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
env["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".10"

master, slave = pty.openpty()
p = subprocess.Popen(["uv", "run", "python", "decode.py", "data/out/sonar/linear/multilingual"], 
                     stdin=subprocess.PIPE, stdout=slave, stderr=slave, close_fds=True, env=env)
os.close(slave)

inputs = [
    b"set 0 k_add 5,10\n",
    b"run The quick brown fox jumps over the lazy dog.\n",
    b"q\n"
]

output = b""
for inp in inputs:
    # Wait for the prompt
    while b"ChatEnv>" not in output:
        r, _, _ = select.select([master], [], [], 1.0)
        if r:
            try:
                chunk = os.read(master, 1024)
                output += chunk
            except OSError:
                break
        else:
            if p.poll() is not None:
                break
    
    # Send input
    p.stdin.write(inp)
    p.stdin.flush()
    output = b"" # Reset output to wait for the next prompt

while True:
    r, _, _ = select.select([master], [], [], 1.0)
    if r:
        try:
            chunk = os.read(master, 1024)
            output += chunk
        except OSError:
            break
    else:
        break

print(output.decode('utf-8', errors='replace'))
