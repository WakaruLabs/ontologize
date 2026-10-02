import subprocess
import os

env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = ""

inputs = """Step 2 alone creates undesirable aliasing (i.e. high-frequency signal components will copy into the lower frequency band and be mistaken for lower frequencies). Step 1, when necessary, suppresses aliasing to an acceptable level. In this application, the filter is called an anti-aliasing filter, and its design is discussed below. Also see undersampling for information about decimating bandpass functions and signals.
The work was done initially with funding from a Lightspeed Grant, and then continued while at PIBBSS.
I do regret it?
Current job openings
Rate reduction by an integer factor M can be explained as a two-step process, with an equivalent implementation that is more efficient:[5]
q
"""

result = subprocess.run(["uv", "run", "python", "decode.py", "data/out/sonar/linear/multilingual"], input=inputs.encode('utf-8'), capture_output=True, env=env)
print(result.stdout.decode('utf-8', errors='replace'))
print(result.stderr.decode('utf-8', errors='replace'))
