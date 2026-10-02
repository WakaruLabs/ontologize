import subprocess
import re

res = subprocess.run(["git", "diff", "-U0", "HEAD", "--", "ontologize/"], capture_output=True, text=True)
lines = res.stdout.splitlines()

deleted_not_in_quotes = []
current_file = None

for line in lines:
    if line.startswith("--- a/"):
        current_file = line[6:]
    elif line.startswith("-") and not line.startswith("---"):
        stripped = line[1:].strip()
        # Check if line looks like code rather than docstring text
        # Old docstrings contained text or triple quotes
        print(f"DEL in {current_file}: {line[1:]}")
