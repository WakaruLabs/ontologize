import subprocess
import re

res = subprocess.run(["git", "diff", "HEAD", "--", "ontologize/"], capture_output=True, text=True)
diff_lines = res.stdout.splitlines()

deleted_lines = []
in_docstring = False
file_name = None

# We want to check every line starting with '-' that is not '---'
for line in diff_lines:
    if line.startswith("diff --git"):
        file_name = line.split()[-1]
    elif line.startswith("--- "):
        continue
    elif line.startswith("-"):
        content = line[1:].strip()
        # If line was a comment
        if content.startswith("#"):
            deleted_lines.append((file_name, "comment", line))
        # If line was not triple-quotes or empty or docstring text
        elif content and not (content.startswith('"""') or content.endswith('"""') or content.startswith("'''") or content.endswith("'''")):
            # Check if this was inside an old docstring
            deleted_lines.append((file_name, "other", line))

print(f"Total deleted lines flagged: {len(deleted_lines)}")
for f, kind, l in deleted_lines[:30]:
    print(f"[{kind}] {f}: {l}")
