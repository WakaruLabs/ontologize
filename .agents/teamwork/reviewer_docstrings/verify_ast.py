import ast
import subprocess
import sys
from pathlib import Path

def strip_docstrings(node):
    """Recursively remove docstring Expr nodes from Module, ClassDef, and FunctionDef."""
    for n in ast.walk(node):
        if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant) and isinstance(n.body[0].value.value, str):
                n.body = n.body[1:]
    return node

def ast_to_tuple(node):
    """Convert an AST node to a canonical tuple representation ignoring line numbers and col offsets."""
    if not isinstance(node, ast.AST):
        return node
    fields = []
    for field, value in ast.iter_fields(node):
        if isinstance(value, list):
            fields.append((field, tuple(ast_to_tuple(item) for item in value)))
        else:
            fields.append((field, ast_to_tuple(value)))
    return (type(node).__name__, tuple(fields))

def check_file_ast(file_path):
    # Get HEAD content
    res = subprocess.run(["git", "show", f"HEAD:{file_path}"], capture_output=True, text=True)
    if res.returncode != 0:
        # New file (e.g. training/__init__.py)
        # Check that it ONLY contains docstrings or empty comments
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        tree = ast.parse(content)
        tree = strip_docstrings(tree)
        if tree.body:
            return False, f"New file {file_path} contains non-docstring AST nodes: {tree.body}"
        return True, "New file contains only docstrings"

    head_content = res.stdout
    with open(file_path, "r", encoding="utf-8") as f:
        curr_content = f.read()

    tree_head = strip_docstrings(ast.parse(head_content))
    tree_curr = strip_docstrings(ast.parse(curr_content))

    t_head = ast_to_tuple(tree_head)
    t_curr = ast_to_tuple(tree_curr)

    if t_head != t_curr:
        return False, f"AST mismatch in {file_path}"
    return True, "AST match"

if __name__ == "__main__":
    files = [
        "ontologize/chat.py",
        "ontologize/data/__init__.py",
        "ontologize/data/langs.py",
        "ontologize/data/loaders.py",
        "ontologize/data/multilingual.py",
        "ontologize/data/pretrained.py",
        "ontologize/fns/classify.py",
        "ontologize/fns/keys.py",
        "ontologize/fns/loss.py",
        "ontologize/inference/steerable.py",
        "ontologize/layers/__init__.py",
        "ontologize/layers/dictblock.py",
        "ontologize/layers/dictenc.py",
        "ontologize/layers/linear.py",
        "ontologize/layers/nlinear.py",
        "ontologize/layers/sparse.py",
        "ontologize/main.py",
        "ontologize/ontologizer.py",
        "ontologize/training/__init__.py",
        "ontologize/training/config.py",
        "ontologize/training/ontostate.py",
        "ontologize/training/serialize.py",
        "ontologize/visualize/loss.py",
    ]

    all_ok = True
    for f in files:
        ok, msg = check_file_ast(f)
        print(f"[{'PASS' if ok else 'FAIL'}] {f}: {msg}")
        if not ok:
            all_ok = False

    if not all_ok:
        sys.exit(1)
