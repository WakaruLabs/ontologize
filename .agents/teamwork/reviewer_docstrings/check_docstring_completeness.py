import ast
import os
import glob

def check_file_docstrings(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=filepath)

    missing = []
    
    # 1. Module docstring
    module_doc = ast.get_docstring(tree)
    if not module_doc:
        missing.append(("module", filepath, 1))

    # 2. Classes and functions
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            if not node.name.startswith("_"):
                if not ast.get_docstring(node):
                    missing.append(("class", f"{filepath}::{node.name}", node.lineno))
            # Methods in class
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Check if public or standard special method like __call__, __init__
                    if not item.name.startswith("_") or item.name in ("__call__", "__init__"):
                        if not ast.get_docstring(item):
                            missing.append(("method", f"{filepath}::{node.name}.{item.name}", item.lineno))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                if not ast.get_docstring(node):
                    missing.append(("function", f"{filepath}::{node.name}", node.lineno))

    return missing

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

total_missing = []
for f in files:
    m = check_file_docstrings(f)
    if m:
        total_missing.extend(m)

print(f"Total missing docstrings: {len(total_missing)}")
for kind, name, line in total_missing:
    print(f"MISSING [{kind}] {name} (line {line})")
