#!/usr/bin/env python3
import ast
import subprocess
import sys
from pathlib import Path

class DocstringStripper(ast.NodeTransformer):
    def visit_Module(self, node):
        self.generic_visit(node)
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:]
        return node

    def visit_ClassDef(self, node):
        self.generic_visit(node)
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:]
        return node

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:]
        return node

    def visit_AsyncFunctionDef(self, node):
        self.generic_visit(node)
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:]
        return node

def strip_and_dump(source: str) -> str:
    tree = ast.parse(source)
    stripper = DocstringStripper()
    clean_tree = stripper.visit(tree)
    ast.fix_missing_locations(clean_tree)
    return ast.dump(clean_tree, include_attributes=False)

def verify_file(rel_path: str) -> bool:
    path = Path(rel_path)
    if not path.exists():
        print(f"File {rel_path} does not exist.")
        return False
    
    # Read current content
    current_content = path.read_text(encoding="utf-8")
    
    # Read git HEAD content
    cmd = ["git", "show", f"HEAD:{rel_path}"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        # File might be untracked or new
        print(f"Note: {rel_path} not found in HEAD (exit code {proc.returncode}).")
        # If it's a new file (e.g. __init__.py), check if it contains only docstrings
        current_dump = strip_and_dump(current_content)
        empty_dump = strip_and_dump("")
        if current_dump == empty_dump:
            print(f"PASS (new file with only docstring): {rel_path}")
            return True
        else:
            print(f"FAIL: {rel_path} is new but contains non-docstring code!")
            return False

    head_content = proc.stdout
    dump_head = strip_and_dump(head_content)
    dump_current = strip_and_dump(current_content)

    if dump_head == dump_current:
        print(f"PASS (AST invariant): {rel_path}")
        return True
    else:
        print(f"FAIL (AST mismatch): {rel_path}")
        return False

def main():
    files = sys.argv[1:]
    if not files:
        print("Usage: ast_verify.py <file1> <file2> ...")
        sys.exit(1)
    
    all_ok = True
    for f in files:
        if not verify_file(f):
            all_ok = False
            
    if all_ok:
        print("\nAll files passed AST syntactic invariance check!")
        sys.exit(0)
    else:
        print("\nSome files failed AST invariance check!")
        sys.exit(1)

if __name__ == "__main__":
    main()
