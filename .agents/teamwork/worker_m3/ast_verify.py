"""AST Syntactic Invariance Verification Tool for Docstring-Only Changes.

Compares an original file against a modified file and confirms that after stripping
module, class, and function/method docstrings, the AST structures are identical.
"""
import ast
import sys
from pathlib import Path


class DocstringStripper(ast.NodeTransformer):
    def _strip_docstring(self, body):
        if not body:
            return body
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
            if isinstance(first.value.value, str):
                return body[1:]
        return body

    def visit_Module(self, node):
        node.body = self._strip_docstring(node.body)
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node):
        node.body = self._strip_docstring(node.body)
        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node):
        node.body = self._strip_docstring(node.body)
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node):
        node.body = self._strip_docstring(node.body)
        self.generic_visit(node)
        return node


def check_invariance(orig_code: str, new_code: str) -> tuple[bool, str]:
    try:
        orig_ast = ast.parse(orig_code)
    except Exception as e:
        return False, f"Failed to parse original code: {e}"
    try:
        new_ast = ast.parse(new_code)
    except Exception as e:
        return False, f"Failed to parse new code: {e}"

    stripper = DocstringStripper()
    orig_stripped = stripper.visit(orig_ast)
    new_stripped = stripper.visit(new_ast)

    dump_orig = ast.dump(orig_stripped, include_attributes=False)
    dump_new = ast.dump(new_stripped, include_attributes=False)

    if dump_orig == dump_new:
        return True, "ASTs match identically (docstrings ignored)."
    else:
        return False, f"AST mismatch!\nOriginal length: {len(dump_orig)}, New length: {len(dump_new)}"


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 ast_verify.py <orig_file> <new_file>")
        sys.exit(1)
    orig_path = Path(sys.argv[1])
    new_path = Path(sys.argv[2])
    ok, msg = check_invariance(orig_path.read_text(), new_path.read_text())
    print(f"Checking {orig_path.name} vs {new_path.name}: {ok} - {msg}")
    sys.exit(0 if ok else 1)
