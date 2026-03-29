"""AST parsing utility to extract lightweight code signatures."""

import ast

class SignatureStripper(ast.NodeTransformer):
    """Keeps class/function signatures and docstrings, replacing implementations with `...`."""
    
    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        new_body = []
        docstring = ast.get_docstring(node)
        if docstring:
            new_body.append(ast.Expr(value=ast.Constant(value=docstring)))
        new_body.append(ast.Expr(value=ast.Constant(value=Ellipsis)))
        node.body = new_body
        return node

    def visit_AsyncFunctionDef(self, node):
        self.generic_visit(node)
        new_body = []
        docstring = ast.get_docstring(node)
        if docstring:
            new_body.append(ast.Expr(value=ast.Constant(value=docstring)))
        new_body.append(ast.Expr(value=ast.Constant(value=Ellipsis)))
        node.body = new_body
        return node
        
    def visit_ClassDef(self, node):
        self.generic_visit(node)
        new_body = []
        docstring = ast.get_docstring(node)
        if docstring:
            new_body.append(ast.Expr(value=ast.Constant(value=docstring)))
            
        # Retain class-level assignments (vital for Hugging Face Config class attributes) and methods
        for stmt in node.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign)):
                new_body.append(stmt)
                
        # If the class was mostly logic outside methods (rare), add Ellipsis
        if not new_body or (docstring and len(new_body) == 1):
            new_body.append(ast.Expr(value=ast.Constant(value=Ellipsis)))
            
        node.body = new_body
        return node

def extract_signatures(source_code: str) -> str:
    """Parse python source code and return only class/function signatures and docstrings.
    
    This helps compress the context window for large LLM code-generation prompts
    by replacing method implementations with `...`.
    """
    try:
        tree = ast.parse(source_code)
        tree = SignatureStripper().visit(tree)
        ast.fix_missing_locations(tree)
        
        if hasattr(ast, 'unparse'):  # Python 3.9+ feature
            return ast.unparse(tree)
        return source_code
    except Exception:
        # Graceful fallback to original code if there's a syntax error
        return source_code
