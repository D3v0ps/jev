"""Check that every recipe keeps its questions and thresholds in one reviewable block.

AGENTS.md asks each recipe to put every question and every tuned number directly
under the module docstring, between two markers, so a reviewer reads one screen
instead of hunting through the file. This checks that mechanically:

  1. exactly one review block, opened and closed once, before the first def/class
  2. no float literal and no tuned-looking int literal below the closing marker
  3. a module docstring, and a frozen dataclass whose name ends in Decision

Run: .venv/bin/python scripts/check_review_block.py
"""

from __future__ import annotations

import ast
import pathlib
import sys

OPEN_MARKER = "# --- questions and thresholds (review this block)"
CLOSE_MARKER = "# --- end of review block"

#: Integers that mean structure rather than policy: indices, arity, a percentage base.
STRUCTURAL_INTS = {0, 1, 2, 10, 100, 1000}

RECIPES = pathlib.Path(__file__).resolve().parent.parent / "jevkit" / "recipes"


def block_bounds(lines: list[str], path: str) -> tuple[int, int, list[str]]:
    """1-based line numbers of the open and close markers, plus any problems found."""
    problems = []
    opens = [i + 1 for i, line in enumerate(lines) if line.startswith(OPEN_MARKER)]
    closes = [i + 1 for i, line in enumerate(lines) if line.startswith(CLOSE_MARKER)]
    if len(opens) != 1 or len(closes) != 1:
        problems.append(
            f"{path}: expected one review block, found {len(opens)} open and {len(closes)} close markers"
        )
        return 0, 0, problems
    if closes[0] <= opens[0]:
        problems.append(f"{path}:{closes[0]}: the close marker comes before the open marker")
    return opens[0], closes[0], problems


def check(path: pathlib.Path) -> list[str]:
    source = path.read_text()
    lines = source.splitlines()
    name = str(path.relative_to(path.parent.parent.parent))
    tree = ast.parse(source, filename=str(path))
    opened, closed, problems = block_bounds(lines, name)
    if not opened:
        return problems

    if not ast.get_docstring(tree):
        problems.append(f"{name}:1: no module docstring saying what decision this recipe makes")

    kinds = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    first = min((node.lineno for node in tree.body if isinstance(node, kinds)), default=len(lines))
    if opened > first:
        problems.append(
            f"{name}:{opened}: the review block must sit above the first def/class (line {first})"
        )

    decisions = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name.endswith("Decision")
        and any(
            isinstance(d, ast.Call)
            and getattr(d.func, "id", getattr(d.func, "attr", "")) == "dataclass"
            and any(kw.arg == "frozen" and getattr(kw.value, "value", False) is True for kw in d.keywords)
            for d in node.decorator_list
        )
    ]
    if not decisions:
        problems.append(f"{name}: no frozen dataclass named *Decision carrying the action and its evidence")

    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or node.lineno <= closed:
            continue
        value = node.value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if isinstance(value, float) or value not in STRUCTURAL_INTS:
            problems.append(
                f"{name}:{node.lineno}: tuned literal {value!r} sits below the review block; "
                "name it in the block so a reviewer sees it"
            )
    return problems


def main() -> int:
    modules = sorted(p for p in RECIPES.glob("*.py") if p.name != "__init__.py")
    if not modules:
        print("no recipes found", file=sys.stderr)
        return 1
    problems = [problem for path in modules for problem in check(path)]
    for problem in problems:
        print(problem)
    print(f"{len(modules)} recipes checked, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
