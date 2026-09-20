"""Every recipe must stay reviewable: one block of questions and thresholds, up top."""

from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from check_review_block import RECIPES, check  # noqa: E402

MODULES = sorted(p for p in RECIPES.glob("*.py") if p.name != "__init__.py")


def test_there_are_recipes():
    assert MODULES, "no recipe modules found"


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.stem)
def test_recipe_keeps_its_constants_reviewable(path):
    problems = check(path)
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.stem)
def test_recipe_has_a_doc_an_example_and_a_test(path):
    companions = [
        ROOT / "docs" / f"{path.stem}.md",
        ROOT / "examples" / f"{path.stem}.py",
        ROOT / "tests" / f"test_{path.stem}.py",
    ]
    for expected in companions:
        assert expected.exists(), f"missing {expected.relative_to(ROOT)}"
