#!/usr/bin/env python3
"""Small repository-local validation for an independently installable skill."""
from __future__ import annotations

import sys
from pathlib import Path

import yaml


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "SKILL.md").read_text(encoding="utf-8")
    if not source.startswith("---"):
        raise SystemExit("SKILL.md is missing YAML frontmatter")
    metadata = yaml.safe_load(source.split("---", 2)[1])
    quant = metadata.get("quantSkills", {})
    required = ["name", "description"]
    if any(not str(metadata.get(key, "")).strip() for key in required):
        raise SystemExit("SKILL.md lacks name or description")
    expected = root.name
    if quant.get("repository") != expected:
        raise SystemExit("quantSkills.repository must match the repository directory")
    if quant.get("repository_url") != f"https://github.com/quantskills/{expected}":
        raise SystemExit("quantSkills.repository_url is inconsistent")
    for path in [root / "pyproject.toml", root / "README.md", root / "README.en.md", root / "LICENSE",
                 root / "scripts" / "run_model.py", root / "tests"]:
        if not path.exists():
            raise SystemExit(f"required path is missing: {path.relative_to(root)}")
    forbidden = ("/home/", "PandaAI_code", "panda_factor_common")
    for source_path in [*root.glob("src/**/*.py"), root / "scripts" / "run_model.py"]:
        content = source_path.read_text(encoding="utf-8")
        if any(token in content for token in forbidden):
            raise SystemExit(f"local dependency found in {source_path.relative_to(root)}")
    print(f"skill validation passed: {expected}")


if __name__ == "__main__":
    main()
