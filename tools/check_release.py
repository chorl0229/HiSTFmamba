"""Check distributable Python/YAML files without importing GPU dependencies."""

import ast
from pathlib import Path
import re
from prepare_archives import source_files

root = Path(__file__).resolve().parents[1]
errors = []
count = 0
for path in source_files(root):
    relative = path.relative_to(root)
    if path.suffix not in {".py", ".yaml", ".yml"}:
        continue
    text = path.read_text()
    count += 1
    if path.suffix == ".py":
        try:
            ast.parse(text, filename=str(relative))
        except SyntaxError as exc:
            errors.append(f"{relative}: {exc}")
    if re.search(r"[\u4e00-\u9fff]", text):
        errors.append(f"{relative}: non-English source text")
    if re.search(r"(?<![A-Za-z0-9_.])/(?:home|data)/[A-Za-z][^ /\n]+/", text):
        errors.append(f"{relative}: machine-specific path")
if errors:
    raise SystemExit("\n".join(errors))
print(f"Passed source checks for {count} Python/YAML files.")
