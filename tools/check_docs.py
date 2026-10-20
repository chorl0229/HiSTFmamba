"""Check local links in maintained Markdown without network or GPU dependencies."""

import html
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

from prepare_archives import source_files

ROOT = Path(__file__).resolve().parents[1]


def without_code(text):
    """Ignore examples inside fenced code blocks and inline code."""
    text = re.sub(r"^(`{3,}|~{3,})[^\n]*\n.*?^\1[ \t]*$", "", text, flags=re.M | re.S)
    return re.sub(r"`[^`\n]+`", "", text)


def heading_anchors(text):
    counts = {}
    anchors = set(re.findall(r'<a\s+(?:id|name)=["\']([^"\']+)', text))
    # Keep inline-code text: GitHub includes it in heading anchors.
    text = re.sub(r"^(`{3,}|~{3,})[^\n]*\n.*?^\1[ \t]*$", "", text, flags=re.M | re.S)
    for heading in re.findall(r"^#{1,6}\s+(.+?)\s*#*\s*$", text, flags=re.M):
        heading = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", heading)
        heading = html.unescape(re.sub(r"<[^>]+>", "", heading)).lower()
        slug = re.sub(r"[^\w\- ]", "", heading).replace(" ", "-")
        number = counts.get(slug, 0)
        counts[slug] = number + 1
        anchors.add(f"{slug}-{number}" if number else slug)
    return anchors


def check_document(path, root):
    text = without_code(path.read_text(encoding="utf-8"))
    # Maintained docs use inline links; support simple reference definitions too.
    links = re.findall(r"!?\[[^\]\n]*\]\(([^)\n]+)\)", text)
    links += re.findall(r"^\s*\[[^\]]+\]:\s*(\S+)", text, flags=re.M)
    errors = []
    for link in links:
        destination = link.split('"', 1)[0].strip().strip("<>")
        parsed = urlsplit(destination)
        if parsed.scheme or parsed.netloc:
            continue
        target = (path.parent / unquote(parsed.path)).resolve() if parsed.path else path
        try:
            target.relative_to(root.resolve())
        except ValueError:
            errors.append(f"{path.relative_to(root)}: link leaves repository: {destination}")
            continue
        if not target.exists():
            errors.append(f"{path.relative_to(root)}: missing link target: {destination}")
        elif parsed.fragment and target.suffix.lower() == ".md":
            if unquote(parsed.fragment) not in heading_anchors(target.read_text(encoding="utf-8")):
                errors.append(f"{path.relative_to(root)}: missing heading: {destination}")
    return errors


def main():
    errors = []
    count = 0
    for path in source_files(ROOT):
        if path.suffix.lower() != ".md" or path.relative_to(ROOT).parts[0] == "third_party":
            continue
        # Retained upstream dataset notes have their own upstream-relative links.
        if path.relative_to(ROOT).as_posix() == "src/data/humanml/README.md":
            continue
        count += 1
        errors.extend(check_document(path, ROOT))
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Local file links and heading anchors passed in {count} maintained Markdown files.")


if __name__ == "__main__":
    main()
