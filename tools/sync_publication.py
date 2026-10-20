"""Generate citation files from verified publication metadata; --check is read-only."""

import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
START = "<!-- BEGIN GENERATED CITATION -->"
END = "<!-- END GENERATED CITATION -->"
HEADER_START = "<!-- BEGIN GENERATED PAPER HEADER -->"
HEADER_END = "<!-- END GENERATED PAPER HEADER -->"


def render_header(paper):
    authors = ", ".join(f"{a['given']} {a['family']}" for a in paper["authors"])
    status = "Published at" if paper["status"] == "published" else "Accepted by"
    publisher = f" · [Publisher DOI](https://doi.org/{paper['doi']})" if paper.get("doi") else ""
    code = f" · [Code]({paper['repository_url']})" if paper.get("repository_url") else ""
    return (
        f"# {paper['name']} — {paper['venue']}\n\n"
        f"**{paper['title']}**\n\n{authors}\n\n"
        f"**{status} {paper['venue']}.**\n"
        f"[Paper](https://arxiv.org/abs/{paper['arxiv']}){publisher}{code} · "
        "[Citation](#citation) · [Setup](docs/SETUP.md) · "
        "[Training and evaluation](docs/USAGE.md)"
    )


def render(paper):
    """Keep GitHub's preferred paper citation and BibTeX in agreement."""
    quote = lambda value: json.dumps(value, ensure_ascii=False)
    authors = "\n".join(
        f"  - family-names: {quote(author['family'])}\n    given-names: {quote(author['given'])}"
        for author in paper["authors"]
    )
    url = f"https://arxiv.org/abs/{paper['arxiv']}"
    cff = (
        "cff-version: 1.2.0\n"
        'message: "If you use this code, please cite the conference paper below."\n'
        "type: software\n"
        f"title: {quote(paper['title'] + ' — Code')}\n"
        "license: MIT\n"
        f"authors:\n{authors}\n"
        "preferred-citation:\n"
        "  type: conference-paper\n"
        f"  title: {quote(paper['title'])}\n"
        "  authors:\n" + "\n".join("  " + line for line in authors.splitlines()) + "\n"
        f"  collection-title: {quote(paper['booktitle'])}\n"
        f"  year: {paper['year']}\n"
        f"  url: {quote(url)}\n"
    )
    if paper.get("doi"):
        cff += f"  doi: {quote(paper['doi'])}\n"
    if paper["status"] == "accepted":
        cff += f"  notes: {quote('Accepted at ' + paper['venue'])}\n"
    if paper.get("repository_url"):
        cff += f"repository-code: {quote(paper['repository_url'])}\n"
    author_text = " and ".join(
        f"{author['family']}, {author['given']}" for author in paper["authors"]
    )
    fields = {
        "title": paper["title"],
        "author": author_text,
        "booktitle": paper["booktitle"],
        "year": str(paper["year"]),
        "url": url,
    }
    if paper.get("doi"):
        fields["doi"] = paper["doi"]
    if paper["status"] == "accepted":
        fields["note"] = f"Accepted at {paper['venue']}"
    bib = (
        f"@inproceedings{{{paper['bibtex_key']},\n"
        + ",\n".join(f"  {key} = {{{value}}}" for key, value in fields.items())
        + "\n}\n"
    )
    return cff, bib


def synchronize(root, check=False):
    paper = json.loads((root / "docs/publication.json").read_text(encoding="utf-8"))
    cff, bib = render(paper)
    readme = (root / "README.md").read_text(encoding="utf-8")
    if readme.count(HEADER_START) != 1 or readme.count(HEADER_END) != 1:
        raise ValueError("README.md must contain exactly one generated paper header.")
    header = f"{HEADER_START}\n{render_header(paper)}\n{HEADER_END}"
    header_pattern = re.escape(HEADER_START) + ".*?" + re.escape(HEADER_END)
    readme, header_count = re.subn(header_pattern, lambda match: header, readme, flags=re.S)
    if header_count != 1:
        raise ValueError("README.md paper header markers are out of order.")
    if readme.count(START) != 1 or readme.count(END) != 1:
        raise ValueError("README.md must contain exactly one generated citation block.")
    pattern = re.escape(START) + ".*?" + re.escape(END)
    block = f"{START}\n```bibtex\n{bib}```\n{END}"
    updated, count = re.subn(pattern, lambda match: block, readme, flags=re.S)
    if count != 1:
        raise ValueError("README.md citation markers are out of order.")
    expected = {"CITATION.cff": cff, "CITATION.bib": bib, "README.md": updated}
    stale = []
    for name, content in expected.items():
        path = root / name
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            stale.append(name)
            if not check:
                path.write_text(content, encoding="utf-8")
    return stale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    stale = synchronize(ROOT, check=args.check)
    if args.check and stale:
        raise SystemExit(
            "Stale publication metadata: "
            + ", ".join(stale)
            + ". Run python tools/sync_publication.py."
        )
    print("Publication metadata is consistent." if args.check else "Citation files synchronized.")


if __name__ == "__main__":
    main()
