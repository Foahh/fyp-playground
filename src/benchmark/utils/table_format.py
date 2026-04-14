"""Shared table rendering: Rich (terminal) vs compact Markdown for LLM context."""

from __future__ import annotations

from typing import Final, Literal, Sequence

TableFormat = Literal["rich", "ai"]

TABLE_FORMAT_CHOICES: Final[tuple[str, ...]] = ("rich", "ai")


def normalize_table_format(raw: str) -> TableFormat:
    v = (raw or "").strip().casefold()
    if v not in TABLE_FORMAT_CHOICES:
        raise ValueError(
            f"invalid --table-format {raw!r}; expected one of: {', '.join(TABLE_FORMAT_CHOICES)}"
        )
    return v  # type: ignore[return-value]


def ai_escape_cell(s: str) -> str:
    """Single-line cell safe inside Markdown pipe tables."""
    t = str(s).replace("\n", " ").replace("\r", " ").strip()
    return t.replace("|", "\\|")


def ai_markdown_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    title: str | None = None,
) -> str:
    """GitHub-flavored Markdown pipe table (compact, no Rich box drawing)."""
    lines: list[str] = []
    if title:
        lines.append(f"### {ai_escape_cell(title)}")
        lines.append("")
    hs = [ai_escape_cell(str(h)) for h in headers]
    lines.append("| " + " | ".join(hs) + " |")
    lines.append("| " + " | ".join("---" for _ in hs) + " |")
    n = len(hs)
    for row in rows:
        cells = [str(c) for c in row]
        if len(cells) < n:
            cells = [*cells, *([""] * (n - len(cells)))]
        else:
            cells = cells[:n]
        lines.append("| " + " | ".join(ai_escape_cell(c) for c in cells) + " |")
    lines.append("")
    return "\n".join(lines)
