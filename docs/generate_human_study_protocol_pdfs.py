#!/usr/bin/env python3
"""Generate PDF versions of the Chinese and English human study protocols."""

from __future__ import annotations

from pathlib import Path

import markdown
from weasyprint import HTML


DOCS_DIR = Path(__file__).resolve().parent

SPECS = [
    {
        "src": DOCS_DIR / "human_study_protocol_zh.md",
        "out": DOCS_DIR / "IMPACT_Scribe_Human_Study_Protocol_ZH.pdf",
        "lang": "zh-CN",
        "title": "IMPACT-Scribe 人类测试操作说明",
    },
    {
        "src": DOCS_DIR / "human_study_protocol_en.md",
        "out": DOCS_DIR / "IMPACT_Scribe_Human_Study_Protocol_EN.pdf",
        "lang": "en",
        "title": "IMPACT-Scribe Human Study Protocol",
    },
]


CSS = """
@page {
  size: A4;
  margin: 20mm 18mm 20mm 18mm;
  @bottom-center {
    content: counter(page);
    font-size: 9pt;
    color: #667085;
  }
}
body {
  font-family: "Noto Sans CJK SC", "Noto Sans", "DejaVu Sans", sans-serif;
  font-size: 11pt;
  line-height: 1.62;
  color: #1f2937;
}
h1 {
  font-size: 24pt;
  color: #17212b;
  margin: 0 0 18px;
}
h2 {
  font-size: 16pt;
  color: #1f3b59;
  margin: 24px 0 8px;
  padding-bottom: 4px;
  border-bottom: 1px solid #d0d7de;
}
h3 {
  font-size: 12.5pt;
  color: #36516d;
  margin: 18px 0 6px;
}
p {
  margin: 6px 0 10px;
}
ul, ol {
  margin: 6px 0 12px 22px;
  padding: 0;
}
li {
  margin: 4px 0;
}
code {
  font-family: "DejaVu Sans Mono", "Noto Sans Mono", monospace;
  font-size: 9.5pt;
  background: #f3f4f6;
  padding: 1px 4px;
  border-radius: 4px;
}
strong {
  color: #111827;
}
"""


def _render_html(md_path: Path, title: str, lang: str) -> str:
    text = md_path.read_text(encoding="utf-8")
    body = markdown.markdown(
        text,
        extensions=["extra", "sane_lists", "smarty", "toc"],
        output_format="html5",
    )
    return f"""<!doctype html>
<html lang="{lang}">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>{CSS}</style>
</head>
<body>
{body}
</body>
</html>
"""


def main() -> None:
    for spec in SPECS:
        html = _render_html(spec["src"], spec["title"], spec["lang"])
        HTML(string=html, base_url=str(DOCS_DIR)).write_pdf(str(spec["out"]))
        print(f"wrote {spec['out']}")


if __name__ == "__main__":
    main()
