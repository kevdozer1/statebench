"""Write statebench_paper.md from statebench_paper.tex (the small subset of LaTeX the paper uses).

    python paper/tex2md.py
"""
from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEX = HERE / "statebench_paper.tex"
MD = HERE / "statebench_paper.md"

SIMPLE = [(r"\\%", "%"), (r"\\&", "&"), (r"\\_", "_"), (r"\\#", "#"), (r"\\\$", "$"), ("``", '"'), ("''", '"'), (r"---", "-"),
          (r"--", "–"), (r"\\times", "×"), (r"\\pm", "±"), (r"\\rightarrow", "→"), (r"\\to\b", "→"), (r"\\leq", "≤"),
          (r"\\geq", "≥"), (r"\\approx", "≈"), (r"\\sim", "~"), (r"\\ldots", "..."), (r"\\,", " "), (r"\\ ", " "),
          (r"\\textasciitilde", "~"), (r"\\noindent", ""), (r"\\centering", ""), (r"\\small", ""), (r"\\footnotesize", "")]


def braces(s: str, i: int) -> tuple[str, int]:
    """The {...} group starting at s[i] == '{' -> (content, index after the group)."""
    depth, j = 0, i
    while j < len(s):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    raise ValueError(s[i:i + 60])


def replace_cmd(s: str, name: str, fn) -> str:
    out, i = [], 0
    pat = "\\" + name + "{"
    while True:
        k = s.find(pat, i)
        if k < 0:
            out.append(s[i:])
            return "".join(out)
        out.append(s[i:k])
        arg, j = braces(s, k + len(pat) - 1)
        out.append(fn(arg))
        i = j


def main() -> None:
    t = TEX.read_text(encoding="utf-8")
    title = re.search(r"\\title\{\\textbf\{(.*?)\}\}", t, re.S).group(1).replace("\\\\", " ")
    body = t[t.index("\\begin{document}"):t.index("\\end{document}")]
    body = re.sub(r"(?m)^%.*$", "", body)

    # bibliography numbering (order of \bibitem) and label numbering (order of appearance)
    keys = re.findall(r"\\bibitem\[[^\]]*\]\{([^}]+)\}", body)
    num = {k: i + 1 for i, k in enumerate(keys)}
    labels, counters = {}, {"fig": 0, "tab": 0}
    for env, lab in re.findall(r"\\begin\{(figure|table)\}.*?\\label\{([^}]+)\}", body, re.S):
        kind = "fig" if env == "figure" else "tab"
        counters[kind] += 1
        labels[lab] = str(counters[kind])
    # section and subsection numbers, in document order; a \label right after a heading takes its number
    heads = re.compile(r"\\appendix|\\section(\*?)\{([^}]*)\}|\\subsection\{([^}]*)\}")
    sec = sub = 0
    appendix = False
    numbered = []  # (start, end, markdown heading)
    for m in heads.finditer(body):
        tok = m.group(0)
        if tok == "\\appendix":
            appendix, sec, sub = True, 0, 0
            numbered.append((m.start(), m.end(), ""))
            continue
        if tok.startswith("\\section"):
            if m.group(1):
                numbered.append((m.start(), m.end(), f"\n## {m.group(2)}\n"))
                continue
            sec, sub = sec + 1, 0
            n = chr(ord("A") + sec - 1) if appendix else str(sec)
            numbered.append((m.start(), m.end(), f"\n## {n}. {m.group(2)}\n"))
        else:
            sub += 1
            n = (chr(ord("A") + sec - 1) if appendix else str(sec)) + f".{sub}"
            numbered.append((m.start(), m.end(), f"\n### {n} {m.group(3)}\n"))
        lab = re.match(r"\s*\\label\{([^}]+)\}", body[m.end():])
        if lab:
            labels[lab.group(1)] = n
    out, pos = [], 0
    for a, b, rep in numbered:
        out.append(body[pos:a])
        out.append(rep)
        pos = b
    out.append(body[pos:])
    body = "".join(out)

    def cite(arg: str) -> str:
        return "[" + ", ".join(str(num[k.strip()]) for k in arg.split(",")) + "]"

    body = replace_cmd(body, "citep", cite)
    body = replace_cmd(body, "cite", cite)
    body = replace_cmd(body, "ref", lambda a: labels.get(a, a))

    # figures
    def figure(m):
        inner = m.group(1)
        img = re.search(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", inner).group(1)
        lab = re.search(r"\\label\{([^}]+)\}", inner)
        cap_start = inner.index("\\caption{") + len("\\caption")
        cap, _ = braces(inner, cap_start)
        n = labels[lab.group(1)] if lab else "?"
        return f"\n\n![Figure {n}](figures/{img})\n\n*Figure {n}. {cap.strip()}*\n\n"

    body = re.sub(r"\\begin\{figure\}(?:\[[^\]]*\])?(.*?)\\end\{figure\}", figure, body, flags=re.S)

    # tables
    def table(m):
        inner = m.group(1)
        lab = re.search(r"\\label\{([^}]+)\}", inner)
        cap_start = inner.index("\\caption{") + len("\\caption")
        cap, _ = braces(inner, cap_start)
        tab = re.search(r"\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}", inner, re.S).group(1)
        rows = [r.strip() for r in re.split(r"\\\\", tab)]
        rows = [re.sub(r"\\(toprule|midrule|bottomrule|hline)", "", r).strip() for r in rows]
        rows = [r for r in rows if r]
        cells = [[c.strip() for c in r.split("&")] for r in rows]
        md = ["| " + " | ".join(cells[0]) + " |", "|" + "---|" * len(cells[0])]
        md += ["| " + " | ".join(c) + " |" for c in cells[1:]]
        n = labels[lab.group(1)] if lab else "?"
        return f"\n\n*Table {n}. {cap.strip()}*\n\n" + "\n".join(md) + "\n\n"

    body = re.sub(r"\\begin\{table\}(?:\[[^\]]*\])?(.*?)\\end\{table\}", table, body, flags=re.S)

    # bibliography
    def bib(m):
        items = re.split(r"\\bibitem\[[^\]]*\]\{[^}]+\}", m.group(1))[1:]
        out = ["\n## References\n"]
        for i, it in enumerate(items, 1):
            it = re.sub(r"\\newblock", " ", it)
            out.append(f"[{i}] " + " ".join(it.split()))
        return "\n\n".join(out) + "\n"

    body = re.sub(r"\\begin\{thebibliography\}\{\d+\}(.*?)\\end\{thebibliography\}", bib, body, flags=re.S)

    body = re.sub(r"\\paragraph\{([^}]*)\}", r"**\1**", body)
    body = re.sub(r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
                  lambda m: "**Abstract.** " + m.group(1).replace("\\noindent", "").strip() + "\n", body, flags=re.S)
    for a, b in (('\\"o', "ö"), ('\\"u', "ü"), ('\\"a', "ä"), ("\\'e", "é")):
        body = body.replace(a, b)
    body = re.sub(r"\\begin\{(itemize|enumerate)\}", "", body)
    body = re.sub(r"\\end\{(itemize|enumerate)\}", "", body)
    body = re.sub(r"\\item\s*", "\n- ", body)
    code: list[str] = []

    def keep(m):
        code.append(m.group(1).strip("\n"))
        return f"\x00CODE{len(code) - 1}\x00"

    body = re.sub(r"\\begin\{verbatim\}(.*?)\\end\{verbatim\}", keep, body, flags=re.S)
    body = replace_cmd(body, "textbf", lambda a: f"**{a}**")
    body = replace_cmd(body, "emph", lambda a: f"*{a}*")
    body = replace_cmd(body, "textit", lambda a: f"*{a}*")
    body = replace_cmd(body, "texttt", lambda a: f"`{a}`")
    body = replace_cmd(body, "url", lambda a: f"<{a}>")
    body = re.sub(r"\\href\{([^}]*)\}\{([^}]*)\}", r"[\2](\1)", body)
    body = re.sub(r"\\label\{[^}]*\}", "", body)
    body = re.sub(r"\\(maketitle|clearpage|newpage|bibliographystyle\{[^}]*\})", "", body)
    body = body.replace("\\begin{document}", "")
    body = body.replace("{\\sloppy ", "").replace("\\par}", "").replace("{\\small", "")
    body = body.replace("\\$", "\x00DOLLAR\x00")  # escaped dollars are text, not math delimiters
    body = re.sub(r"\$\\pi_\{?([0-9.]+)\}?\$", r"π\1", body)
    body = re.sub(r"\$([^$]+)\$", lambda m: m.group(1).replace("\\", ""), body)
    body = body.replace("\x00DOLLAR\x00", "$")
    for a, b in SIMPLE:
        body = re.sub(a, b, body)
    body = re.sub(r"~", " ", body)
    for i, c in enumerate(code):
        body = body.replace(f"\x00CODE{i}\x00", "\n```\n" + c + "\n```\n")
    body = re.sub(r"\n{3,}", "\n\n", body)
    head = f"# {title}\n\nKevin Hopkins · kevinhop@usc.edu · September 2026\n\n"
    MD.write_text(head + body.strip() + "\n", encoding="utf-8", newline="\n")
    print("wrote", MD)


if __name__ == "__main__":
    main()
