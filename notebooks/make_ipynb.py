"""Convert notebooks/analysis.py ('# %%' cells) into analysis.ipynb."""
import os, re, nbformat as nbf
here = os.path.dirname(os.path.abspath(__file__)); src = open(os.path.join(here, "analysis.py")).read()
nb = nbf.v4.new_notebook()
for chunk in re.split(r"^# %%", src, flags=re.M)[1:]:
    md = chunk.startswith(" [markdown]"); body = chunk.split("\n", 1)[1] if "\n" in chunk else ""
    nb.cells.append(nbf.v4.new_markdown_cell("\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.strip().split("\n"))) if md else nbf.v4.new_code_cell(body.strip()))
nbf.write(nb, os.path.join(here, "analysis.ipynb")); print("wrote analysis.ipynb")
