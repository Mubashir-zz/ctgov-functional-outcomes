# -*- coding: utf-8 -*-
"""Regenerate figures from the current tables and replace the manuscript's tables,
captions and images so the document describes one release throughout."""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import docx

os.makedirs("figures", exist_ok=True)
LAB = {"pain":"Pain","fatigue":"Fatigue","physical_function":"Physical function","bowel":"Bowel",
 "cognition":"Cognition","urinary":"Urinary","respiratory":"Respiratory","neuropathy":"Neuropathy",
 "sexual_reproductive":"Sexual/reproductive","swallowing":"Swallowing",
 "limb_lymphedema":"Limb/lymphoedema","speech_voice":"Speech/voice","visual":"Vision",
 "hearing":"Hearing","any_domain":"Any domain"}

def read(p):
    with open(p) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))

# ---------------- Figure 1
t1 = [r for r in read("tables/table1.tsv") if r["Domain"] != "any_domain"]
t1.sort(key=lambda r: float(r["Observed %"]))
y = range(len(t1))
obs = [float(r["Observed %"]) for r in t1]
cor = [float(r["Corrected %"]) for r in t1]
lo = [float(r["95% credible"].split("-")[0]) for r in t1]
hi = [float(r["95% credible"].split("-")[1]) for r in t1]
fig, ax = plt.subplots(figsize=(7.2, 5.4))
for i, r in enumerate(t1):
    ax.plot([lo[i], hi[i]], [i, i], color="#4C6EF5", lw=2.2, alpha=.55, solid_capstyle="round")
ax.scatter(obs, y, s=42, facecolor="white", edgecolor="#868E96", zorder=3, label="Observed")
ax.scatter(cor, y, s=44, color="#364FC7", zorder=4, label="Corrected")
ax.set_yticks(list(y)); ax.set_yticklabels([LAB[r["Domain"]] for r in t1], fontsize=9)
ax.set_xlabel("Trials registering the domain (%)", fontsize=9)
ax.grid(axis="x", alpha=.25); ax.set_axisbelow(True)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=9, loc="lower right")
ax.set_title("Observed and corrected registration, 93,371 records",
             fontsize=10, loc="left")
fig.tight_layout(); fig.savefig("figures/figure1.png", dpi=200); plt.close(fig)

# ---------------- Figure 2
t3 = [r for r in read("tables/table3.tsv") if r["Standardised rate ratio"] != "not estimable"]
t3.sort(key=lambda r: float(r["Standardised rate ratio"]))
fig, ax = plt.subplots(figsize=(7.8, 4.8))
colour = {"primary": "#364FC7", "secondary": "#4C6EF5",
          "negative control": "#868E96",
          "negative control (post hoc excluded)": "#ADB5BD"}
for i, r in enumerate(t3):
    rr = float(r["Standardised rate ratio"])
    a, b = [float(x) for x in r["95% CI (BCa)"].split("-")]
    c = colour.get(r["Pair"], "#868E96")
    ax.plot([a, b], [i, i], color=c, lw=2.2, alpha=.6, solid_capstyle="round")
    ax.scatter([rr], [i], s=46, color=c, zorder=3)
ax.axvline(1.0, color="#212529", lw=1, ls="--", alpha=.6)
ax.set_xscale("log")
ax.set_yticks(range(len(t3)))
ax.set_yticklabels([f"{r['Agent']} – {LAB.get(r['Domain'], r['Domain'])} ({r['Measuring']})"
                    for r in t3], fontsize=8.5)
ax.set_xlabel("Standardised rate ratio against matched comparator strata (log scale)", fontsize=9)
ax.grid(axis="x", alpha=.25); ax.set_axisbelow(True)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.set_title("Agent–domain alignment, therapeutic cohort (events in brackets)",
             fontsize=10, loc="left")
fig.tight_layout(); fig.savefig("figures/figure2.png", dpi=200); plt.close(fig)
print("figures regenerated")

# ---------------- manuscript tables, captions, images
d = docx.Document("MANUSCRIPT.docx")

def replace_table(old, rows):
    new = d.add_table(rows=len(rows), cols=len(rows[0]))
    try:
        new.style = old.style
    except Exception:
        pass
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            new.cell(i, j).text = str(val)
    old._tbl.addprevious(new._tbl)
    old._tbl.getparent().remove(old._tbl)

def as_rows(path, header_map=None):
    rd = list(csv.reader(open(path), delimiter="\t"))
    if header_map:
        rd[0] = header_map
    for r in rd[1:]:
        r[0] = LAB.get(r[0], r[0])
    return rd

t = d.tables
replace_table(t[0], as_rows("tables/table1.tsv"))
replace_table(t[1], as_rows("tables/table2.tsv"))
rows3 = list(csv.reader(open("tables/table3.tsv"), delimiter="\t"))
for r in rows3[1:]:
    r[2] = LAB.get(r[2], r[2])
replace_table(t[2], rows3)

CAP = [
 ("96,936 ClinicalTrials.gov cancer-trial records", "93,371 ClinicalTrials.gov cancer-trial records"),
 ("across 96,936 ClinicalTrials.gov", "across 93,371 ClinicalTrials.gov"),
 ("(n = 55,178)", "(n = 51,227)"),
 ("The 96,936-record corpus", "The 93,371-record corpus"),
 ("Sponsor class and paediatric patterns: two cautions about composition",
  "Sponsor class: a caution about composition"),
]
for p in d.paragraphs:
    for a, b in CAP:
        if a in p.text:
            txt = p.text.replace(a, b)
            for r in list(p.runs)[1:]:
                r._element.getparent().remove(r._element)
            if p.runs:
                p.runs[0].text = txt
            else:
                p.add_run(txt)

imgs = sorted([(rid, part) for rid, part in d.part.related_parts.items()
               if "image" in part.content_type], key=lambda kv: kv[1].partname)
print("embedded images:", [str(p.partname) for _, p in imgs])
new_blobs = [open("figures/figure1.png", "rb").read(), open("figures/figure2.png", "rb").read()]
for (rid, part), blob in zip(imgs, new_blobs):
    part._blob = blob
d.save("MANUSCRIPT.docx")
print("manuscript tables, captions and figures replaced")
