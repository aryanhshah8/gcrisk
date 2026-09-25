# Submission package — Editorial Manager upload guide

Rebuilt to follow Aries/EM's own official process (`EM_PM_LaTeX_Guide.pdf`,
"Most journals accept a PDF of your manuscript at initial submission" branch —
confirmed applicable here since this journal offers a distinct "LaTeX Source
File" item type). Every file below has been compiled and verified locally.

| Folder / file | EM item type | What it is |
|---|---|---|
| `01_manuscript_pdf/paper.pdf` | **Manuscript** | The compiled PDF, built locally from the exact same source in the zip below. This is what reviewers actually read. |
| `latex_source.zip` | **LaTeX Source File** | **One single zip**, flat — no subfolders (EM explicitly cannot process subfolders in a zip; confirmed by the guide and by our own failed build). Contains `paper.tex`, `methodology.tex`, `discussion.tex`, `references.bib`, and all 7 figure PDFs, all at the same folder level. Verified: this exact flat file set compiles clean and produces a byte-for-byte text match to `paper.pdf`. **Do not add `cas-dc.cls`/`cas-common.sty`** — both are pre-installed on Aries' TeX Live (confirmed in the guide's installed-file list: `\latex\aries\elsevier\cas-dc.cls`), and including them is redundant. |
| `03_cover_letter/cover_letter.pdf` | **Cover Letter** | Unchanged. |
| `04_supplementary_material/supplementary.pdf` | **Supplementary Material** (or whatever the dropdown calls it — try "e-Component" if no explicit "Supplementary Material" option shows) | Sent as a compiled **PDF**, not `.tex`/`.cls` source. Supplementary material doesn't need to be typeset by EM's compiler the way the main manuscript does, so shipping the PDF directly avoids a second LaTeX-compile risk entirely. |
| `05_highlights/highlights.txt` | **Highlights** | 5 bullets, plain text. |
| `06_conflict_of_interest/declaration_of_competing_interests.docx` | **Conflict of Interest** | Real `.docx` (verified). EM's in-portal "declarations tool" questionnaire may generate its own — if so, use that instead and skip this file. |

## What changed from the earlier (wrong) approach

The first two build attempts failed because of a wrong mental model of EM's
item types — treating "LaTeX Source File" as a bucket for individually
uploaded loose files (methodology.tex, discussion.tex, cas-dc.cls, etc.), the
way a normal file-attachment UI works. That's not how EM's LaTeX compiler
works: per the official guide, when a journal accepts PDF at submission, all
LaTeX support files should be bundled into **one flat zip** and attached as a
single "LaTeX Source File" item — not several separate loose-file uploads.
Loose `.tex` files under that item type were apparently not being pulled into
the same compile directory as the Manuscript item, causing the
`File 'methodology.tex' not found` fatal error on the first attempt.

The `figures/` subdirectory prefix in `\includegraphics{figures/foo.pdf}`
was a second, independent bug — EM stores all attached/zipped files flat, so
any path prefix breaks image lookup. Both `paper.tex` copies here (in the zip,
and the one compiled into `paper.pdf`) use bare filenames with no `figures/`
prefix.

## Not included here (handle in-portal)

- **Conflict of Interest** — may come from EM's own in-portal declarations
  questionnaire instead of the docx above; check for that step first.
- **Title, abstract, keywords, author list** — enter manually in Manuscript
  Data; LaTeX submissions don't auto-populate these the way Word files do.

## Source of truth

The working LaTeX build (with `\input` paths, `figures/` subdirectory, and
`cas-dc.cls`/`cas-common.sty` for local compiling) lives one level up in
`paper/`. If the manuscript changes, regenerate this folder and the zip from
there rather than hand-editing copies here.
