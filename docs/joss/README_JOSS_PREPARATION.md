# Mustatil JOSS enhanced preparation package

This package contains an improved JOSS paper and repository-readiness files. Copy files into the matching locations of the Mustatil GitHub repository **after reviewing them**.

## Most important files

- `paper/paper.md` — revised JOSS manuscript
- `paper/paper.bib` — checked bibliography with the 5.6 Zenodo DOI
- `JOSS_READINESS_REPORT.md` — honest pre-review assessment
- `SUBMISSION_CHECKLIST.md` — final gate checklist
- `REPOSITORY_LAYOUT.md` — recommended source layout
- `.github/workflows/draft-pdf.yml` — JOSS PDF build
- `.github/workflows/python-package.yml` — proposed Python test matrix

## Important limitation

The included tests and CI are scaffolding because this package does not contain the Mustatil source tree. Do not present them as evidence of comprehensive testing until real tests for tiling, georeferencing, de-duplication, and export have been implemented and pass in the public repository.

## Paper build

After pushing the files, open the repository's **Actions** tab and run **Draft JOSS paper**. Download the generated PDF artifact and inspect it before submission.
