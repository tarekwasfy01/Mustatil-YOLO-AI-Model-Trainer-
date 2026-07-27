# JOSS readiness report for Mustatil

Assessment date: 27 July 2026

## Overall assessment

**Do not submit to JOSS yet.** The revised paper is within JOSS scope and format, but the public repository is the decisive part of review. The items below should be treated as release work, not as optional presentation improvements.

## Red — likely pre-review blockers

### 1. Source files must be directly browsable

The GitHub root currently presents historical ZIP archives and only a small number of directly visible source files. JOSS requires users and reviewers to browse the submitted source online, clone it, open issues, and propose changes. Unpack the exact maintained 5.6 source tree into normal repository folders. Keep old release ZIPs in GitHub Releases or Zenodo rather than as the main source representation.

### 2. Six months of public, iterative development

JOSS currently requires more than six months of public repository history with active development across that period. Confirm the repository's public date and inspect the commit distribution. If the public history began in May or June 2026, the earliest realistic submission is after the six-month threshold, provided development continues publicly rather than being uploaded in one later batch.

### 3. Objective tests and continuous integration

A minimal test scaffold and CI workflow are included here, but they do not yet test the scientific core. Add tests for raster tiling, pixel-to-map coordinate conversion, overlapping-detection handling, and GIS export. Add a small public reference dataset with known expected output.

## Yellow — substantial improvements before submission

### Packaging

PyPI currently offers a wheel but no source distribution. Publish an sdist from the same browsable source tree and use trusted publishing if practical. Verify that `pip install .`, `pip install -e .`, and the PyPI wheel all install the same code.

### Exact archival release

Create a tagged release from the exact commit submitted to JOSS and archive that release on Zenodo. The paper should cite that version-specific DOI, not merely an older software archive. Update `CITATION.cff`, `paper.bib`, and the submission metadata together.

### Community and support signals

The included `CONTRIBUTING.md`, `SUPPORT.md`, `GOVERNANCE.md`, issue templates, and pull-request template establish clear pathways. Publish them early enough that real issues, responses, and incremental maintenance are visible before submission.

### Research impact

The four cited datasets demonstrate repeated author-led research use across Saudi Arabia and Mongolia. This satisfies the minimum idea of developer use, but independent adoption would strengthen the case substantially. Ask at least one external researcher to install and test the software, document their feedback in a public issue, and—only with permission—mention genuine research use in the paper.

### AI disclosure

Verify the model names used in past development sessions. The revised disclosure states that historical ChatGPT model routing varied and names GPT-5.6 Thinking for final manuscript work. Replace this wording if more precise records are available.

## Green — existing strengths

- The software has an obvious research application.
- LGPL v3 is an OSI-approved open-source licence, provided the full licence text is present in `LICENSE`.
- Version 5.6 is installable from PyPI and archived on Zenodo.
- The project has multiple releases and several open research datasets.
- The paper now contains the required JOSS sections and a direct build-versus-contribute comparison.
- The local-first and human-in-the-loop design provides a defensible scholarly rationale.

## Recommended order of work

1. Put the complete maintained source tree directly in GitHub.
2. Make installation from a clean clone work.
3. Add synthetic reference data and tests for core geospatial transformations.
4. Enable CI on every push and pull request.
5. Add the community and governance files from this package.
6. Continue genuine public development for at least six months.
7. Obtain and document an external installation test.
8. Create a clean tagged release and Zenodo archive.
9. Compile the JOSS PDF with the included workflow.
10. Re-check every factual statement and submit only after all red items are closed.
