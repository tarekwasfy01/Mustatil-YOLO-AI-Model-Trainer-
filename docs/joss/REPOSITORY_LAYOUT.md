# Recommended repository layout

```text
Mustatil-YOLO-AI-Model-Trainer-/
├── src/
│   └── mustatil/               # Unpacked importable source package
├── tests/
├── paper/
│   ├── paper.md
│   └── paper.bib
├── docs/                        # User and developer documentation
├── examples/                    # Small redistributable sample project
├── .github/
│   ├── ISSUE_TEMPLATE/
│   └── workflows/
├── pyproject.toml
├── README.md
├── LICENSE
├── CITATION.cff
├── CHANGELOG.md
├── CONTRIBUTING.md
├── GOVERNANCE.md
├── SUPPORT.md
├── SECURITY.md
└── CODE_OF_CONDUCT.md
```

Do not blindly replace a working source layout. The essential requirement is that the maintained source be visible as ordinary files, that packaging metadata installs it from a clean clone, and that tests refer to the actual package structure.
