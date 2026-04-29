# Git Gnome

This repository's Git gnome keeps the tree tidy and reviewable.

## What it watches

- local secrets such as `.env`
- Python cache and virtualenv files
- editor-specific project files
- throwaway training outputs created inside the repo

## Ground rules

- keep source code, docs, and reproducible config in Git
- keep large generated artifacts and secrets out of Git
- prefer small, focused commits around one training, inference, or publishing change

## Before you commit

Run:

```bash
git status --short
```

If you see generated files that should stay untracked, extend `.gitignore` rather than committing them by accident.
