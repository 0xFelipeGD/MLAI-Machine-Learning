# Contributing to MLAI

Thank you for considering a contribution! This file explains how to set up a dev environment and submit changes.

## Code of Conduct

Be kind. Help newcomers. Assume good faith.

## Development Setup

```bash
# 1. Clone
git clone https://github.com/<you>/MLAI-Machine-Learning.git
cd MLAI-Machine-Learning

# 2. Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pytest

# 3. Frontend
cd web
npm install
cd ..
```

## Running Locally (PC, no Pi required)

The engine and pipeline fall back to **mock mode** when no TFLite model files or camera are present, so you can run the entire stack on a regular development machine:

```bash
# Terminal A — engine
python -m engine.main

# Terminal B — API
python -m uvicorn api.main:app --reload

# Terminal C — frontend
cd web && npm run dev
```

Then open <http://localhost:3000>.

## Tests

```bash
pytest tests/
```

Frontend type checks:

```bash
cd web && npm run type-check
```

## Pull Request Checklist

- [ ] Code is formatted and lint-clean
- [ ] New behavior is covered by a test where reasonable
- [ ] `pytest` passes
- [ ] `npm run type-check` passes
- [ ] No new dependencies added without discussion
- [ ] Touched ML code? Confirm a beginner could read it

## Reporting Issues

Open a GitHub issue with:
- Pi model + RAM
- OS version
- `pip freeze` output
- Steps to reproduce
- Logs from `journalctl -u mlai-api -u mlai-web`
