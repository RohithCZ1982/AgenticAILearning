# Agentic AI Learning Demo

## What this project is
A very small Python application that teaches someone new to AI what an
"agentic AI" is, by being one. The user types a comment, and the app:

1. Classifies it as positive or negative using a HuggingFace sentiment model.
2. Stores the comment in a vector database (ChromaDB).
3. Recalls and shows similar comments it has seen before.

This loop of "perceive -> reason with a model -> remember -> recall" is what
makes the app "agentic" rather than a one-shot script.

## Files
- [app.py](app.py) — the entire application (single file, intentionally small).
- [requirements.txt](requirements.txt) — Python dependencies (`transformers`, `torch`, `chromadb`).
- [agentic_ai_demo_colab.ipynb](agentic_ai_demo_colab.ipynb) — the same app as a Google Colab notebook (see "Running it" below — needed because `chromadb` won't build locally on Windows ARM64).
- [EXPLANATION.md](EXPLANATION.md) — line-by-line walkthrough of `app.py` for AI newcomers.
- `vector_db/` — created automatically at runtime; holds the persisted ChromaDB data. Safe to delete to reset memory.

## Running it

### Locally (x86-64 machines)
```
pip install -r requirements.txt
python app.py
```
The first run downloads the HuggingFace sentiment model and Chroma's default
embedding model, so it needs an internet connection and may take a minute.

### Via Google Colab (recommended on Windows ARM64)
On this machine, `pip install -r requirements.txt` fails: `chromadb` depends on
`grpcio`/`httptools`, which have no prebuilt wheels for Windows ARM64, so pip
tries to compile them from source and fails. Rather than fight that toolchain,
upload [agentic_ai_demo_colab.ipynb](agentic_ai_demo_colab.ipynb) to
https://colab.research.google.com (File → Upload notebook), then Runtime → Run
all. Colab runs on x86-64 Linux where `chromadb`'s dependencies install
cleanly, and its cells mirror `app.py` line for line — `input()` prompts appear
as interactive boxes under the last cell.

## Design notes / constraints
- Kept to a single script on purpose — this is a teaching example, not a
  production app. Don't split it into modules/packages or add config layers.
- Sentiment model: `distilbert-base-uncased-finetuned-sst-2-english` (HuggingFace default for the `sentiment-analysis` pipeline).
- Vector DB: ChromaDB `PersistentClient`, stored locally in `./vector_db`, using Chroma's built-in default embedding function (no separate embedding model to wire up).
- No web framework/UI — interaction is via the terminal `input()` prompt, matching the "very small" brief.

## If extending this
- Keep explanations beginner-friendly; this app's audience is people new to AI.
- If you change `app.py`, update [EXPLANATION.md](EXPLANATION.md) to match — it documents the code line by line.
