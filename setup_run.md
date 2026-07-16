# VectorMind-AI — Setup & Run Guide

A local, offline-first RAG chatbot: upload PDFs/text, they get embedded into a
FAISS vector index, and a locally-running LLM (via **Ollama**) answers your
questions using the retrieved context — all wrapped in a Streamlit UI.

This guide takes a freshly cloned copy to a running app.

---

## 1. Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Python 3.10 – 3.11** | 3.11 is tested. Check with `python --version`. |
| **pip** | Comes with Python. |
| **Ollama** | Local LLM runtime — install from https://ollama.com/download |
| **Git** | To clone the repo. |
| **~4 GB free disk** | For the embedding model + at least one LLM. |

---

## 2. Clone the repository

```bash
git clone <your-repo-url>
cd VectorMind-AI
```

---

## 3. Create a virtual environment (recommended)

Keeping dependencies isolated avoids the kind of package collision that breaks
`fitz`/PyMuPDF (see Troubleshooting).

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 4. Install Python dependencies

```bash
pip install --upgrade pip
pip install -r req.txt
```

> ⚠️ **Do NOT run `pip install fitz`.** PyMuPDF already provides the `fitz`
> module. The standalone PyPI `fitz` package is an unrelated, broken project
> that shadows PyMuPDF and makes `import fitz` crash. `req.txt` intentionally
> lists only `PyMuPDF`.

---

## 5. Install & start Ollama, then pull a model

1. Install Ollama: https://ollama.com/download
2. Start the Ollama server (the app also tries to auto-start it, but doing it
   manually is more reliable):

   ```bash
   ollama serve
   ```

3. In a **separate** terminal, pull at least one supported model. The app knows
   about `llama3`, `mistral`, and `gemma`:

   ```bash
   ollama pull llama3
   # optional extras:
   ollama pull mistral
   ollama pull gemma
   ```

   Models you haven't pulled show up as 🔴 (not available) in the UI; pulled
   models show 🟢.

Verify the server is reachable:
```bash
curl http://localhost:11434/api/tags
```

---

## 6. (Optional) Seed the vector index

The repo ships with a prebuilt `faiss_index.index` + `doc_metadata.pkl`
containing a few demo sentences, so you can skip this. To rebuild that starter
index from scratch:

```bash
python index_builder.py
```

You don't need this step to use the app — you can also just upload your own
documents from the UI once it's running.

---

## 7. Run the app

**This is a Streamlit app — do NOT run `python app.py`.** Use:

```bash
python -m streamlit run app.py
```

or, if the `streamlit` command is on your PATH:

```bash
streamlit run app.py
```

Then open the URL it prints (default **http://localhost:8501**) in your browser.

To run headless / on a specific port:
```bash
python -m streamlit run app.py --server.headless true --server.port 8501
```

---

## 8. Using the app

1. **Choose a model** from the dropdown (🟢 = installed, 🔴 = not pulled).
2. **Upload a document** (`.pdf` or `.txt`) in the *Upload Document* section —
   it's chunked, embedded, and added to the FAISS index automatically.
3. **Ask a question** — the top-3 most relevant chunks are retrieved and sent to
   the LLM as context.
4. **Safe Prompt Mode** (sidebar) skips retrieved context and answers from the
   model's general knowledge only — useful for privacy-sensitive queries.
5. **Theme** toggle switches between light and dark.
6. Prior Q&A (with the retrieved context) is kept in **Chat History** for the
   session.

---

## 9. Troubleshooting

**`RuntimeError: Directory 'static/' does not exist` on `import fitz`**
The wrong `fitz` package is installed and shadowing PyMuPDF. Fix:
```bash
pip uninstall -y fitz
pip install --force-reinstall --no-deps PyMuPDF
```
Verify:
```bash
python -c "import fitz; print(fitz.__version__)"
```

**`streamlit: command not found` (exit 127)**
Streamlit isn't on your PATH. Run it as a module instead:
```bash
python -m streamlit run app.py
```

**App says "Ollama not running" / can't start it**
Start it manually in its own terminal and leave it running:
```bash
ollama serve
```

**"Selected model is not available in Ollama"**
Pull the model first: `ollama pull llama3` (or `mistral` / `gemma`).

**Slow first run**
On first launch the app downloads the `all-MiniLM-L6-v2` embedding model and the
`bert-base-uncased` tokenizer. This is a one-time download; later runs are faster.

---

## Quick reference

```bash
# one-time setup
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows
# source .venv/bin/activate         # macOS/Linux
pip install -r req.txt
ollama pull llama3

# every run
ollama serve                        # terminal 1 (leave running)
python -m streamlit run app.py      # terminal 2
```
