# Verdia ESG Assistant (Chatbot + RAG)

## Setup (once)
```bash
pip install -r requirements-build.txt
cp .env.example .env        # put GROQ_API_KEY
python scripts/build_index.py   # creates storage/esg.index + storage/chunks.json
```
Commit the `storage/` files so the server never has to rebuild the index.

## Run
```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

- `GET /health`
- `POST /api/esg/chat`  body: `{"question": "What is Scope 2?", "history": []}`
