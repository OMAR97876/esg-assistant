import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.rag_engine import ask_rag, warm_up
from app.schemas import ChatRequest, ChatResponse


@asynccontextmanager
async def lifespan(_app):
    # Load the model/index in the background so the server is up immediately.
    threading.Thread(target=warm_up, daemon=True).start()
    yield


app = FastAPI(
    title='Verdia ESG Assistant API',
    description=(
        'Evidence-grounded ESG assistant powered by the fixed Verdia knowledge base. '
        'Ask about GHG accounting, Scope 1/2/3, CBAM, and related ESG topics.'
    ),
    version='1.1.0',
    docs_url='/docs',
    redoc_url=None,
    lifespan=lifespan,
)


@app.get('/health', tags=['System'], summary='Health Check')
def health():
    return {'status': 'ok', 'service': 'Verdia ESG Assistant'}


def _chat(request: ChatRequest):
    history = [turn.model_dump() for turn in request.history]
    return {'answer': ask_rag(request.question, history)}


@app.post(
    '/api/esg/chat',
    response_model=ChatResponse,
    tags=['ESG Assistant'],
    summary='Ask Verdia ESG Assistant',
)
def esg_chat(request: ChatRequest):
    return _chat(request)


# Backward-compatible alias, hidden from Swagger.
@app.post('/api/rag/chat', response_model=ChatResponse, include_in_schema=False)
def rag_chat(request: ChatRequest):
    return _chat(request)
