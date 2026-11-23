import json
import threading
from functools import lru_cache

import faiss
import numpy as np
from groq import Groq

from app.config import (
    CHUNKS_PATH,
    EMBEDDING_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
    GROQ_TIMEOUT,
    INDEX_PATH,
    SIMILARITY_THRESHOLD,
    TOP_K,
)


SYSTEM_PROMPT = """
You are Verdia ESG Assistant.

Your role is to help with ESG, sustainability, climate, carbon accounting,
GHG emissions, Scope 1, Scope 2, Scope 3, CBAM, reporting, and related topics.

Rules:
- If the question is related to ESG or sustainability, answer it clearly and professionally.
- Use the provided document context whenever it is relevant.
- You may use your general knowledge to explain ESG and sustainability concepts when the documents do not contain enough information.
- If a question is not generally related to ESG or sustainability but the provided documents clearly contain the answer, answer using the documents.
- If the question is unrelated to ESG/sustainability and is not supported by the documents, politely say that you can only assist with ESG and sustainability-related topics.
- Never invent company-specific data, emissions, targets, reports, or calculations.
- Answer in the same language as the user when practical.
- Keep answers clear and easy to read.
- Do not mention file names, pages, similarity scores, chunks, retrieval, or internal system details.
- Do not add a Sources section.
"""

GREETINGS = {
    'hi', 'hello', 'hey',
    'اهلا', 'أهلا',
    'السلام عليكم', 'سلام',
    'شكرا', 'شكرًا',
    'thanks', 'thank you',
}

FOLLOW_UP_STARTS = (
    'and ', 'what about', 'how about', 'and what',
    'طيب', 'طب', 'وماذا', 'وما ', 'وايه', 'و ايه', 'وإيه',
)

client = Groq(api_key=GROQ_API_KEY, timeout=GROQ_TIMEOUT, max_retries=1)

# Heavy assets are loaded once, lazily (warm_up() is started at app startup).
_lock = threading.Lock()
_assets = {}


def _load():
    if not _assets:
        with _lock:
            if not _assets:
                from sentence_transformers import SentenceTransformer

                with CHUNKS_PATH.open('r', encoding='utf-8') as file:
                    _assets['chunks'] = json.load(file)
                _assets['index'] = faiss.read_index(str(INDEX_PATH))
                _assets['model'] = SentenceTransformer(EMBEDDING_MODEL, device='cpu')
    return _assets


def warm_up():
    _load()


@lru_cache(maxsize=512)
def _embed(question):
    vector = _load()['model'].encode(
        [f'query: {question}'],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return np.asarray(vector, dtype='float32')


def retrieve(question, k=TOP_K):
    assets = _load()
    index, chunks = assets['index'], assets['chunks']
    scores, indices = index.search(_embed(question), min(k, index.ntotal))

    return [
        {**chunks[idx], 'score': float(score)}
        for score, idx in zip(scores[0], indices[0])
        if idx != -1
    ]


def build_context(results):
    parts = []
    for i, result in enumerate(results, start=1):
        part = f'[Context {i}]\n'
        if result['heading']:
            part += f'Section: {result["heading"]}\n'
        parts.append(part + result['text'])
    return '\n\n'.join(parts)


def generate_answer(question, context='', previous_question=''):
    note = f'Previous user question:\n{previous_question}\n\n' if previous_question else ''
    user_prompt = f"""{note}Document context:
{context if context else 'No document context was retrieved.'}

Current user question:
{question}"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': user_prompt},
        ],
        temperature=0,
    )
    return response.choices[0].message.content


def is_greeting(question):
    return question.strip().lower() in GREETINGS


def is_follow_up(question):
    text = question.strip().lower()
    return len(text.split()) <= 5 or text.startswith(FOLLOW_UP_STARTS)


def ask_rag(question, history=None):
    if is_greeting(question):
        return generate_answer(question)

    previous_question = history[-1]['question'] if history else ''
    follow_up = bool(previous_question) and is_follow_up(question)
    search_question = f'{previous_question} {question}' if follow_up else question

    results = retrieve(search_question)
    if not results or results[0]['score'] < SIMILARITY_THRESHOLD:
        if any('؀' <= char <= 'ۿ' for char in question):
            return 'المعلومات المتاحة لدي لا تكفي للإجابة على هذا السؤال بثقة.'
        return "I don't have enough information to answer that confidently."

    return generate_answer(
        question,
        build_context(results),
        previous_question=previous_question if follow_up else '',
    )
