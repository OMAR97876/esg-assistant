"""Build storage/esg.index + storage/chunks.json from data/pdfs (run once, offline).

    pip install -r requirements-build.txt
    python scripts/build_index.py
"""
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import faiss
import numpy as np
import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from app.config import CHUNKS_PATH, EMBEDDING_MODEL, INDEX_PATH, STORAGE_DIR

PDF_DIR = BASE_DIR / 'data' / 'pdfs'
HEADERS = ['h1', 'h2', 'h3', 'h4']

header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[('#', 'h1'), ('##', 'h2'), ('###', 'h3'), ('####', 'h4')],
    strip_headers=False,
)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=180,
    separators=['\n\n', '\n', '. ', ' ', ''],
)


def iter_chunks(pdf_path):
    pages = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True, show_progress=False)
    for page_number, page in enumerate(pages, start=1):
        page_text = page.get('text', '').strip()
        if not page_text:
            continue
        for section in header_splitter.split_text(page_text):
            heading = ' > '.join(section.metadata[k] for k in HEADERS if section.metadata.get(k))
            for text in text_splitter.split_text(section.page_content):
                if text.strip():
                    yield {'text': text.strip(), 'source': pdf_path.name,
                           'page': page_number, 'heading': heading}


def main():
    pdf_files = sorted(PDF_DIR.glob('*.pdf'))
    if not pdf_files:
        raise ValueError(f'No PDFs found in {PDF_DIR}')

    chunks = []
    for pdf_path in pdf_files:
        print(f'Extracting: {pdf_path.name}', flush=True)
        chunks.extend(iter_chunks(pdf_path))
    if not chunks:
        raise ValueError('No chunks were created from the PDFs.')

    print(f'Embedding {len(chunks)} chunks...', flush=True)
    model = SentenceTransformer(EMBEDDING_MODEL, device='cpu')
    embeddings = np.asarray(model.encode(
        [f'passage: {c["text"]}' for c in chunks],
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ), dtype='float32')

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    CHUNKS_PATH.write_text(json.dumps(chunks, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    print(f'PDFs: {len(pdf_files)} | chunks: {len(chunks)} | vectors: {index.ntotal}')


if __name__ == '__main__':
    main()
