"""
Document Ingestion Pipeline for RAG System
Robust PDF ingestion with OCR fallback
"""

import os
import pickle
from typing import List, Dict, Tuple

import numpy as np
import faiss
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
from sentence_transformers import SentenceTransformer


class DocumentIngestionPipeline:
    """
    PDF → Text → Chunks → Embeddings → FAISS
    Handles scanned + CID encoded PDFs using OCR fallback
    """

    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 600,
        chunk_overlap: int = 50,
        faiss_index_path: str = "faiss_index",
    ):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.faiss_index_path = faiss_index_path

        os.makedirs(self.faiss_index_path, exist_ok=True)

        self.chunks: List[str] = []
        self.metadata: List[Dict] = []

    # ---------------------------------------------------------
    # PDF TEXT EXTRACTION (TEXT → OCR FALLBACK)
    # ---------------------------------------------------------
    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, bool]:
        """
        Extract text from PDF.
        Uses OCR automatically if text is unreadable.
        """

        text = ""

        # ---------- Attempt 1: Normal text extraction ----------
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            print(f"[WARN] pdfplumber failed for {pdf_path}: {e}")

        # ---------- Detect unreadable / CID encoded text ----------
        unreadable = (
            len(text.strip()) < 200
            or "����" in text
            or "(cid:" in text
        )

        # ---------- OCR Fallback ----------
        if unreadable:
            print(f"[INFO] OCR fallback triggered for {os.path.basename(pdf_path)}")
            text = ""
            try:
                images = convert_from_path(pdf_path)
                for img in images:
                    text += pytesseract.image_to_string(img, lang="eng") + "\n"
            except Exception as e:
                print(f"[ERROR] OCR failed for {pdf_path}: {e}")
                return "", False

        # ---------- Final validation ----------
        if len(text.strip()) < 200:
            print(f"[ERROR] No readable text found in {pdf_path}")
            return "", False

        return text, True

    # ---------------------------------------------------------
    # CHUNKING
    # ---------------------------------------------------------
    def chunk_text(self, text: str, source: str) -> List[Dict]:
        """
        Split text into overlapping chunks
        """
        words = text.split()
        chunks = []

        start = 0
        chunk_id = 0

        while start < len(words):
            end = start + self.chunk_size
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text,
                    "source": source,
                    "chunk_id": chunk_id
                })
                chunk_id += 1

            start += self.chunk_size - self.chunk_overlap

        return chunks

    # ---------------------------------------------------------
    # PDF PROCESSING
    # ---------------------------------------------------------
    def process_pdfs(self, pdf_paths: List[str]) -> int:
        """
        Extract + chunk multiple PDFs
        """
        self.chunks = []
        self.metadata = []

        print(f"\n[INFO] Processing {len(pdf_paths)} PDF(s)")

        for pdf_path in pdf_paths:
            filename = os.path.basename(pdf_path)
            print(f"\n[INFO] Extracting: {filename}")

            text, success = self.extract_text_from_pdf(pdf_path)
            if not success:
                print(f"[SKIP] {filename}")
                continue

            doc_chunks = self.chunk_text(text, filename)
            print(f"[INFO] {len(doc_chunks)} chunks created from {filename}")

            for chunk in doc_chunks:
                self.chunks.append(chunk["text"])
                self.metadata.append({
                    "source": chunk["source"],
                    "chunk_id": chunk["chunk_id"]
                })

        print(f"\n[INFO] Total chunks created: {len(self.chunks)}")
        return len(self.chunks)

    # ---------------------------------------------------------
    # EMBEDDINGS
    # ---------------------------------------------------------
    def generate_embeddings(self) -> np.ndarray:
        if not self.chunks:
            raise ValueError("No chunks available for embedding.")

        print(f"\n[INFO] Generating embeddings...")
        embeddings = self.embedding_model.encode(
            self.chunks,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings

    # ---------------------------------------------------------
    # FAISS INDEX
    # ---------------------------------------------------------
    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.IndexFlatIP:
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        print(f"[INFO] FAISS index built with {index.ntotal} vectors")
        return index

    def save_index(self, index: faiss.IndexFlatIP):
        faiss.write_index(index, os.path.join(self.faiss_index_path, "index.faiss"))

        with open(os.path.join(self.faiss_index_path, "metadata.pkl"), "wb") as f:
            pickle.dump({
                "chunks": self.chunks,
                "metadata": self.metadata
            }, f)

        print(f"[INFO] Index saved to {self.faiss_index_path}/")

    # ---------------------------------------------------------
    # FULL PIPELINE
    # ---------------------------------------------------------
    def ingest(self, pdf_paths: List[str]) -> bool:
        try:
            num_chunks = self.process_pdfs(pdf_paths)
            if num_chunks == 0:
                return False

            embeddings = self.generate_embeddings()
            index = self.build_faiss_index(embeddings)
            self.save_index(index)

            print("\n[SUCCESS] Ingestion completed successfully")
            return True

        except Exception as e:
            print(f"[ERROR] Ingestion failed: {e}")
            return False


# ---------------------------------------------------------
# STANDALONE RUN
# ---------------------------------------------------------
if __name__ == "__main__":
    pipeline = DocumentIngestionPipeline()
    data_dir = "data"

    pdfs = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith(".pdf")
    ]

    if not pdfs:
        print("[ERROR] No PDFs found")
    else:
        pipeline.ingest(pdfs)