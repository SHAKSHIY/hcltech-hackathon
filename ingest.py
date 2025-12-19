## pdf ingestion
"""
Document Ingestion Pipeline for RAG System
Handles PDF extraction, chunking, embedding, and FAISS indexing
"""

import os
import pickle
from typing import List, Dict, Tuple
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss


class DocumentIngestionPipeline:
    """
    Manages the complete document ingestion workflow:
    PDF → Text → Chunks → Embeddings → FAISS Index
    """
    
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 600,
        chunk_overlap: int = 50,
        faiss_index_path: str = "faiss_index"
    ):
        """
        Initialize ingestion pipeline
        
        Args:
            embedding_model_name: HuggingFace model for embeddings
            chunk_size: Target tokens per chunk (500-800 range)
            chunk_overlap: Overlap between consecutive chunks
            faiss_index_path: Directory to save FAISS index
        """
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.faiss_index_path = faiss_index_path
        
        # Create directories if they don't exist
        os.makedirs(self.faiss_index_path, exist_ok=True)
        
        # Storage for chunks and metadata
        self.chunks = []
        self.metadata = []
    
    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, bool]:
        """
        Extract text from a single PDF file
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Tuple of (extracted_text, success_flag)
        """
        try:
            reader = PdfReader(pdf_path)
            text = ""
            
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}"
            
            # Validate that we extracted meaningful content
            if len(text.strip()) < 100:
                print(f"Warning: {pdf_path} contains minimal text (< 100 chars)")
                return "", False
            
            return text, True
            
        except Exception as e:
            print(f"Error extracting {pdf_path}: {str(e)}")
            return "", False
    
    def chunk_text(self, text: str, doc_name: str) -> List[Dict]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Full document text
            doc_name: Source document name for metadata
            
        Returns:
            List of chunk dictionaries with metadata
        """
        # Simple word-based chunking (approximates tokens)
        words = text.split()
        chunks = []
        
        start = 0
        while start < len(words):
            # Extract chunk
            end = start + self.chunk_size
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)
            
            # Only add non-empty chunks
            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text,
                    "source": doc_name,
                    "chunk_id": len(chunks)
                })
            
            # Move to next chunk with overlap
            start += (self.chunk_size - self.chunk_overlap)
        
        return chunks
    
    def process_pdfs(self, pdf_paths: List[str]) -> int:
        """
        Process multiple PDF files through the ingestion pipeline
        
        Args:
            pdf_paths: List of paths to PDF files
            
        Returns:
            Number of chunks successfully created
        """
        self.chunks = []
        self.metadata = []
        
        print(f"\n Processing {len(pdf_paths)} PDF(s)...")
        
        for pdf_path in pdf_paths:
            doc_name = os.path.basename(pdf_path)
            print(f"\n Extracting: {doc_name}")
            
            # Extract text
            text, success = self.extract_text_from_pdf(pdf_path)
            
            if not success:
                print(f" Skipping {doc_name} (extraction failed)")
                continue
            
            # Chunk the text
            doc_chunks = self.chunk_text(text, doc_name)
            print(f"Created {len(doc_chunks)} chunks from {doc_name}")
            
            # Store chunks and metadata
            for chunk in doc_chunks:
                self.chunks.append(chunk["text"])
                self.metadata.append({
                    "source": chunk["source"],
                    "chunk_id": chunk["chunk_id"]
                })
        
        print(f"\n Total chunks created: {len(self.chunks)}")
        return len(self.chunks)
    
    def generate_embeddings(self) -> np.ndarray:
        """
        Generate embeddings for all chunks using sentence-transformers
        
        Returns:
            Numpy array of embeddings (shape: [num_chunks, embedding_dim])
        """
        if not self.chunks:
            raise ValueError("No chunks available. Process PDFs first.")
        
        print(f"\n Generating embeddings for {len(self.chunks)} chunks...")
        embeddings = self.embedding_model.encode(
            self.chunks,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        print(f" Embeddings shape: {embeddings.shape}")
        
        return embeddings
    
    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.IndexFlatIP:
        """
        Build FAISS index with Inner Product (cosine similarity after normalization)
        
        Args:
            embeddings: Numpy array of embeddings
            
        Returns:
            FAISS index
        """
        # Normalize embeddings for cosine similarity via inner product
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        
        print(f" FAISS index built with {index.ntotal} vectors")
        return index
    
    def save_index(self, index: faiss.IndexFlatIP):
        """
        Persist FAISS index and metadata to disk
        
        Args:
            index: FAISS index to save
        """
        # Save FAISS index
        index_file = os.path.join(self.faiss_index_path, "index.faiss")
        faiss.write_index(index, index_file)
        
        # Save metadata and chunks
        metadata_file = os.path.join(self.faiss_index_path, "metadata.pkl")
        with open(metadata_file, "wb") as f:
            pickle.dump({
                "chunks": self.chunks,
                "metadata": self.metadata
            }, f)
        
        print(f" Index saved to {self.faiss_index_path}/")
    
    def ingest(self, pdf_paths: List[str]) -> bool:
        """
        Complete ingestion pipeline: PDFs → FAISS index
        
        Args:
            pdf_paths: List of PDF file paths
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Step 1: Process PDFs into chunks
            num_chunks = self.process_pdfs(pdf_paths)
            
            if num_chunks == 0:
                print(" No chunks created. Ingestion failed.")
                return False
            
            # Step 2: Generate embeddings
            embeddings = self.generate_embeddings()
            
            # Step 3: Build FAISS index
            index = self.build_faiss_index(embeddings)
            
            # Step 4: Save to disk
            self.save_index(index)
            
            print("\n Ingestion pipeline completed successfully!")
            return True
            
        except Exception as e:
            print(f" Ingestion failed: {str(e)}")
            return False


def main():
    """
    Example usage of the ingestion pipeline
    """
    # Initialize pipeline
    pipeline = DocumentIngestionPipeline()
    
    # Example: Process PDFs from a directory
    pdf_directory = "data"
    if os.path.exists(pdf_directory):
        pdf_files = [
            os.path.join(pdf_directory, f)
            for f in os.listdir(pdf_directory)
            if f.endswith(".pdf")
        ]
        
        if pdf_files:
            pipeline.ingest(pdf_files)
        else:
            print(f"No PDF files found in {pdf_directory}/")
    else:
        print(f"Directory {pdf_directory}/ does not exist")


if __name__ == "__main__":
    main()