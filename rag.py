"""
RAG Query Engine
Handles retrieval from FAISS and answer generation using FLAN-T5
"""

import os
import pickle
from typing import List, Dict, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class RAGQueryEngine:
    """
    Retrieval-Augmented Generation engine for query answering
    """
    
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model_name: str = "google/flan-t5-base",
        faiss_index_path: str = "faiss_index",
        top_k: int = 3,
        max_context_length: int = 1024
    ):
        """
        Initialize RAG query engine
        
        Args:
            embedding_model_name: Model for query embeddings (must match ingestion)
            llm_model_name: LLM for answer generation
            faiss_index_path: Path to saved FAISS index
            top_k: Number of chunks to retrieve
            max_context_length: Maximum tokens for context (avoid overflow)
        """
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.top_k = top_k
        self.max_context_length = max_context_length
        self.faiss_index_path = faiss_index_path
        
        # Load FAISS index and metadata
        self.index = None
        self.chunks = []
        self.metadata = []
        self._load_index()
        
        # Load LLM and tokenizer
        print(" Loading FLAN-T5 model...")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.llm = AutoModelForSeq2SeqLM.from_pretrained(llm_model_name)
        
        # Use GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.llm.to(self.device)
        print(f" Model loaded on {self.device}")
    
    def _load_index(self):
        """
        Load FAISS index and metadata from disk
        """
        index_file = os.path.join(self.faiss_index_path, "index.faiss")
        metadata_file = os.path.join(self.faiss_index_path, "metadata.pkl")
        
        if not os.path.exists(index_file):
            print(f"  No FAISS index found at {index_file}")
            return
        
        # Load FAISS index
        self.index = faiss.read_index(index_file)
        
        # Load chunks and metadata
        with open(metadata_file, "rb") as f:
            data = pickle.load(f)
            self.chunks = data["chunks"]
            self.metadata = data["metadata"]
        
        print(f" Loaded index with {self.index.ntotal} vectors")
    
    def is_index_ready(self) -> bool:
        """
        Check if FAISS index is loaded and ready
        
        Returns:
            True if index exists and has vectors
        """
        return self.index is not None and self.index.ntotal > 0
    
    def retrieve(self, query: str) -> List[Dict]:
        """
        Retrieve top-K relevant chunks from FAISS
        
        Args:
            query: User query string
            
        Returns:
            List of retrieved chunks with metadata and scores
        """
        if not self.is_index_ready():
            raise ValueError("FAISS index not loaded. Please ingest documents first.")
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True
        )
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search FAISS index
        scores, indices = self.index.search(query_embedding, self.top_k)
        
        # Prepare results
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.chunks):  # Valid index
                results.append({
                    "text": self.chunks[idx],
                    "source": self.metadata[idx]["source"],
                    "chunk_id": self.metadata[idx]["chunk_id"],
                    "score": float(score)
                })
        
        return results
    
    def construct_prompt(self, query: str, retrieved_chunks: List[Dict]) -> str:
        """
        Build instruction-style prompt for FLAN-T5
        
        Args:
            query: User query
            retrieved_chunks: Retrieved context chunks
            
        Returns:
            Formatted prompt string
        """
        # Concatenate retrieved chunks
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks):
            context_parts.append(
                f"[Source: {chunk['source']}]\n{chunk['text']}"
            )
        
        context = "\n\n".join(context_parts)
        
        # Truncate context if too long
        context_tokens = self.tokenizer.encode(context, add_special_tokens=False)
        if len(context_tokens) > self.max_context_length:
            context_tokens = context_tokens[:self.max_context_length]
            context = self.tokenizer.decode(context_tokens)
        
        # Build instruction prompt
        prompt = f"""You are a knowledgeable assistant for Indian dietary guidelines and public health nutrition.

Context (from official dietary guidelines):
{context}

Question: {query}

Instructions:
- Answer ONLY based on the context provided above
- If the context does not contain relevant information, say "I cannot find information about this in the provided guidelines."
- Be concise and factual
- Do not add information not present in the context

Answer:"""
        
        return prompt
    
    def generate_answer(self, prompt: str) -> str:
        """
        Generate answer using FLAN-T5
        
        Args:
            prompt: Formatted prompt with context and query
            
        Returns:
            Generated answer string
        """
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=2048,
            truncation=True
        ).to(self.device)
        
        # Generate answer
        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_length=256,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
        
        # Decode answer
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer.strip()
    
    def query(self, user_query: str) -> Dict:
        """
        Complete RAG pipeline: Retrieve → Generate
        
        Args:
            user_query: User's question
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        # Validation
        if not user_query.strip():
            return {
                "answer": "Please provide a valid query.",
                "sources": [],
                "error": "Empty query"
            }
        
        if not self.is_index_ready():
            return {
                "answer": "No documents have been indexed yet. Please upload and ingest PDFs first.",
                "sources": [],
                "error": "Index not ready"
            }
        
        try:
            # Step 1: Retrieve relevant chunks
            print(f"\n Retrieving context for: '{user_query}'")
            retrieved = self.retrieve(user_query)
            
            if not retrieved:
                return {
                    "answer": "No relevant information found in the indexed documents.",
                    "sources": [],
                    "error": "No results"
                }
            
            print(f" Retrieved {len(retrieved)} chunks")
            
            # Step 2: Construct prompt
            prompt = self.construct_prompt(user_query, retrieved)
            
            # Step 3: Generate answer
            print(" Generating answer...")
            answer = self.generate_answer(prompt)
            
            # Extract unique sources
            sources = list(set([chunk["source"] for chunk in retrieved]))
            
            print(" Answer generated")
            
            return {
                "answer": answer,
                "sources": sources,
                "retrieved_chunks": retrieved,
                "error": None
            }
            
        except Exception as e:
            print(f" Query failed: {str(e)}")
            return {
                "answer": f"An error occurred: {str(e)}",
                "sources": [],
                "error": str(e)
            }


def main():
    """
    Example usage of RAG query engine
    """
    # Initialize engine
    engine = RAGQueryEngine()
    
    # Check if index is ready
    if not engine.is_index_ready():
        print(" No FAISS index found. Please run ingest.py first.")
        return
    
    # Example queries
    test_queries = [
        "What are the recommended daily servings of fruits and vegetables?",
        "What are the dietary guidelines for protein intake?",
        "How much water should adults drink per day?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        result = engine.query(query)
        print(f"\nAnswer: {result['answer']}")
        print(f"\nSources: {', '.join(result['sources'])}")


if __name__ == "__main__":
    main()