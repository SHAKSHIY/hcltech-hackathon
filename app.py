## streamlit ui
"""
Streamlit UI for RAG-Powered Nutrition Assistant
Provides document upload and query interface
"""

import os
import streamlit as st
from ingest import DocumentIngestionPipeline
from rag import RAGQueryEngine


# Page configuration
st.set_page_config(
    page_title="Indian Dietary Guidelines Assistant",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
DATA_DIR = "data"
FAISS_INDEX_DIR = "faiss_index"

# Create directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if "rag_engine" not in st.session_state:
        st.session_state.rag_engine = None
    if "index_ready" not in st.session_state:
        st.session_state.index_ready = False
    if "query_history" not in st.session_state:
        st.session_state.query_history = []


def load_rag_engine():
    """Load or reload the RAG engine"""
    try:
        with st.spinner("Loading RAG engine..."):
            engine = RAGQueryEngine(faiss_index_path=FAISS_INDEX_DIR)
            st.session_state.rag_engine = engine
            st.session_state.index_ready = engine.is_index_ready()
            
            if st.session_state.index_ready:
                st.success(f" Ready! Index contains {engine.index.ntotal} vectors")
            else:
                st.warning("  No index found. Please upload and ingest documents.")
    except Exception as e:
        st.error(f" Failed to load RAG engine: {str(e)}")
        st.session_state.index_ready = False


def document_ingestion_tab():
    """UI for document upload and ingestion"""
    st.header(" Document Ingestion")
    
    st.markdown("""
    ### Instructions:
    1. Upload one or more PDF files containing Indian dietary guidelines
    2. Click "Ingest Documents" to process and index them
    3. Once ingestion is complete, you can query the documents
    """)
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload PDF Documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload official dietary guideline PDFs (e.g., ICMR-NIN guidelines)"
    )
    
    if uploaded_files:
        st.write(f"**{len(uploaded_files)} file(s) uploaded:**")
        for file in uploaded_files:
            st.write(f"- {file.name} ({file.size / 1024:.1f} KB)")
        
        # Ingest button
        if st.button(" Ingest Documents", type="primary", use_container_width=True):
            with st.spinner("Processing documents... This may take a few minutes."):
                # Save uploaded files
                pdf_paths = []
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(DATA_DIR, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.read())
                    pdf_paths.append(file_path)
                
                # Run ingestion pipeline
                pipeline = DocumentIngestionPipeline(faiss_index_path=FAISS_INDEX_DIR)
                
                # Capture output in expander
                with st.expander(" Ingestion Logs", expanded=True):
                    success = pipeline.ingest(pdf_paths)
                
                if success:
                    st.success(" Ingestion completed successfully!")
                    # st.balloons()
                    
                    # Reload RAG engine
                    load_rag_engine()
                    
                    # Show statistics
                    st.info(f"""
                    **Ingestion Summary:**
                    - Files processed: {len(pdf_paths)}
                    - Total chunks: {len(pipeline.chunks)}
                    - Embedding dimension: {pipeline.embedding_model.get_sentence_embedding_dimension()}
                    """)
                else:
                    st.error(" Ingestion failed. Check logs above.")
    
    # Show current index status
    st.divider()
    st.subheader(" Current Index Status")
    
    if st.session_state.index_ready and st.session_state.rag_engine:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Vectors", st.session_state.rag_engine.index.ntotal)
        with col2:
            st.metric("Total Chunks", len(st.session_state.rag_engine.chunks))
        with col3:
            st.metric("Embedding Model", "all-MiniLM-L6-v2")
    else:
        st.warning("No index found. Please upload and ingest documents first.")


def query_interface_tab():
    """UI for querying the RAG system"""
    st.header(" Query Assistant")
    
    # Check if index is ready
    if not st.session_state.index_ready:
        st.error(" No documents indexed. Please go to 'Document Ingestion' tab and upload PDFs first.")
        return
    
    st.markdown("""
    ### Ask questions about Indian dietary guidelines
    Examples:
    - What are the recommended daily servings of fruits and vegetables?
    - How much protein should adults consume daily?
    - What are the dietary recommendations for pregnant women?
    """)
    
    # Query input
    user_query = st.text_area(
        "Enter your question:",
        height=100,
        placeholder="Type your question here..."
    )
    
    # Query button
    col1, col2 = st.columns([3, 1])
    with col1:
        query_button = st.button(" Get Answer", type="primary", use_container_width=True)
    with col2:
        clear_button = st.button(" Clear History", use_container_width=True)
    
    if clear_button:
        st.session_state.query_history = []
        st.rerun()
    
    # Process query
    if query_button and user_query.strip():
        with st.spinner("Searching and generating answer..."):
            result = st.session_state.rag_engine.query(user_query)
        
        # Display answer
        st.divider()
        st.subheader(" Answer")
        
        if result["error"]:
            st.error(f"Error: {result['error']}")
        else:
            st.markdown(result["answer"])
            
            # Disclaimer
            st.warning("  **Disclaimer:** This information is for educational purposes only and not a substitute for professional medical or nutritional advice.")
            
            # Show sources
            if result["sources"]:
                st.divider()
                st.subheader(" Sources")
                for source in result["sources"]:
                    st.write(f"- {source}")
            
            # Show retrieved chunks in expander
            if "retrieved_chunks" in result and result["retrieved_chunks"]:
                with st.expander(" View Retrieved Context"):
                    for i, chunk in enumerate(result["retrieved_chunks"], 1):
                        st.markdown(f"""
                        **Chunk {i}** (Score: {chunk['score']:.3f})  
                        *Source: {chunk['source']}*
                        
                        {chunk['text'][:500]}...
                        """)
                        st.divider()
            
            # Add to history
            st.session_state.query_history.append({
                "query": user_query,
                "answer": result["answer"],
                "sources": result["sources"]
            })
    
    # Show query history
    if st.session_state.query_history:
        st.divider()
        st.subheader(" Query History")
        
        for i, item in enumerate(reversed(st.session_state.query_history[-5:]), 1):
            with st.expander(f"Q{len(st.session_state.query_history) - i + 1}: {item['query'][:80]}..."):
                st.markdown(f"**Question:** {item['query']}")
                st.markdown(f"**Answer:** {item['answer']}")
                if item['sources']:
                    st.markdown(f"**Sources:** {', '.join(item['sources'])}")


def main():
    """Main application entry point"""
    # Initialize session state
    initialize_session_state()
    
    # Load RAG engine on first run
    if st.session_state.rag_engine is None:
        load_rag_engine()
    
    # Header
    st.title(" Indian Dietary Guidelines Assistant")
    st.markdown("""
    ### RAG-Powered Nutrition Information System
    
    This application uses Retrieval-Augmented Generation to answer questions about Indian dietary guidelines 
    based on official documents (e.g., ICMR-NIN Dietary Guidelines).
    
    **Technology Stack:**
    - **Vector Store:** FAISS
    - **Embeddings:** Sentence Transformers (all-MiniLM-L6-v2)
    - **LLM:** FLAN-T5-Base
    - **Frontend:** Streamlit
    """)
    
    st.divider()
    
    # Tabs
    tab1, tab2, tab3 = st.tabs([" Document Ingestion", "💬 Query Assistant", "ℹ️ About"])
    
    with tab1:
        document_ingestion_tab()
    
    with tab2:
        query_interface_tab()
    
    with tab3:
        st.header(" About This Application")
        st.markdown("""
        ### Purpose
        This RAG system provides evidence-based answers to questions about Indian dietary guidelines 
        and public health nutrition by retrieving information from official PDF documents.
        
        ### How It Works
        1. **Document Ingestion:** PDFs are extracted, chunked, and embedded into a vector database (FAISS)
        2. **Query Processing:** User questions are embedded and matched against the vector database
        3. **Answer Generation:** Retrieved context is passed to FLAN-T5 to generate grounded answers
        
        ### Key Features
        -  No external API calls (runs locally)
        -  Answers grounded in official documents
        -  Source attribution for transparency
        -  No hallucination (LLM only uses provided context)
        
        ### Limitations
        - Answers are limited to the ingested documents
        - Does not provide personalized medical advice
        - Should not replace professional consultation
        
        ### Architecture
        - **Vector Database:** FAISS (Facebook AI Similarity Search)
        - **Embedding Model:** sentence-transformers/all-MiniLM-L6-v2
        - **Language Model:** google/flan-t5-base
        - **Chunking Strategy:** 600 tokens with 50-token overlap
        - **Retrieval:** Top-3 most similar chunks (cosine similarity)
        
        ---
        
        **Developed for:** Public Health Authority / Wellness Programs (India)  
        **Domain:** Public Health & Nutrition - Indian Dietary Guidelines
        """)


if __name__ == "__main__":
    main()