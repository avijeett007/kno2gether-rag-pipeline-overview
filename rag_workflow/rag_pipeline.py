#!/usr/bin/env python3
"""
RAG Pipeline Demo
This script demonstrates different RAG configurations:
1. Normal chunking vs Hierarchical chunking
2. Basic text loading vs LlamaParse
3. Different embedding options

Usage:
    python rag_pipeline.py [--with-hierarchical-chunking] [--with-llamaparse] [--embedding-type]
"""

import os
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path
from enum import Enum
import argparse
import textwrap

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
    Document,
    Response
)
from llama_index.core.node_parser import HierarchicalNodeParser, SentenceWindowNodeParser
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_parse import LlamaParse
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingType(Enum):
    """Types of embeddings we support"""
    OPENAI = "openai"
    LOCAL = "local"

class ChunkingStrategy(Enum):
    """Types of chunking strategies"""
    NORMAL = "normal"
    HIERARCHICAL = "hierarchical"

class RAGPipeline:
    def __init__(
        self,
        embedding_type: EmbeddingType = EmbeddingType.OPENAI,
        chunking_strategy: ChunkingStrategy = ChunkingStrategy.NORMAL,
        use_llama_parse: bool = False
    ):
        """
        Initialize the RAG pipeline with configurable options:
        1. Chunking strategy (normal or hierarchical)
        2. Document processing (basic or LlamaParse)
        3. Embedding type
        
        Args:
            embedding_type: Type of embeddings to use
            chunking_strategy: Type of chunking to use
            use_llama_parse: Whether to use LlamaParse for document processing
        """
        # Load environment variables
        load_dotenv()
        
        # Initialize embedding model based on type
        if embedding_type == EmbeddingType.OPENAI:
            logger.info("Using OpenAI embeddings")
            embed_model = OpenAIEmbedding(
                model_name="text-embedding-3-small",
                dimensions=1536,
                api_key=os.getenv("OPENAI_API_KEY")
            )
        else:
            logger.info("Using local FastEmbed embeddings")
            embed_model = FastEmbedEmbedding(
                model_name="BAAI/bge-small-en-v1.5"  # Good balance of speed and quality
            )
        
        # Initialize LLM for response synthesis
        llm = OpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Configure global settings
        Settings.llm = llm
        Settings.embed_model = embed_model
        
        # Initialize node parser based on strategy
        if chunking_strategy == ChunkingStrategy.HIERARCHICAL:
            logger.info("Using hierarchical chunking strategy")
            self.node_parser = HierarchicalNodeParser.from_defaults(
                chunk_sizes=[2048, 512, 128],
                chunk_overlap=50  # Moderate overlap for context
            )
        else:
            logger.info("Using normal chunking strategy")
            self.node_parser = SentenceWindowNodeParser.from_defaults(
                window_size=3,
                window_metadata_key="window",
                original_text_metadata_key="original_text"
            )
            Settings.chunk_size = 512
            Settings.chunk_overlap = 50
        
        # Initialize document parser
        if use_llama_parse:
            logger.info("Using LlamaParse for document processing")
            self.parser = LlamaParse(
                api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
                result_type="markdown"
            )
            self.file_extractors = {
                ".pdf": self.parser,
                ".docx": self.parser,
                ".pptx": self.parser,
                ".html": self.parser,
                ".png": self.parser,
                ".jpg": self.parser,
                ".jpeg": self.parser,
                ".txt": self.parser
            }
        else:
            logger.info("Using basic text processing")
            self.file_extractors = None
        
        # Set up storage directory with configuration
        config_str = f"{embedding_type.value}_{chunking_strategy.value}"
        if use_llama_parse:
            config_str += "_llamaparse"
        self.storage_dir = Path(f"rag_workflow/storage_{config_str}")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    def process_documents(self, docs_dir: str) -> VectorStoreIndex:
        """Process documents with the configured strategy"""
        logger.info(f"Processing documents from {docs_dir}")
        
        # Load documents
        reader = SimpleDirectoryReader(
            input_dir=docs_dir,
            file_extractor=self.file_extractors,
            filename_as_id=True
        )
        documents = reader.load_data()
        
        # Add metadata
        for doc in documents:
            doc.metadata.update({
                'created_at': datetime.utcnow().isoformat(),
                'file_type': Path(doc.metadata['file_path']).suffix,
                'file_name': Path(doc.metadata['file_path']).name,
                'char_count': len(doc.text),
                'word_count': len(doc.text.split())
            })
        
        # Create nodes
        nodes = self.node_parser.get_nodes_from_documents(documents)
        logger.info(f"Created {len(nodes)} nodes from {len(documents)} documents")
        
        # Create and persist index
        storage_context = StorageContext.from_defaults()
        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            show_progress=True
        )
        
        # Persist index
        index.storage_context.persist(persist_dir=str(self.storage_dir))
        logger.info(f"Index persisted to {self.storage_dir}")
        
        return index
    
    def load_index(self) -> VectorStoreIndex:
        """Load the persisted index"""
        if not self.storage_dir.exists():
            raise ValueError(f"No index found at {self.storage_dir}")
        
        storage_context = StorageContext.from_defaults(
            persist_dir=str(self.storage_dir)
        )
        return load_index_from_storage(storage_context)
    
    def query(
        self,
        query_text: str,
        index: Optional[VectorStoreIndex] = None,
        similarity_top_k: int = 3,
        rerank_top_n: int = 2
    ) -> str:
        """Query the knowledge base"""
        if index is None:
            index = self.load_index()
        
        try:
            # First try simple vector retrieval
            retriever = VectorIndexRetriever(
                index=index,
                similarity_top_k=similarity_top_k
            )
            
            # Get nodes and log
            nodes = retriever.retrieve(query_text)
            logger.info(f"Retrieved {len(nodes)} nodes")
            for i, node in enumerate(nodes):
                logger.info(f"Node {i+1} text: {textwrap.shorten(node.text, width=200)}...")
            
            if not nodes:
                return "I couldn't find any relevant information to answer your question."
            
            # Combine context
            context = "\n".join(f"[{i+1}] {node.text}" for i, node in enumerate(nodes))
            
            # Direct LLM query with structured prompt
            prompt = f"""You are a helpful AI assistant answering questions about Android's history and technical details.
            Below are relevant excerpts from documentation, numbered in order of relevance.
            
            {context}
            
            Based ONLY on the information provided above, answer this question: {query_text}
            
            If you cannot find a complete answer in the provided excerpts, say so.
            Cite the excerpt numbers [1], [2], etc. when using information from them.
            """
            
            logger.debug("Sending prompt to LLM...")
            response = Settings.llm.complete(prompt)
            
            if not response or not str(response).strip():
                # Fallback to simpler prompt
                logger.warning("Empty response, trying simpler prompt...")
                simple_prompt = f"Context: {context}\n\nQuestion: {query_text}\n\nAnswer:"
                response = Settings.llm.complete(simple_prompt)
            
            response_text = str(response).strip()
            if not response_text:
                return "I processed the information but couldn't generate a coherent response. Please try rephrasing your question."
                
            return response_text
            
        except Exception as e:
            logger.error(f"Error during query processing: {str(e)}")
            return f"An error occurred while processing your query: {str(e)}"

def main():
    """Main demo function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='RAG Pipeline Demo with different configurations')
    parser.add_argument('--with-hierarchical-chunking', action='store_true',
                      help='Use hierarchical chunking instead of normal chunking')
    parser.add_argument('--with-llamaparse', action='store_true',
                      help='Use LlamaParse for document processing')
    parser.add_argument('--embedding-type', type=str, choices=['openai', 'local'],
                      default='openai', help='Type of embeddings to use')
    args = parser.parse_args()
    
    # Set up chunking strategy
    chunking_strategy = (ChunkingStrategy.HIERARCHICAL 
                        if args.with_hierarchical_chunking 
                        else ChunkingStrategy.NORMAL)
    
    # Set up embedding type
    embedding_type = EmbeddingType.OPENAI if args.embedding_type == 'openai' else EmbeddingType.LOCAL
    
    # Initialize pipeline with configuration
    pipeline = RAGPipeline(
        embedding_type=embedding_type,
        chunking_strategy=chunking_strategy,
        use_llama_parse=args.with_llamaparse
    )
    
    # Process documents
    docs_dir = "rag_workflow/sample_docs"
    index = pipeline.process_documents(docs_dir)
    
    # Demo queries
    demo_queries = [
        "What functionalities are listed within the Linux Kernel block?",
        "When did Gingerbread v2.3.7 were released ?"
    ]
    
    # Print configuration
    logger.info("\n" + "="*80)
    logger.info(f"Configuration:")
    logger.info(f"- Chunking: {chunking_strategy.value}")
    logger.info(f"- Document Processing: {'LlamaParse' if args.with_llamaparse else 'Basic'}")
    logger.info(f"- Embeddings: {embedding_type.value}")
    logger.info("="*80 + "\n")
    
    # Run queries
    logger.info("Demo Queries:")
    for query in demo_queries:
        logger.info(f"\nQuery: {query}")
        response = pipeline.query(query, index)
        logger.info(f"Response: {response}")
        logger.info("-"*80)

if __name__ == "__main__":
    main()
