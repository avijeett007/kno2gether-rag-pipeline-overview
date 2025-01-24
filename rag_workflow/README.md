# Configurable RAG Pipeline with LlamaIndex 

## Watch DeepDive Video

Watch the DeepDive Tutorial on Our YouTube Channel:

<p align="center">
    <a href="https://youtu.be/DYbwbs1sYa8">
        <img src="https://img.youtube.com/vi/DYbwbs1sYa8/0.jpg" alt="Ultimate RAG Pipeline Crash Course Tutorial" width="560" height="315">
    </a>
</p>

<p align="center">
    <a href="https://www.youtube.com/channel/UCxgkN3luQgLQOd_L7tbOdhQ?sub_confirmation=1">
        <img src="https://img.shields.io/badge/Subscribe-FF0000?style=for-the-badge&logo=youtube&logoColor=white" alt="Subscribe">
    </a>
</p>

## Introduction

This project demonstrates a flexible and configurable RAG (Retrieval-Augmented Generation) pipeline using LlamaIndex, featuring:
- Multiple chunking strategies (Normal vs Hierarchical)
- Flexible document processing (Basic vs LlamaParse)
- Choice of embeddings (OpenAI vs Local FastEmbed)
- Persistent storage of vector indices
- Advanced retrieval and reranking capabilities

## Key Features

- **Multiple Chunking Strategies**: 
  - Normal chunking with sentence windows
  - Hierarchical chunking with multiple chunk sizes
- **Flexible Document Processing**:
  - Basic text processing
  - Advanced LlamaParse support for multiple file formats (PDF, DOCX, PPTX, HTML, images)
- **Embedding Options**:
  - OpenAI embeddings (text-embedding-3-small)
  - Local FastEmbed (BAAI/bge-small-en-v1.5)
- **Advanced Retrieval**:
  - Configurable similarity search
  - Smart response synthesis
  - Source citation in responses

## Getting Started

1. Clone this repository:
```bash
git clone https://github.com/yourusername/rag-pipeline.git
cd rag-pipeline
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables in .env:
```bash
OPENAI_API_KEY='your_openai_api_key'
LLAMA_CLOUD_API_KEY='your_llama_parse_api_key'  # Optional, for LlamaParse
```

4. Run the pipeline:
```bash
python rag_pipeline.py [--with-hierarchical-chunking] [--with-llamaparse] [--embedding-type openai|local]
```

## Project Structure

- `rag_pipeline.py`: Main pipeline implementation
  - `RAGPipeline` class with configurable options
  - Document processing and indexing
  - Query handling and response generation
- `sample_docs/`: Directory for input documents
- `storage_*/`: Auto-generated storage for vector indices

## Core Components

### RAGPipeline Class
- Configurable initialization with embedding and chunking options
- Document processing with metadata enrichment
- Persistent storage of vector indices
- Smart query handling with source citations

### Document Processing
- Support for multiple file formats via LlamaParse
- Metadata enrichment (creation time, file type, statistics)
- Flexible chunking strategies

### Query Engine
- Vector similarity search
- Response synthesis with source citations
- Fallback mechanisms for robust responses

## Requirements

- Python 3.8+
- OpenAI API key
- LlamaParse API key (optional)
- Dependencies listed in requirements.txt

## Usage Examples

```python
# Initialize with OpenAI embeddings and hierarchical chunking
pipeline = RAGPipeline(
    embedding_type=EmbeddingType.OPENAI,
    chunking_strategy=ChunkingStrategy.HIERARCHICAL,
    use_llama_parse=True
)

# Process documents
index = pipeline.process_documents("sample_docs")

# Query the knowledge base
response = pipeline.query("What is the main topic of the document?")
```

## Error Handling

The system handles:
- Missing API keys
- Document processing errors
- Index loading failures
- Query processing issues

## Contributing

We welcome contributions! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

### Community and Support
- Join our community: [Kno2gether Community](https://community.kno2gether.com)
- Full Production Ready SaaS Launch Course (50% OFF): [End-to-End SaaS Launch Course](https://knolabs.biz/course-at-discount)

### Hosting Partners
- [Kamatera - Get $100 Free VPS Credit](https://knolabs.biz/100-dollar-free-credit)
- [Hostinger - Additional 20% Discount](https://knolabs.biz/20-Percent-Off-VPS)

## Video Tutorials

Follow along with our detailed video tutorials on the [Kno2gether YouTube Channel](https://youtube.com/@kno2gether) for step-by-step guidance and best practices.

## Conclusion

This RAG pipeline demonstrates a powerful and flexible approach to document processing and question answering, with multiple configuration options to suit different use cases.

Happy coding! 
