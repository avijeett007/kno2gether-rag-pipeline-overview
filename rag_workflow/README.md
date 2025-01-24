# RAG Pipeline Demo

This is a simple demonstration of a RAG (Retrieval Augmented Generation) pipeline that works with local files. It shows how to:
1. Load documents from your local filesystem
2. Process different file types (PDF, DOCX, TXT, etc.)
3. Create embeddings
4. Store vectors locally
5. Query the knowledge base

## Directory Structure
```
rag_workflow/
├── sample_docs/     # Put your documents here
├── storage/         # Vector store and index will be saved here
├── rag_pipeline.py  # Main pipeline code
└── README.md       # This file
```

## Supported File Types
- Documents: PDF (.pdf), Word (.docx), PowerPoint (.pptx)
- Web: HTML (.html)
- Images: PNG (.png), JPEG (.jpg, .jpeg)
- Text: Plain text (.txt)

## Setup

1. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install llama-index-core llama-parse llama-index-readers-file python-dotenv llama-index-llms-openai llama-index-embeddings-openai
```

3. Create a `.env` file in the root directory with your API keys:
```bash
OPENAI_API_KEY=your_openai_api_key
LLAMA_CLOUD_API_KEY=your_llama_parse_api_key
```

4. Put your documents in the `sample_docs` directory

## Usage

1. Run the pipeline:
```bash
python rag_pipeline.py
```

This will:
- Process all documents in `sample_docs/`
- Create embeddings using OpenAI's text-embedding-3-small model
- Store the vectors locally in `storage/`
- Run some demo queries

2. The script will output:
- Number of documents processed
- Number of nodes created
- Query results

## How It Works

1. **Document Loading**: Uses `SimpleDirectoryReader` to load documents from the filesystem

2. **Document Processing**: 
   - Files are processed based on their type using LlamaParse
   - Each document is chunked using hierarchical chunking
   - Metadata is added including creation time and file type

3. **Embedding Creation**:
   - Uses OpenAI's text-embedding-3-small model
   - Embeds each chunk of text into a 1536-dimensional vector

4. **Vector Storage**:
   - Vectors are stored locally in the `storage/` directory
   - No external vector database needed

5. **Querying**:
   - Creates a query engine from the index
   - Finds relevant chunks using vector similarity
   - Returns responses based on the matched chunks

## Customization

You can customize the pipeline by:
1. Modifying chunk sizes in `RAGPipeline.__init__`
2. Adding more file types to `file_extractors`
3. Changing the embedding model
4. Adding your own demo queries in `main()`

## Example

```python
from rag_pipeline import RAGPipeline

# Initialize pipeline
pipeline = RAGPipeline()

# Process documents
index = pipeline.process_documents("sample_docs")

# Query
response = pipeline.query("What are the key points discussed in the documents?")
print(response)
```
