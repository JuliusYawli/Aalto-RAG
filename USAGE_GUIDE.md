# Aalto-RAG Usage Guide

## Quick Start Guide

### Step 1: Installation

```bash
# Install Python dependencies
pip install -r requirements.txt
```

### Step 2: Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=sk-your-api-key-here
```

Get your API key from: https://platform.openai.com/api-keys

### Step 3: Index Documents

The repository comes with sample documents. Index them:

```bash
python main.py --index
```

This will:
- Load documents from the `documents/` directory
- Split them into chunks
- Create embeddings using OpenAI
- Store in a local ChromaDB vector store

### Step 4: Ask Questions

**Interactive Mode** (Recommended for exploration):
```bash
python main.py --interactive
```

**Single Question Mode**:
```bash
python main.py --question "What is RAG and how does it work?"
```

## Example Questions

Try these with the included sample documents:

1. "What is Retrieval-Augmented Generation?"
2. "What are some machine learning best practices?"
3. "How do transformers work in NLP?"
4. "What are the benefits of using RAG systems?"
5. "How can I prevent overfitting in machine learning?"
6. "What is the difference between BERT and GPT?"

## Using Your Own Documents

1. Add your documents to the `documents/` directory:
   - Supported formats: TXT, PDF, DOCX
   - You can organize in subdirectories

2. Re-index:
```bash
python main.py --index
```

3. Start asking questions about your documents!

## Using Custom Document Directory

```bash
# Index from a custom directory
python main.py --index --documents-path /path/to/your/documents

# Then use normally
python main.py --interactive
```

## Using the Jupyter Notebook

For a more interactive experience:

```bash
jupyter notebook rag_notebook.ipynb
```

Or open in Google Colab:
1. Go to https://colab.research.google.com/
2. File → Upload notebook
3. Upload `rag_notebook.ipynb`
4. Follow the cells step by step

## Tips for Best Results

### Document Quality
- Use clear, well-structured documents
- Remove unnecessary formatting or headers
- Keep documents focused on specific topics

### Chunking
- Default chunk size is 1000 characters
- Adjust `CHUNK_SIZE` in `src/config.py` if needed
- Larger chunks = more context, but less precise retrieval
- Smaller chunks = more precise, but may lose context

### Number of Retrieved Documents
- Default is 4 documents (TOP_K_RESULTS)
- Increase for complex questions requiring more context
- Decrease for simple questions or to reduce costs

### Model Selection
- **GPT-3.5-turbo**: Fast, cost-effective, good for most cases
- **GPT-4**: More accurate, better reasoning, but slower and more expensive
- Change in `.env` or `src/config.py`

## Troubleshooting

### "OPENAI_API_KEY not found" Error
- Make sure you created the `.env` file
- Check that your API key is correctly set
- Verify no extra spaces in the `.env` file

### "Vector store not found" Error
- Run `python main.py --index` first
- Make sure indexing completed successfully
- Check that `vectorstore/` directory was created

### "No documents found" Error
- Ensure documents are in the `documents/` directory
- Check file extensions (.txt, .pdf, .docx)
- Verify files are not empty

### High API Costs
- Use GPT-3.5-turbo instead of GPT-4
- Reduce TOP_K_RESULTS to retrieve fewer documents
- Cache frequently asked questions
- Consider using smaller chunk sizes

### Slow Performance
- First query is slower (loads the vector store)
- Subsequent queries are much faster
- For production, keep the application running

## Advanced Usage

### Programmatic Usage

```python
from src import Config, DocumentLoader, VectorStore, RAGChain

# Initialize
Config.validate()
doc_loader = DocumentLoader()
vector_store = VectorStore(Config.VECTORSTORE_PATH)

# Load documents
documents = doc_loader.load_directory('documents/')

# Create vector store
vector_store.create_vectorstore(documents)

# Create RAG chain
rag_chain = RAGChain(
    vectorstore=vector_store.get_vectorstore(),
    llm_model="gpt-3.5-turbo"
)

# Ask questions
response = rag_chain.ask("What is RAG?")
print(response['result'])
```

### Custom Prompts

Edit `src/rag_chain.py` to customize how the AI responds:

```python
self.prompt_template = """Your custom instructions here...
Context: {context}
Question: {question}
Answer: """
```

### Different Vector Databases

To use a different vector database (e.g., Pinecone, FAISS):

1. Install the appropriate package
2. Modify `src/vectorstore.py`
3. Update the initialization code

## Cost Estimation

Typical costs per 1000 questions (GPT-3.5-turbo):
- Embeddings: ~$0.10 (one-time for indexing)
- Queries: ~$1-2 (depends on document length)

For GPT-4, multiply by ~15x

## Performance Optimization

1. **Use caching** for frequently asked questions
2. **Batch indexing** for large document sets
3. **Async processing** for multiple queries
4. **Pre-compute** common query embeddings

## Security Best Practices

1. Never commit `.env` file with API keys
2. Use environment variables in production
3. Implement rate limiting for public APIs
4. Sanitize user inputs
5. Monitor API usage and costs

## Next Steps

1. Try the included sample documents
2. Add your own documents
3. Experiment with different models
4. Customize prompts for your use case
5. Build a web interface (Flask/FastAPI)
6. Deploy to production (Docker, cloud platforms)

## Support

For issues or questions:
- Check the documentation in README.md
- Review the example notebook
- Open an issue on GitHub

Happy building! 🚀
