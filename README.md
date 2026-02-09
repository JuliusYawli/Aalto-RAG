# Aalto-RAG: Retrieval-Augmented Generation Application

A production-ready RAG (Retrieval-Augmented Generation) system for domain-specific question answering. This application combines the power of large language models with custom document retrieval to provide accurate, contextual answers based on your own documents.

> **📝 Note:** This project is configured to use **free Ollama models** by default (no API key needed!). You can switch to OpenAI when you get credits by changing `USE_OLLAMA=false` in the `.env` file.

## 🌟 Features

- **Multi-format Document Support**: Load PDF, TXT, and DOCX files
- **Semantic Search**: Vector-based similarity search with embeddings
- **Context-Aware Answers**: LLM-powered responses grounded in your documents
- **Source Attribution**: See which documents were used for each answer
- **Multiple Interfaces**:
  - Command-line interface (CLI)
  - Interactive Q&A mode
  - Jupyter notebook for experimentation
- **Persistent Storage**: FAISS vector store for fast retrieval
- **Free Local Option**: Use Ollama for completely free local LLM and embeddings
- **Flexible LLM Support**: Switch between Ollama (free) and OpenAI (paid)

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- **Option 1 (Free):** [Ollama](https://ollama.ai) installed locally (recommended for students)
- **Option 2 (Paid):** OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

### Installation

1. Clone the repository:

```bash
git clone https://github.com/JuliusYawli/Aalto-RAG.git
cd Aalto-RAG
```

1. Install dependencies:

```bash
pip install -r requirements.txt
```

1. Set up your environment:

**For Free Ollama (Default):**

```bash
# Install Ollama (macOS)
brew install ollama

# Start Ollama service
brew services start ollama

# Download models
ollama pull llama3.2
ollama pull nomic-embed-text

# The .env file is already configured for Ollama
```

**For OpenAI (When You Have Credits):**

```bash
# Edit .env file and change:
# USE_OLLAMA=false
# OPENAI_API_KEY=your_key_here
```

### Basic Usage

#### 1. Index Your Documents

First, add your documents to the `documents/` directory, then index them:

```bash
python main.py --index
```

This will:

- Load all documents from the `documents/` directory
- Split them into chunks
- Create embeddings
- Store them in a vector database

#### 2. Ask Questions

**Single Question Mode:**

```bash
python main.py --question "What are the best practices for machine learning?"
```

**Interactive Mode:**

```bash
python main.py --interactive
```

In interactive mode, you can ask multiple questions in a conversation-like interface.

## 📚 Documentation

### Project Structure

```text
Aalto-RAG/
├── documents/              # Place your documents here
│   ├── ml_best_practices.txt
│   ├── nlp_guide.txt
│   └── rag_systems.txt
├── src/                    # Source code
│   ├── config.py          # Configuration management
│   ├── document_loader.py # Document loading and processing
│   ├── vectorstore.py     # Vector database management
│   └── rag_chain.py       # RAG chain implementation
├── main.py                # Main CLI application
├── rag_notebook.ipynb     # Jupyter notebook version
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

### Command-Line Options

```bash
# Index documents from default directory
python main.py --index

# Index documents from custom directory
python main.py --index --documents-path /path/to/docs

# Ask a single question
python main.py --question "Your question here"

# Start interactive mode
python main.py --interactive

# Index and then start interactive mode
python main.py --index --interactive
```

### Using the Jupyter Notebook

For experimentation and learning, use the included Jupyter notebook:

```bash
jupyter notebook rag_notebook.ipynb
```

The notebook includes:

- Step-by-step explanations
- Sample documents
- Example questions
- Interactive Q&A cells

You can also open it in Google Colab for cloud-based execution.

## 🔧 Configuration

Edit `.env` or `src/config.py` to customize:

- **OPENAI_API_KEY**: Your OpenAI API key (required)
- **EMBEDDING_MODEL**: Embedding model to use (default: `text-embedding-ada-002`)
- **LLM_MODEL**: Language model to use (default: `gpt-3.5-turbo`)
- **CHUNK_SIZE**: Size of text chunks (default: 1000)
- **CHUNK_OVERLAP**: Overlap between chunks (default: 200)
- **TOP_K_RESULTS**: Number of documents to retrieve (default: 4)

## 📖 How It Works

### RAG Pipeline

1. **Document Indexing**
   - Documents are loaded from the `documents/` directory
   - Text is split into manageable chunks with overlap
   - Each chunk is converted to a vector embedding
   - Embeddings are stored in ChromaDB

2. **Query Processing**
   - User submits a question
   - Question is converted to a vector embedding
   - Similar document chunks are retrieved using cosine similarity
   - Top-K most relevant chunks are selected

3. **Answer Generation**
   - Retrieved context is combined with the user's question
   - The language model generates an answer based on the context
   - Answer is returned with source document references

### Architecture

```text
User Question
     ↓
Question Embedding (OpenAI)
     ↓
Vector Similarity Search (ChromaDB)
     ↓
Top-K Relevant Documents
     ↓
Prompt Construction
     ↓
LLM Generation (GPT-3.5/GPT-4)
     ↓
Answer + Sources
```

## 💡 Example Use Cases

### Customer Support

- Create a knowledge base from product manuals and FAQs
- Answer customer questions automatically
- Provide consistent, accurate responses

### Technical Documentation

- Index API documentation, code examples, and guides
- Help developers find relevant information quickly
- Reduce time spent searching documentation

### Research & Analysis

- Process scientific papers and research documents
- Extract relevant information across multiple sources
- Compare and synthesize information

### Legal & Compliance

- Search through contracts and regulations
- Find relevant case law and precedents
- Ensure compliance with policies

## 🎯 Sample Documents

The repository includes three sample documents:

1. **ml_best_practices.txt**: Machine learning best practices and guidelines
2. **nlp_guide.txt**: Natural language processing concepts and techniques
3. **rag_systems.txt**: In-depth explanation of RAG systems

Try these example questions:

- "What is RAG and how does it work?"
- "What are some best practices for machine learning?"
- "How do transformers work in NLP?"
- "What are the benefits of using RAG systems?"

## 🔒 Security & Privacy

- API keys are stored in `.env` (not committed to git)
- All processing happens locally except API calls to OpenAI
- Vector database is stored locally
- No data is sent to third parties except OpenAI for embeddings/generation

## 🛠️ Development

### Adding Support for New File Types

Edit `src/document_loader.py` to add loaders for additional formats:

```python
elif file_ext == '.md':
    loader = UnstructuredMarkdownLoader(file_path)
```

### Customizing the Prompt

Edit `src/rag_chain.py` to modify the prompt template:

```python
self.prompt_template = """Your custom prompt here..."""
```

### Using Different Vector Databases

The system uses ChromaDB by default, but you can swap it for others:

- Pinecone (managed, scalable)
- Weaviate (feature-rich)
- FAISS (high performance)
- Qdrant (efficient)

## 📊 Performance Considerations

- **Chunk Size**: Smaller chunks = more precise retrieval, but may lose context
- **Chunk Overlap**: Ensures important information isn't split between chunks
- **Top-K Results**: More documents = better context, but higher cost and latency
- **Embedding Model**: Better models = better retrieval accuracy
- **LLM Model**: GPT-4 provides better answers but costs more than GPT-3.5

## 💬 Questions You Can Ask in the App

This is a comprehensive list of all questions you can ask the RAG system. These questions cover the course material across AI/ML, NLP, LLMs, and RAG systems.

### AI & ML Fundamentals

1. What is Artificial Intelligence?
2. What is Machine Learning?
3. What is the difference between AI, ML, and Deep Learning?
4. What are supervised learning and unsupervised learning?
5. What is reinforcement learning?
6. What are the main challenges in AI?
7. How does machine learning differ from traditional programming?
8. What are some real-world applications of AI?
9. What is the importance of data in machine learning?
10. What are the main machine learning paradigms?

### Machine Learning Basics

1. What is a training set and a test set?
2. What is overfitting and underfitting?
3. What is cross-validation and why is it important?
4. What are hyperparameters?
5. What is feature scaling and why is it necessary?
6. What is the bias-variance tradeoff?
7. What are different types of machine learning algorithms?

### Machine Learning Best Practices

1. What are the best practices for feature engineering?
2. How should you approach data preprocessing?
3. What is the importance of exploratory data analysis?
4. How do you handle missing values in datasets?
5. What strategies help prevent overfitting?
6. How do you evaluate model performance?
7. What is the importance of model interpretability?

### Data Preprocessing for NLP

1. What is tokenization?
2. What is stemming and lemmatization?
3. What is stop word removal?
4. What is vectorization in NLP?
5. What is TF-IDF?
6. What are n-grams?
7. What is word embeddings?
8. How do you handle text normalization?
9. What is the importance of data cleaning in NLP?
10. What techniques are used for handling imbalanced datasets in NLP?

### NLP Techniques

1. What is Named Entity Recognition (NER)?
2. What is Part-of-Speech (POS) tagging?
3. What is sentiment analysis?
4. What is machine translation?
5. What is text classification?
6. What are the main NLP tasks?

### Supervised Learning for NLP Tasks

1. What is Naive Bayes and how is it used in NLP?
2. What is Support Vector Machines (SVM)?
3. What is logistic regression?
4. What are decision trees?
5. What is random forest?
6. What is gradient boosting?
7. How do ensemble methods work?
8. What is the difference between classification and regression?
9. What metrics are used to evaluate classification models?
10. What is the confusion matrix?

### Large Language Models (LLMs) Foundation

1. What is a Large Language Model?
2. What are transformers and how do they work?
3. What is the attention mechanism?
4. What is self-attention?
5. What is BERT and how does it work?
6. What is GPT and what makes it different from BERT?
7. How do LLMs generate text?
8. What is the role of pre-training and fine-tuning?
9. What are some popular LLMs available today?
10. How do LLMs understand context?

### LLMs Applications & Usage

1. What are the practical applications of LLMs?
2. How can LLMs be used for question answering?
3. What is prompt engineering?
4. What are few-shot learning and zero-shot learning?
5. How can LLMs be fine-tuned for specific tasks?
6. What are the limitations of LLMs?
7. How can LLMs be made more efficient?
8. What is the role of LLMs in code generation?
9. How can LLMs assist in content creation?
10. What are the ethical considerations when using LLMs?

### RAG Systems

1. What is RAG (Retrieval-Augmented Generation)?
2. How does RAG improve upon standard LLM responses?
3. What are the components of a RAG system?
4. How do vector databases work in RAG?
5. What are the benefits of using RAG systems?

### Course & Instructor Information

1. What is the name of the lecturer?
2. Who teaches this course?
3. What is the course about?
4. Who is Dariush Salami?

### Feature Engineering & Evaluation

1. What is feature importance?
2. How do you select relevant features?
3. What is dimensionality reduction?
4. What metrics are commonly used to evaluate machine learning models?

---

Feel free to ask any of these questions in the web interface or CLI! The system will search through the course materials and provide detailed answers with source references.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

Built with:

- [LangChain](https://python.langchain.com/) - Framework for LLM applications
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [OpenAI](https://openai.com/) - Embeddings and language models

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

## Happy RAG-ing! 🚀
