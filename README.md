# RAG FastAPI Application with LlamaIndex and ChromaDB

A production-ready Retrieval-Augmented Generation (RAG) application built with FastAPI, LlamaIndex, and ChromaDB. This system enables intelligent document retrieval and question-answering across FAQs, website content, and policy documents.

## 🚀 Features

- **Multi-Document Type Support**:
  - FAQ documents (Excel format with Q&A pairs)
  - Website content (text files)
  - Policy documents (text files)

- **Intelligent Document Processing**:
  - Automatic optimal chunking based on document type
  - Smart text splitting with configurable overlap
  - Metadata extraction and tagging

- **Advanced RAG Pipeline**:
  - Vector similarity search using OpenAI embeddings
  - ChromaDB for efficient vector storage
  - Context-aware answer generation
  - Source attribution for transparency

- **Production-Ready Architecture**:
  - Modular, maintainable code structure
  - Comprehensive error handling
  - Health checks and monitoring
  - CORS support for web applications
  - Configurable via environment variables

## 📋 Prerequisites

- Python 3.13+
- OpenAI API key
- 4GB+ RAM recommended
- 1GB+ disk space for ChromaDB persistence

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone https://github.com/sagar-j-gurav/rag-fastapi-llamaindex.git
cd rag-fastapi-llamaindex
```

2. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

## 🔧 Configuration

Edit the `.env` file with your settings:

```env
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional (defaults shown)
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
CHROMA_PERSIST_DIR=./chroma_db
```

### Document Type Chunking Strategies

The system automatically applies optimal chunking based on document type:

- **FAQ**: 256 tokens chunk size, 20 tokens overlap
- **Website**: 512 tokens chunk size, 100 tokens overlap  
- **Policy**: 384 tokens chunk size, 75 tokens overlap

## 🚀 Running the Application

### Development Mode
```bash
python main.py
```

### Production Mode
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

The API will be available at `http://localhost:8000`

## 📚 API Documentation

Once running, access the interactive API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### API Endpoints

#### 1. Upload Document
**POST** `/api/upload`

Upload and index a document into the RAG system.

```bash
# Upload FAQ Excel file
curl -X POST "http://localhost:8000/api/upload" \
  -F "file=@faq.xlsx" \
  -F "document_type=faq"

# Upload website content
curl -X POST "http://localhost:8000/api/upload" \
  -F "file=@website.txt" \
  -F "document_type=website"
```

#### 2. Query RAG System
**POST** `/api/query`

Query the indexed documents for answers.

```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is your refund policy?",
    "document_type": "faq",
    "top_k": 5,
    "include_sources": true
  }'
```

#### 3. Health Check
**GET** `/api/health`

Check system health status.

```bash
curl "http://localhost:8000/api/health"
```

#### 4. System Statistics
**GET** `/api/stats`

Get indexed document statistics.

```bash
curl "http://localhost:8000/api/stats"
```

## 📁 Project Structure

```
rag-fastapi-llamaindex/
├── main.py                 # FastAPI application entry point
├── config.py              # Configuration management
├── models.py              # Pydantic models for validation
├── document_processor.py  # Document processing logic
├── rag_engine.py          # RAG indexing and retrieval
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── .gitignore            # Git ignore file
└── README.md             # This file
```

## 🧪 Testing

### Upload Test Files

1. **Create a test FAQ Excel file** with columns:
   - Question
   - Answer
   - Category (optional)

2. **Create a test policy text file** with your policy content

3. **Upload and test**:
```python
import requests

# Upload FAQ
with open('test_faq.xlsx', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/upload',
        files={'file': f},
        data={'document_type': 'faq'}
    )
    print(response.json())

# Query
response = requests.post(
    'http://localhost:8000/api/query',
    json={
        'query': 'Your question here',
        'top_k': 5,
        'include_sources': True
    }
)
print(response.json())
```

## 🔒 Security Considerations

- Store API keys securely using environment variables
- Implement rate limiting for production deployments
- Add authentication/authorization as needed
- Validate and sanitize all user inputs
- Use HTTPS in production
- Regular security updates for dependencies

## 🐛 Troubleshooting

### Common Issues

1. **ChromaDB initialization error**:
   - Ensure the persist directory has write permissions
   - Try deleting the `chroma_db` folder and restarting

2. **OpenAI API errors**:
   - Verify your API key is correct
   - Check your OpenAI account has credits
   - Ensure you're using a supported model

3. **Memory issues**:
   - Reduce chunk size in configuration
   - Limit the number of concurrent requests
   - Consider using a smaller embedding model

## 🚀 Deployment

### Using Docker

```dockerfile
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Using Docker Compose

```yaml
version: '3.8'

services:
  rag-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./chroma_db:/app/chroma_db
```

## 📈 Performance Optimization

1. **Caching**: Implement Redis for query result caching
2. **Async Processing**: Use background tasks for large document uploads
3. **Batch Processing**: Process multiple documents in batches
4. **Index Optimization**: Periodically optimize ChromaDB indices
5. **Load Balancing**: Deploy multiple instances behind a load balancer

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- [LlamaIndex](https://www.llamaindex.ai/) for the RAG framework
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [OpenAI](https://openai.com/) for embeddings and LLM

## 📧 Contact

For questions or support, please open an issue on GitHub.