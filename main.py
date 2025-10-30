from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import uvicorn
import requests
from datetime import datetime
import traceback

# Ollama configuration
OLLAMA_HOST = "http://localhost:11434"
EMBED_MODEL = "bge-m3"

# Custom embedding function for Ollama
class OllamaEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(self, model_name: str, ollama_host: str):
        self.model_name = model_name
        self.ollama_host = ollama_host
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        """Generate embeddings using Ollama"""
        embeddings = []
        for text in input:
            try:
                response = requests.post(
                    f"{self.ollama_host}/api/embeddings",
                    json={"model": self.model_name, "prompt": text},
                    timeout=30
                )
                response.raise_for_status()
                embeddings.append(response.json()["embedding"])
            except Exception as e:
                print(f"❌ Error generating embedding: {e}")
                # Return a zero vector as fallback
                raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")
        return embeddings

# Initialize FastAPI app
app = FastAPI(
    title="ChromaDB RAG API",
    description="API for managing RAG (Retrieval-Augmented Generation) with ChromaDB",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ChromaDB client
try:
    chroma_client = chromadb.PersistentClient(
        path="./chroma_db",
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )
    )
    print("✅ ChromaDB client initialized successfully")
except Exception as e:
    print(f"❌ Error initializing ChromaDB: {e}")
    chroma_client = None

# Initialize Ollama embedding function
ollama_ef = OllamaEmbeddingFunction(
    model_name=EMBED_MODEL,
    ollama_host=OLLAMA_HOST
)

# Default collection name
DEFAULT_COLLECTION = "jewelry_catalog_2"

# Pydantic models
class DocumentInput(BaseModel):
    text: str
    metadata: Optional[dict] = {}
    id: Optional[str] = None

class QueryInput(BaseModel):
    query: str
    n_results: Optional[int] = 5
    collection_name: Optional[str] = DEFAULT_COLLECTION

class CollectionCreate(BaseModel):
    name: str
    metadata: Optional[dict] = {}

class DeleteDocuments(BaseModel):
    ids: List[str]
    collection_name: Optional[str] = DEFAULT_COLLECTION

# Helper functions
def get_or_create_collection(collection_name: str = DEFAULT_COLLECTION):
    """Get or create a ChromaDB collection with Ollama embeddings"""
    try:
        if chroma_client is None:
            raise HTTPException(status_code=500, detail="ChromaDB client not initialized")
        
        collection = chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=ollama_ef,
            metadata={"description": f"Collection for {collection_name}"}
        )
        return collection
    except Exception as e:
        print(f"❌ Error accessing collection: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error accessing collection: {str(e)}")

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "ChromaDB RAG API with Ollama Embeddings",
        "version": "1.0.0",
        "embedding_model": EMBED_MODEL,
        "ollama_host": OLLAMA_HOST,
        "endpoints": {
            "health": "/health",
            "collections": "/collections",
            "add_document": "/documents/add",
            "add_batch": "/documents/add_batch",
            "query": "/query",
            "delete": "/documents/delete",
            "get_documents": "/documents/{collection_name}"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        if chroma_client is None:
            return {
                "status": "unhealthy",
                "error": "ChromaDB client not initialized",
                "timestamp": datetime.now().isoformat()
            }
        
        collections = chroma_client.list_collections()
        
        # Check Ollama connection
        ollama_status = "connected"
        try:
            test_response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
            test_response.raise_for_status()
        except Exception as e:
            ollama_status = f"disconnected: {str(e)}"
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "collections_count": len(collections),
            "ollama_status": ollama_status,
            "embedding_model": EMBED_MODEL
        }
    except Exception as e:
        print(f"❌ Health check error: {e}")
        print(traceback.format_exc())
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/collections")
async def list_collections():
    """List all collections"""
    try:
        if chroma_client is None:
            raise HTTPException(status_code=500, detail="ChromaDB client not initialized")
        
        collections = chroma_client.list_collections()
        return {
            "collections": [
                {
                    "name": col.name,
                    "count": col.count(),
                    "metadata": col.metadata
                }
                for col in collections
            ]
        }
    except Exception as e:
        print(f"❌ Error listing collections: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error listing collections: {str(e)}")

@app.post("/collections/create")
async def create_collection(collection: CollectionCreate):
    """Create a new collection"""
    try:
        if chroma_client is None:
            raise HTTPException(status_code=500, detail="ChromaDB client not initialized")
        
        new_collection = chroma_client.create_collection(
            name=collection.name,
            embedding_function=ollama_ef,
            metadata=collection.metadata
        )
        return {
            "message": f"Collection '{collection.name}' created successfully",
            "name": new_collection.name,
            "count": new_collection.count()
        }
    except Exception as e:
        print(f"❌ Error creating collection: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=400, detail=f"Error creating collection: {str(e)}")

@app.delete("/collections/{collection_name}")
async def delete_collection(collection_name: str):
    """Delete a collection"""
    try:
        if chroma_client is None:
            raise HTTPException(status_code=500, detail="ChromaDB client not initialized")
        
        chroma_client.delete_collection(name=collection_name)
        return {
            "message": f"Collection '{collection_name}' deleted successfully"
        }
    except Exception as e:
        print(f"❌ Error deleting collection: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=400, detail=f"Error deleting collection: {str(e)}")

@app.post("/documents/add")
async def add_document(
    document: DocumentInput,
    collection_name: Optional[str] = DEFAULT_COLLECTION
):
    """Add a single document to the collection"""
    try:
        print(f"=== Adding Document to ChromaDB ===")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"Collection: {collection_name}")
        print(f"Text length: {len(document.text)} characters")
        print(f"Generating embedding with {EMBED_MODEL}...")
        
        collection = get_or_create_collection(collection_name)
        
        # Generate ID if not provided
        doc_id = document.id or f"doc_{datetime.now().timestamp()}"
        
        # Add metadata timestamp
        metadata = document.metadata.copy() if document.metadata else {}
        metadata["timestamp"] = datetime.now().isoformat()
        
        # Embeddings will be generated automatically by the collection's embedding function
        collection.add(
            documents=[document.text],
            metadatas=[metadata],
            ids=[doc_id]
        )
        
        print(f"✅ Document added successfully with ID: {doc_id}")
        print("===================================")
        
        return {
            "message": "Document added successfully",
            "id": doc_id,
            "collection": collection_name,
            "total_documents": collection.count()
        }
    except Exception as e:
        print(f"❌ Error adding document: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error adding document: {str(e)}")

@app.post("/documents/add_batch")
async def add_documents_batch(
    documents: List[DocumentInput],
    collection_name: Optional[str] = DEFAULT_COLLECTION
):
    """Add multiple documents to the collection"""
    try:
        print(f"=== Batch Adding {len(documents)} Documents ===")
        print(f"Collection: {collection_name}")
        
        collection = get_or_create_collection(collection_name)
        
        texts = []
        metadatas = []
        ids = []
        
        for i, doc in enumerate(documents):
            texts.append(doc.text)
            
            # Add metadata with timestamp
            metadata = doc.metadata.copy() if doc.metadata else {}
            metadata["timestamp"] = datetime.now().isoformat()
            metadatas.append(metadata)
            
            # Generate ID if not provided
            doc_id = doc.id or f"doc_{datetime.now().timestamp()}_{i}"
            ids.append(doc_id)
        
        collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✅ {len(documents)} documents added successfully")
        
        return {
            "message": f"{len(documents)} documents added successfully",
            "ids": ids,
            "collection": collection_name,
            "total_documents": collection.count()
        }
    except Exception as e:
        print(f"❌ Error adding documents: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error adding documents: {str(e)}")

@app.post("/query")
async def query_documents(query_input: QueryInput):
    """Query documents from the collection"""
    try:
        collection = get_or_create_collection(query_input.collection_name)
        
        print(f"=== ChromaDB Query ===")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"Collection: {query_input.collection_name}")
        print(f"Query: {query_input.query}")
        print(f"N Results: {query_input.n_results}")
        print(f"Generating query embedding with {EMBED_MODEL}...")
        
        # Query will automatically use the embedding function
        results = collection.query(
            query_texts=[query_input.query],
            n_results=query_input.n_results
        )
        
        # Format results
        formatted_results = []
        if results['documents'] and len(results['documents']) > 0:
            for i in range(len(results['documents'][0])):
                # Calculate score (1 - distance for similarity)
                distance = results['distances'][0][i] if results['distances'] else 0
                score = 1 - distance if distance is not None else 0
                
                formatted_results.append({
                    "id": results['ids'][0][i] if results['ids'] else None,
                    "document": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                    "distance": distance,
                    "score": score
                })
        
        print(f"✅ Found {len(formatted_results)} results")
        print("====================")
        
        return {
            "query": query_input.query,
            "collection": query_input.collection_name,
            "results": formatted_results,
            "count": len(formatted_results)
        }
    except Exception as e:
        print(f"❌ Query error: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error querying documents: {str(e)}")

@app.delete("/documents/delete")
async def delete_documents(delete_input: DeleteDocuments):
    """Delete documents by IDs"""
    try:
        collection = get_or_create_collection(delete_input.collection_name)
        
        collection.delete(ids=delete_input.ids)
        
        return {
            "message": f"{len(delete_input.ids)} documents deleted successfully",
            "deleted_ids": delete_input.ids,
            "remaining_documents": collection.count()
        }
    except Exception as e:
        print(f"❌ Delete error: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error deleting documents: {str(e)}")

@app.get("/documents/{collection_name}")
async def get_all_documents(collection_name: str = DEFAULT_COLLECTION):
    """Get all documents from a collection"""
    try:
        print(f"=== Getting All Documents ===")
        print(f"Collection: {collection_name}")
        
        if chroma_client is None:
            raise HTTPException(status_code=500, detail="ChromaDB client not initialized")
        
        # Check if collection exists
        existing_collections = [col.name for col in chroma_client.list_collections()]
        if collection_name not in existing_collections:
            return {
                "collection": collection_name,
                "documents": [],
                "count": 0,
                "message": "Collection does not exist yet"
            }
        
        collection = get_or_create_collection(collection_name)
        
        # Get all documents
        results = collection.get()
        
        documents = []
        if results and results.get('ids'):
            for i in range(len(results['ids'])):
                documents.append({
                    "id": results['ids'][i],
                    "document": results['documents'][i] if results.get('documents') else None,
                    "metadata": results['metadatas'][i] if results.get('metadatas') else {}
                })
        
        print(f"✅ Retrieved {len(documents)} documents")
        
        return {
            "collection": collection_name,
            "documents": documents,
            "count": len(documents)
        }
    except Exception as e:
        print(f"❌ Get documents error: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error retrieving documents: {str(e)}")

@app.post("/upload_text_file")
async def upload_text_file(
    file: UploadFile = File(...),
    collection_name: Optional[str] = DEFAULT_COLLECTION
):
    """Upload a text file and add its content to the collection"""
    try:
        # Read file content
        content = await file.read()
        text = content.decode('utf-8')
        
        # Split into chunks (simple split by paragraphs)
        chunks = [chunk.strip() for chunk in text.split('\n\n') if chunk.strip()]
        
        collection = get_or_create_collection(collection_name)
        
        # Add chunks to collection
        ids = [f"{file.filename}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": file.filename, "chunk_index": i, "timestamp": datetime.now().isoformat()} for i in range(len(chunks))]
        
        collection.add(
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )
        
        return {
            "message": "File uploaded and processed successfully",
            "filename": file.filename,
            "chunks_added": len(chunks),
            "collection": collection_name,
            "total_documents": collection.count()
        }
    except Exception as e:
        print(f"❌ Upload error: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

if __name__ == "__main__":
    print("=" * 50)
    print("Starting ChromaDB RAG API Server")
    print(f"Server will run on: http://127.0.0.1:23456")
    print(f"Documentation: http://127.0.0.1:23456/docs")
    print(f"Embedding Model: {EMBED_MODEL}")
    print(f"Ollama Host: {OLLAMA_HOST}")
    print("=" * 50)
    
    # Test Ollama connection
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        response.raise_for_status()
        print("✅ Ollama connection successful")
    except Exception as e:
        print(f"⚠️  Warning: Cannot connect to Ollama: {e}")
        print("Make sure Ollama is running with: ollama serve")
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=23456,
        log_level="info"
    )
