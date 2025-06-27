import os
import logging
import re
from typing import List, Any, Dict, Optional

from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.retrievers import BaseRetriever
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from pydantic.v1 import PrivateAttr
from langchain.prompts import ChatPromptTemplate

# Load environment variables from .env file
load_dotenv()

class PineconeConfig:
    """Configuration for Pinecone-based RAG."""
    def __init__(self):
        # --- Pinecone and OpenAI Credentials ---
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        if not all([self.pinecone_api_key, self.pinecone_index_name, self.openai_api_key]):
            raise ValueError("PINECONE_API_KEY, PINECONE_INDEX_NAME, and OPENAI_API_KEY environment variables are required.")

        # --- Model and Search Defaults ---
        self.default_embedding_model = os.getenv("DEFAULT_EMBEDDING_MODEL", "BAAI/bge-m3")
        self.default_chat_model = os.getenv("DEFAULT_CHAT_MODEL", "gpt-4o-mini")
        self.default_top_k = int(os.getenv("DEFAULT_TOP_K", 10))
        self.default_threshold = float(os.getenv("DEFAULT_THRESHOLD", 0.2))
        self.default_temperature = float(os.getenv("DEFAULT_TEMPERATURE", 0.1))

class PineconeRetriever(BaseRetriever):
    """Custom retriever for Pinecone vector search."""
    embeddings: Embeddings
    top_k: int
    threshold: float
    _index: Any = PrivateAttr()

    def __init__(self, index: Any, embeddings: Embeddings, **kwargs):
        super().__init__(embeddings=embeddings, **kwargs)
        self._index = index

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieves relevant documents from Pinecone based on vector similarity.
        """
        try:
            query_embedding = self.embeddings.embed_query(query)
            
            # Query Pinecone index
            response = self._index.query(
                vector=query_embedding,
                top_k=self.top_k,
                include_metadata=True,
                include_values=False
            )
            
            # Filter and format documents
            documents = []
            for match in response.matches:
                if match.score >= self.threshold:
                    metadata = match.metadata or {}
                    documents.append(Document(
                        page_content=metadata.get("text", ""),
                        metadata={
                            "source": metadata.get("file_name"),
                            "doc_type": metadata.get("doc_type"),
                            "score": match.score
                        }
                    ))
            return documents
        except Exception as e:
            logging.error(f"Error during Pinecone query: {e}")
            return []

class RAGService:
    """RAG Service implementation using Pinecone."""
    def __init__(self, config: PineconeConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

        try:
            # Initialize Pinecone client and index
            pc = Pinecone(api_key=self.config.pinecone_api_key)
            self.index = pc.Index(self.config.pinecone_index_name)
            self.logger.info(f"Successfully connected to Pinecone index '{self.config.pinecone_index_name}'.")
        except Exception as e:
            self.logger.critical(f"Failed to initialize Pinecone: {e}")
            raise

    def get_embeddings(self, model_name: Optional[str] = None) -> Embeddings:
        """Creates and returns an embedding model instance."""
        model = model_name or self.config.default_embedding_model
        try:
            if "text-embedding-" in model:
                return OpenAIEmbeddings(model=model, openai_api_key=self.config.openai_api_key)
            return HuggingFaceEmbeddings(model_name=model)
        except Exception as e:
            self.logger.error(f"Error creating embeddings for model {model}: {e}")
            raise

    def get_retriever(self, **kwargs) -> PineconeRetriever:
        """Builds and returns a Pinecone retriever."""
        embeddings = self.get_embeddings(kwargs.get("embedding_model"))
        return PineconeRetriever(
            index=self.index,
            embeddings=embeddings,
            top_k=kwargs.get("top_k", self.config.default_top_k),
            threshold=kwargs.get("threshold", self.config.default_threshold)
        )

    def fix_markdown_numbered_lists(self, text):
        logging.info('Raw answer before fix:\n' + text)
        # Join number-dot and bolded title, regardless of whitespace/newlines in between
        text = re.sub(r'(\d+)\.\s*\n*\s*(\*\*.+?\*\*)', r'\1. \2', text)
        # Remove extra newlines after the bolded title
        text = re.sub(r'(\*\*.+?\*\*)\n+', r'\1 ', text)
        logging.info('Answer after fix:\n' + text)
        return text

    def query_rag(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Executes a RAG query against Pinecone and a language model.
        """
        # Set parameters, falling back to defaults if not provided
        params = {
            "embedding_model": kwargs.get("embedding_model", self.config.default_embedding_model),
            "top_k": kwargs.get("top_k", self.config.default_top_k),
            "threshold": kwargs.get("threshold", self.config.default_threshold),
            "temperature": kwargs.get("temperature", self.config.default_temperature),
            "chat_model": kwargs.get("chat_model", self.config.default_chat_model),
        }
        
        try:
            retriever = self.get_retriever(**params)
            llm = ChatOpenAI(model=params["chat_model"], temperature=params["temperature"], openai_api_key=self.config.openai_api_key)
            
            # Define prompt structure
            system_message_content = "Eres un experto en el sistema de pensiones dominicano..."
            human_template = "Por favor, responde la siguiente pregunta... Contexto:\n{context}\n\n---\nPregunta: {question}"
            custom_prompt = ChatPromptTemplate.from_messages([("system", system_message_content), ("human", human_template)])

            # Create and run the QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": custom_prompt}
            )
            result = qa_chain.invoke({"query": query})

            # Format source documents for the final response
            unique_sources = {}
            for doc in result.get("source_documents", []):
                source_name = (doc.metadata.get("source") or "").strip()
                doc_type = (doc.metadata.get("doc_type") or "").strip().lower()
                norm = source_name.lower()

                if source_name and norm != "none" and norm not in unique_sources:
                    if doc_type == "leyes":
                        link = f"https://www.sipen.gob.do/descarga/{source_name}"
                    else:
                        link = f"https://www.sipen.gob.do/documentos/{source_name}"
                    unique_sources[norm] = {"source_name": source_name, "url": link}
            
            answer = result["result"]
            answer = self.fix_markdown_numbered_lists(answer)
            return {
                "answer": answer,
                "source_documents": list(unique_sources.values()),
                "parameters_used": params
            }
        except Exception as e:
            self.logger.error(f"Error during RAG query: {e}", exc_info=True)
            return {
                "answer": f"An error occurred: {e}",
                "source_documents": [],
                "parameters_used": params
            }

    def test_connection(self) -> Dict[str, str]:
        """
        Tests the connection to the Pinecone index by fetching stats.
        """
        try:
            stats = self.index.describe_index_stats()
            return {"status": "success", "message": f"Connected to Pinecone. Index stats: {stats}"}
        except Exception as e:
            return {"status": "error", "message": f"Connection failed: {e}"}

# --- Example Usage ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        config = PineconeConfig()
        rag_service = RAGService(config)

        # Test the connection
        connection_status = rag_service.test_connection()
        print(f"--- Connection Test ---\n{connection_status}\n")

        if connection_status["status"] == "success":
            # Perform a sample query
            user_query = "¿Qué dice la ley 87-01 sobre el régimen contributivo de pensiones y jubilaciones?"
            response = rag_service.query_rag(user_query)
            print("\n--- RAG Query Response ---")
            print(f"Answer: {response['answer']}")
            print(f"Source Documents: {response['source_documents']}")
    except ValueError as e:
        print(f"Configuration Error: {e}")
        print("\nPlease ensure you have a .env file in the root of your project.")
        print("The .env file should contain the following keys:")
        print("PINECONE_API_KEY='your_actual_pinecone_api_key'")
        print("PINECONE_INDEX_NAME='your_actual_pinecone_index_name'")
        print("OPENAI_API_KEY='your_actual_openai_api_key'")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
