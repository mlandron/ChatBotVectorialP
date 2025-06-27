import logging
import sys
from flask import Blueprint, request, jsonify
from rag_service import RAGService, PineconeConfig

# --- Blueprint Setup ---
chat_bp = Blueprint('chat', __name__)

# --- Initialize Service (Singleton Pattern) ---
# This single RAG service instance is initialized once and shared across all requests.
try:
    # Use the new PineconeConfig
    config = PineconeConfig()
    rag_service = RAGService(config)
    logger = logging.getLogger(__name__)
except ValueError as e:
    # If configuration fails (e.g., missing API keys), log a critical error and exit.
    logging.critical(f"FATAL: Could not start RAG service due to config error: {e}")
    sys.exit(f"Application shutdown: {e}")

@chat_bp.route('/chat', methods=['POST'])
def chat():
    """
    Handles chat requests by querying the Pinecone RAG service.
    """
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Missing 'query' in request body"}), 400

    try:
        query = data['query']
        
        # Extract parameters from the request body.
        # These correspond to the settings the user can change in the frontend.
        params = {
            "embedding_model": data.get('embedding_model'),
            "top_k": data.get('top_k'),
            "threshold": data.get('threshold'),
            "temperature": data.get('temperature'),
            "chat_model": data.get('chat_model'),
        }
        
        # Filter out any parameters that were not provided (i.e., are None).
        # The RAG service will use its default values for these.
        final_params = {k: v for k, v in params.items() if v is not None}

        # Perform the RAG query using the service
        result = rag_service.query_rag(query, **final_params)
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in /chat endpoint: {e}", exc_info=True)
        return jsonify({"error": "Internal server error", "message": str(e)}), 500

@chat_bp.route('/test-connection', methods=['GET'])
def test_connection():
    """
    Endpoint to test the connection to the Pinecone service.
    """
    try:
        status = rag_service.test_connection()
        if status["status"] == "success":
            return jsonify(status), 200
        else:
            return jsonify(status), 500
    except Exception as e:
        logger.error(f"Error testing connection: {e}", exc_info=True)
        return jsonify({"status": "error", "message": f"Connection test failed: {e}"}), 500

@chat_bp.route('/parameters', methods=['GET'])
def get_parameters():
    """
    Provides the frontend with available models and parameter details.
    """
    return jsonify({
        "embedding_models": [
            {"value": "BAAI/bge-m3", "label": "BAAI/bge-m3 (Default)"},
            {"value": "text-embedding-3-small", "label": "OpenAI text-embedding-3-small"},
            {"value": "text-embedding-3-large", "label": "OpenAI text-embedding-3-large"}
        ],
        "chat_models": [
            {"value": "gpt-4o-mini", "label": "GPT-4o Mini (Default)"},
            {"value": "gpt-4o", "label": "GPT-4o"},
            {"value": "gpt-4", "label": "GPT-4"},
            {"value": "gpt-3.5-turbo", "label": "GPT-3.5 Turbo"}
        ]
        # Note: You can add ranges for top_k, threshold, etc. here if you want
        # the frontend to have sliders or validated number inputs.
    })
