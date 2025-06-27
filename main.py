import os
import sys
import logging
# DON'T CHANGE THIS !!!
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, send_from_directory
from flask_cors import CORS
from chat import chat_bp
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- REMOVED Supabase Client Initialization ---
# The Supabase client is no longer needed as the service now uses Pinecone.
# url = os.getenv("SUPABASE_URL")
# key = os.getenv("SUPABASE_SERVICE_KEY")
# supabase: Client = create_client(url, key)

app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), 'static'))
app.config['SECRET_KEY'] = 'asdf#FGSgvasgf$5$WGT'

# More permissive CORS configuration
CORS(app, origins=[
    "http://localhost:3000", 
    "https://chat-bot-vectorial-p.vercel.app"
], 
allow_headers=["Content-Type", "Authorization"],
methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])

# Register chat blueprint, which contains all our API routes
app.register_blueprint(chat_bp, url_prefix='/api')

logging.basicConfig(level=logging.DEBUG)

@app.route('/health', methods=['GET'])
def health_check():
    """General health check endpoint for the service."""
    return {"status": "healthy", "service": "RAG Chatbot Backend"}

# --- REMOVED /test-supabase route ---
# This is no longer needed. The connection test is now handled by
# the '/api/test-connection' route within the chat blueprint.

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    """Serves the static files for the frontend."""
    static_folder_path = app.static_folder
    if static_folder_path is None:
            return "Static folder not configured", 404

    if path != "" and os.path.exists(os.path.join(static_folder_path, path)):
        return send_from_directory(static_folder_path, path)
    else:
        index_path = os.path.join(static_folder_path, 'index.html')
        if os.path.exists(index_path):
            return send_from_directory(static_folder_path, 'index.html')
        else:
            return "RAG Chatbot Backend API is running. Use /api/chat for chat functionality.", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
