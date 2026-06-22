"""
Sistem Analisis Kurikulum — Flask Backend (Restful API)
Run: python app.py
"""
import os
from dotenv import load_dotenv
from flask import Flask, jsonify, send_file
from flask_cors import CORS

load_dotenv()

# Impor router (Blueprints)
from routers.curricula import curricula_bp
from routers.documents import documents_bp
from routers.analysis import analysis_bp
from routers.evaluations import evaluations_bp

app = Flask(__name__)

# ── CORS ──────────────────────────────────────────────────────────
_raw_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5500,http://127.0.0.1:5500,null,http://localhost:3000,http://localhost:8000,http://127.0.0.1:8000",
)
allowed_origins = [o.strip() for o in _raw_origins.split(",")]

CORS(app, resources={r"/api/*": {"origins": allowed_origins}}, supports_credentials=True)

# ── Blueprints ────────────────────────────────────────────────────
app.register_blueprint(curricula_bp)
app.register_blueprint(documents_bp)
app.register_blueprint(analysis_bp)
app.register_blueprint(evaluations_bp)


# ── Frontend ─────────────────────────────────────────────────────
_FRONTEND_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "index.html"))

@app.route("/")
def index():
    """Serve the frontend SPA."""
    return send_file(_FRONTEND_PATH)


@app.route("/testing")
@app.route("/testing.html")
def testing():
    """Serve the testing page."""
    testing_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "testing.html"))
    return send_file(testing_path)

@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    # Menjalankan server pada port 8000 (sesuai setting frontend sebelumnya)
    app.run(host="0.0.0.0", port=8000, debug=True)
