import os
from flask import Blueprint, request, jsonify
from services import db, extractor, preprocessor

documents_bp = Blueprint("documents", __name__, url_prefix="/api/curricula/<curriculum_id>/documents")

ALLOWED_EXTENSIONS = {".docx", ".pptx", ".pdf", ".doc", ".ppt"}
MAX_FILE_SIZE_MB = 100


@documents_bp.route("", methods=["GET"])
def list_documents(curriculum_id: str):
    if not db.get_curriculum(curriculum_id):
        return jsonify({"error": "Kurikulum tidak ditemukan."}), 404
    return jsonify(db.get_documents(curriculum_id))


@documents_bp.route("", methods=["POST"])
def upload_document(curriculum_id: str):
    if not db.get_curriculum(curriculum_id):
        return jsonify({"error": "Kurikulum tidak ditemukan."}), 404

    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    sumber = request.form.get("sumber", "3. Handout")

    if not file or file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Format file {ext} tidak didukung."}), 400

    file_bytes = file.read()
    if len(file_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
        return jsonify({"error": f"Ukuran file melebihi {MAX_FILE_SIZE_MB} MB."}), 400

    # Extract & Preprocess
    try:
        raw_text = extractor.extract_text(file_bytes, filename)
        if not raw_text.strip():
            return jsonify({"error": "Gagal mengekstrak teks. File mungkin kosong atau terenkripsi."}), 400

        # Run preprocessing pipeline v4
        kalimat_list = preprocessor.split_to_sentences(raw_text)
        lemma_text = preprocessor.preprocess_v4(raw_text)

        doc_data = {
            "filename": filename,
            "sumber": sumber,
            "teks_original": raw_text,
            "teks_lemma": lemma_text,
            "kalimat_list": kalimat_list,
            "word_count": len(raw_text.split()),
            "n_sentences": len(kalimat_list),
            "status": "processed",
        }

        doc_id = db.add_document(curriculum_id, doc_data)
        # Update status kurikulum agar perlu re-analisis
        db.update_curriculum_status(curriculum_id, "stale")

        return jsonify({"id": doc_id, "status": "success"}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@documents_bp.route("/<doc_id>", methods=["GET"])
def get_document(curriculum_id: str, doc_id: str):
    if not db.get_curriculum(curriculum_id):
        return jsonify({"error": "Kurikulum tidak ditemukan."}), 404
    doc = db.get_document_detail(curriculum_id, doc_id)
    if not doc:
        return jsonify({"error": "Dokumen tidak ditemukan."}), 404
    return jsonify(doc)


@documents_bp.route("/<doc_id>", methods=["DELETE"])
def delete_document(curriculum_id: str, doc_id: str):
    if not db.get_curriculum(curriculum_id):
        return jsonify({"error": "Kurikulum tidak ditemukan."}), 404
    db.delete_document(curriculum_id, doc_id)
    db.update_curriculum_status(curriculum_id, "stale")
    return "", 204
