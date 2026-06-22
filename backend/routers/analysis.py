import threading
from flask import Blueprint, jsonify
from services import db, preprocessor

analysis_bp = Blueprint("analysis", __name__, url_prefix="/api")


# ── Internal Analysis Runner ──────────────────────────────────────
def run_analysis_task(curriculum_id: str):
    """Worker function for background analysis."""
    try:
        # 1. Fetch all documents for this curriculum
        docs = db.get_documents(curriculum_id)
        if not docs:
            db.save_analysis_result(curriculum_id, {"status": "error", "error": "Tidak ada dokumen untuk dianalisis."})
            db.update_curriculum_status(curriculum_id, "error")
            return

        # 2. Build corpus using the specialized service function
        # This handles deduplication and source weighting automatically
        # Map field name 'id' to 'doc_id' as expected by the service
        for d in docs: d['doc_id'] = d['id']
        
        # Updated Step: Document-level curation v4
        # Matches logic in 01_kurasi_data_cosine.ipynb
        docs_filtered, ignored_info, inclusion_info = preprocessor.curate_documents_v4(docs, threshold=0.9)
        
        used_ids = [d['id'] for d in docs_filtered]
        
        corpus = preprocessor.build_corpus_from_documents(docs_filtered)
        if not corpus:
            db.save_analysis_result(curriculum_id, {"status": "error", "error": "Tidak ada kalimat valid yang ditemukan setelah kurasi."})
            db.update_curriculum_status(curriculum_id, "error")
            return

        # 3. Run model pipeline (BERT only - no fallback to Lexical)
        from services import model_bert
        analysis_data = model_bert.run_bert_textrank(corpus)

        # 4. Save Results
        result_data = {
            "status": "completed",
            "result": {
                **analysis_data,
                "used_doc_ids": used_ids,
                "ignored_docs": ignored_info,  # {id: reason}
                "inclusion_reasons": inclusion_info, # {id: reason}
            }
        }
        db.save_analysis_result(curriculum_id, result_data)
        db.update_curriculum_status(curriculum_id, "completed")

    except Exception as e:
        import traceback
        error_msg = str(e) + "\n" + traceback.format_exc()
        print(f"ERROR Analysis ({curriculum_id}): {error_msg}") # Menampilkan error di terminal backend
        db.save_analysis_result(curriculum_id, {"status": "error", "error": error_msg})
        db.update_curriculum_status(curriculum_id, "error")


# ── Analysis Endpoints ───────────────────────────────────────────

@analysis_bp.route("/curricula/<curriculum_id>/analyze", methods=["POST"])
def start_analysis(curriculum_id: str):
    if not db.get_curriculum(curriculum_id):
        return jsonify({"error": "Kurikulum tidak ditemukan."}), 404

    # FORCE RE-ANALYSIS: Reset status dan hapus hasil lama agar threshold baru (0.9) langsung ngefek
    db.update_curriculum_status(curriculum_id, "processing")
    db.save_analysis_result(curriculum_id, {"status": "processing"})
    
    # Hapus cache overlap yang berkaitan dengan kurikulum ini
    # (Opsional, agar perbandingan antar kurikulum juga terupdate)

    # Run in background using threading
    thread = threading.Thread(target=run_analysis_task, args=(curriculum_id,))
    thread.start()

    return jsonify({"message": "Analisis dimulai di latar belakang."}), 202


@analysis_bp.route("/curricula/<curriculum_id>/results", methods=["GET"])
def get_results(curriculum_id: str):
    result = db.get_analysis_result(curriculum_id)
    if not result:
        return jsonify({"status": "idle", "message": "Analisis belum pernah dijalankan."})
    return jsonify(result)


# ── Overlap ───────────────────────────────────────────────────────

@analysis_bp.route("/overlap/<id_a>/<id_b>", methods=["GET"])
def get_overlap(id_a: str, id_b: str):
    """
    Compute or retrieve overlap between two completed analyses.
    Uses Firestore as cache for speed.
    """
    # Use consistent key order for cache
    key = f"{min(id_a, id_b)}_{max(id_a, id_b)}"
    cached = db.get_db().collection("overlaps").document(key).get()

    if cached.exists:
        return jsonify(cached.to_dict())

    # Not cached, compute it
    res_a = db.get_analysis_result(id_a)
    res_b = db.get_analysis_result(id_b)

    if not res_a or not res_b or res_a["status"] != "completed" or res_b["status"] != "completed":
        return jsonify({"error": "Kedua kurikulum harus selesai dianalisis dahulu."}), 400

    name_a = db.get_curriculum(id_a).get("name", "A")
    name_b = db.get_curriculum(id_b).get("name", "B")

    overlap_data = preprocessor.compute_overlap(
        res_a["result"]["summary"],
        res_b["result"]["summary"],
        name_a,
        name_b
    )

    db.get_db().collection("overlaps").document(key).set(overlap_data)
    return jsonify(overlap_data)


@analysis_bp.route("/overlap/<id_a>/<id_b>/refresh", methods=["POST"])
def refresh_overlap_cache(id_a: str, id_b: str):
    """Force recomputation on next overlap request."""
    key = f"{min(id_a, id_b)}_{max(id_a, id_b)}"
    db.get_db().collection("overlaps").document(key).delete()
    return "", 204
