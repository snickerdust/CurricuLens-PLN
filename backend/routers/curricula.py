from flask import Blueprint, request, jsonify
from services import db

curricula_bp = Blueprint("curricula", __name__, url_prefix="/api/curricula")

COLORS = ["indigo", "amber", "teal", "rose", "violet", "sky", "orange", "emerald"]


@curricula_bp.route("", methods=["GET"])
def list_curricula():
    curricula = db.get_all_curricula()
    # Populate documents for each curriculum
    for curr in curricula:
        curr["documents"] = db.get_documents(curr["id"])
        # Remove large fields (keep teks_original for frontend search capability)
        for d in curr["documents"]:
            d.pop("teks_lemma", None)
            d.pop("kalimat_list", None)
    return jsonify(curricula)


@curricula_bp.route("", methods=["POST"])
def create_curriculum():
    data = request.json
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "Nama kurikulum wajib diisi."}), 400

    import random
    color = random.choice(COLORS)
    curr_id = db.create_curriculum(name, color)
    return jsonify({"id": curr_id, "name": name, "color": color}), 201


@curricula_bp.route("/<curriculum_id>", methods=["DELETE"])
def delete_curriculum(curriculum_id: str):
    if not db.get_curriculum(curriculum_id):
        return jsonify({"error": "Kurikulum tidak ditemukan."}), 404
    db.delete_curriculum(curriculum_id)
    return "", 204
