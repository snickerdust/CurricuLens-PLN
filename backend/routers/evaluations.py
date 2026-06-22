from flask import Blueprint, request, jsonify
from services.db import save_evaluation

evaluations_bp = Blueprint("evaluations", __name__, url_prefix="/api/evaluations")

@evaluations_bp.route("", methods=["POST"])
def create_evaluation():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        doc_id = save_evaluation(data)
        return jsonify({"status": "success", "id": doc_id}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@evaluations_bp.route("/export", methods=["GET"])
def export_evaluations():
    try:
        from services.db import get_all_evaluations
        import csv
        import io
        from flask import Response
        
        evaluations = get_all_evaluations()
        if not evaluations:
            return jsonify({"message": "No data found"}), 404
            
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Build headers based on the first document's structure (assuming standard structure)
        # Standard columns: id, testerName, programStudi, fakultas, universitas, created_at
        headers = ["id", "testerName", "programStudi", "fakultas", "universitas", "created_at"]
        
        # Add blackbox columns (bb1 to bb7 status and notes)
        for i in range(1, 8):
            headers.extend([f"bb{i}_status", f"bb{i}_notes"])
            
        # Add SUS columns (q1 to q10)
        for i in range(1, 11):
            headers.append(f"sus_q{i}")
            
        writer.writerow(headers)
        
        for eval_data in evaluations:
            row = [
                eval_data.get("id", ""),
                eval_data.get("testerName", ""),
                eval_data.get("programStudi", ""),
                eval_data.get("fakultas", ""),
                eval_data.get("universitas", ""),
                eval_data.get("created_at", "")
            ]
            
            blackbox = eval_data.get("blackbox", {})
            for i in range(1, 8):
                bb = blackbox.get(f"bb{i}", {})
                row.extend([bb.get("status", ""), bb.get("notes", "")])
                
            sus = eval_data.get("sus", {})
            for i in range(1, 11):
                row.append(sus.get(f"q{i}", ""))
                
            writer.writerow(row)
            
        output.seek(0)
        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment;filename=evaluations_export.csv"}
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500
