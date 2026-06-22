import os
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1 import SERVER_TIMESTAMP

_db = None


def get_db() -> firestore.Client:
    global _db
    if _db is None:
        cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "firebase_credentials.json")
        if not firebase_admin._apps:
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
        _db = firestore.client()
    return _db


# ── Curricula ────────────────────────────────────────────────────

def get_all_curricula() -> list[dict]:
    db = get_db()
    docs = db.collection("curricula").order_by("created_at").stream()
    return [{"id": d.id, **d.to_dict()} for d in docs]


def get_curriculum(curriculum_id: str) -> dict | None:
    db = get_db()
    doc = db.collection("curricula").document(curriculum_id).get()
    if not doc.exists:
        return None
    return {"id": doc.id, **doc.to_dict()}


def create_curriculum(name: str, color: str = "indigo") -> dict:
    db = get_db()
    ref = db.collection("curricula").document()
    data = {"name": name, "color": color, "created_at": SERVER_TIMESTAMP}
    ref.set(data)
    return {"id": ref.id, "name": name, "color": color}


def delete_curriculum(curriculum_id: str):
    db = get_db()
    # Delete all documents in subcollections first
    _delete_subcollection(db, f"curricula/{curriculum_id}/documents")
    _delete_subcollection(db, f"curricula/{curriculum_id}/results")
    db.collection("curricula").document(curriculum_id).delete()


# ── Documents ────────────────────────────────────────────────────

def get_documents(curriculum_id: str) -> list[dict]:
    db = get_db()
    docs = (
        db.collection("curricula")
        .document(curriculum_id)
        .collection("documents")
        .order_by("uploaded_at")
        .stream()
    )
    return [{"id": d.id, **d.to_dict()} for d in docs]


def add_document(curriculum_id: str, data: dict) -> str:
    db = get_db()
    ref = (
        db.collection("curricula")
        .document(curriculum_id)
        .collection("documents")
        .document()
    )
    ref.set({**data, "uploaded_at": SERVER_TIMESTAMP, "status": "uploaded"})
    return ref.id


def update_document(curriculum_id: str, doc_id: str, data: dict):
    db = get_db()
    (
        db.collection("curricula")
        .document(curriculum_id)
        .collection("documents")
        .document(doc_id)
        .update(data)
    )


def get_document_detail(curriculum_id: str, doc_id: str) -> dict | None:
    db = get_db()
    doc = (
        db.collection("curricula")
        .document(curriculum_id)
        .collection("documents")
        .document(doc_id)
        .get()
    )
    if not doc.exists:
        return None
    return {"id": doc.id, **doc.to_dict()}


def delete_document(curriculum_id: str, doc_id: str):
    db = get_db()
    (
        db.collection("curricula")
        .document(curriculum_id)
        .collection("documents")
        .document(doc_id)
        .delete()
    )


# ── Analysis Results ─────────────────────────────────────────────

def save_analysis_result(curriculum_id: str, result: dict) -> str:
    db = get_db()
    ref = (
        db.collection("curricula")
        .document(curriculum_id)
        .collection("results")
        .document()
    )
    ref.set({**result, "created_at": SERVER_TIMESTAMP})
    return ref.id


def get_analysis_result(curriculum_id: str) -> dict | None:
    db = get_db()
    docs = (
        db.collection("curricula")
        .document(curriculum_id)
        .collection("results")
        .order_by("created_at", direction=firestore.Query.DESCENDING)
        .limit(1)
        .stream()
    )
    for d in docs:
        return {"id": d.id, **d.to_dict()}
    return None


def update_curriculum_status(curriculum_id: str, status: str):
    db = get_db()
    db.collection("curricula").document(curriculum_id).update({"analysis_status": status})


# ── Overlap ──────────────────────────────────────────────────────

def save_overlap(id_a: str, id_b: str, data: dict):
    db = get_db()
    key = f"{min(id_a, id_b)}_{max(id_a, id_b)}"
    db.collection("overlaps").document(key).set({**data, "created_at": SERVER_TIMESTAMP})


def get_overlap(id_a: str, id_b: str) -> dict | None:
    db = get_db()
    key = f"{min(id_a, id_b)}_{max(id_a, id_b)}"
    doc = db.collection("overlaps").document(key).get()
    if not doc.exists:
        return None
    return doc.to_dict()


# ── Helpers ──────────────────────────────────────────────────────

def _delete_subcollection(db, path: str, batch_size: int = 100):
    parts = path.strip("/").split("/")
    ref = db
    for i, part in enumerate(parts):
        if i % 2 == 0:
            ref = ref.collection(part)
        else:
            ref = ref.document(part)
    docs = list(ref.limit(batch_size).stream())
    while docs:
        for doc in docs:
            doc.reference.delete()
        docs = list(ref.limit(batch_size).stream())

def save_evaluation(data: dict) -> str:
    db = get_db()
    ref = db.collection("evaluations").document()
    ref.set({**data, "created_at": SERVER_TIMESTAMP})
    return ref.id

def get_all_evaluations() -> list:
    db = get_db()
    docs = db.collection("evaluations").stream()
    return [{"id": doc.id, **doc.to_dict()} for doc in docs]
