import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sentence_transformers import models as st_models
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
import torch
import re
from collections import defaultdict
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Model for semantic representation on Hugging Face Model Hub
MODEL_NAME = "snickerdust/FT-PLN-IndoBERT"

device = "cuda" if torch.cuda.is_available() else "cpu"
model = None

def get_model():
    global model
    if model is None:
        print(f"Loading BERT model ({MODEL_NAME}) on {device}...")
        # Load MLM fine-tuned IndoBERT using Transformer + Pooling (no Dense) as M2-IndoBERT-FT
        word_emb = st_models.Transformer(MODEL_NAME)
        pooling  = st_models.Pooling(
            word_emb.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
        )
        model = SentenceTransformer(modules=[word_emb, pooling], device=device)
    return model

def normalize_text(s: str) -> str:
    """Collapse multiple whitespace ke single space + strip."""
    if not isinstance(s, str):
        return ""
    return re.sub(r'\s+', ' ', s).strip()

def _clean_token(text: str) -> str:
    if not isinstance(text, str):
        return ''
    tokens = text.split()
    return ' '.join(
        t for t in tokens
        if len(t) >= 2 and t != '-' and not re.match(r'^\d+$', t)
    )

# ── Inisialisasi Stopword ───────────────────────────────────────
sw_factory   = StopWordRemoverFactory()
SW_SASTRAWI  = sw_factory.get_stop_words()   # list, bukan set

# Custom BI — domain kurikulum PLN
SW_DOMAIN_ID = [
    'daftar', 'isi', 'gambar', 'tabel', 'lampiran', 'halaman',
    'bab', 'subbab', 'bagian', 'nomor', 'hlm', 'hal',
    'peserta', 'pelatihan', 'mata', 'pelajaran',
    'tujuan', 'durasi', 'jp', 'kompetensi', 'modul',
    'edisi', 'tahun', 'revisi', 'versi',
    'kata', 'sambutan', 'pengantar', 'puji', 'syukur', 'panjatkan',
    'hadirat', 'allah', 'swt', 'rahmat', 'taufiq', 'hidayah',
    'berhasil', 'disusun', 'tepat', 'waktu', 'seiring', 'metamorfosa',
    'pln', 'pusdiklat', 'updl', 'lsc', 'kkj', 'pt', 'persero',
    'i', 'ii', 'iii', 'iv', 'vi', 'vii', 'viii', 'ix', 'xi', 'xii',
]

# Custom EN — kata non-teknis Bahasa Inggris
SW_DOMAIN_EN = [
    'the', 'and', 'of', 'in', 'to', 'is', 'for', 'with', 'that',
    'this', 'from', 'are', 'or', 'on', 'an', 'as', 'at', 'be', 'by',
    'it', 'not', 'was', 'but', 'we', 'all', 'can', 'has', 'have',
    'will', 'one', 'if', 'each', 'so', 'its', 'do', 'no', 'up',
    'out', 'about', 'into', 'then', 'than', 'also', 'other', 'which',
    'learning', 'steering', 'commitee', 'corporate', 'university',
    'mandatori', 'program',
]

# Gabungkan semua stopword dan ubah menjadi set untuk menghapus duplikat, lalu kembali ke list
FINAL_STOPWORDS = list(set(SW_SASTRAWI + SW_DOMAIN_ID + SW_DOMAIN_EN))

def build_bertopic(nr_topics, min_topic_size, min_cluster_size,
                   n_neighbors, n_components, min_dist, seed=42):
    """
    BERTopic dengan hyperparameter dari Bab 3 Tabel 3.9 dan stopword dari notebook.
    """
    vectorizer = CountVectorizer(
        ngram_range=(1, 2), 
        min_df=1, 
        stop_words=FINAL_STOPWORDS
    )
    return BERTopic(
        umap_model = UMAP(
            n_neighbors=n_neighbors, n_components=n_components,
            min_dist=min_dist, metric='cosine',
            random_state=seed, low_memory=True,
        ),
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_cluster_size, metric='euclidean',
            cluster_selection_method='eom', prediction_data=True,
        ),
        vectorizer_model = vectorizer,
        nr_topics      = nr_topics,
        min_topic_size = min_topic_size,
        verbose        = False,
        calculate_probabilities = False,
    )

def extract_summary(valid, sentences, embeddings, weights, topic_labels, num_sentences):
    """
    Extractive summarizer per cluster BERTopic.
    Scoring: cosine_similarity(kalimat, centroid_cluster) × source_weight (Common Fix #3)
    Trigram Blocking: kalimat tidak dipilih jika trigram-nya sudah ada (Y. Liu, 2019).
    """
    clusters = defaultdict(list)
    for i, (s, e, w, t) in enumerate(zip(sentences, embeddings, weights, topic_labels)):
        if t != -1:   # exclude outliers
            clusters[t].append((i, s, e, w))

    # Fallback: tidak ada cluster valid → fallback weight-based
    if not clusters:
        scored = sorted(zip(weights, range(len(sentences))), reverse=True)
        selected = [(i, sentences[i], weights[i], weights[i]) for _, i in scored[:num_sentences]]
    else:
        n_cl   = len(clusters)
        per_cl = max(1, num_sentences // n_cl)
        selected, used_trigrams = [], set()

        for tl in sorted(clusters):
            items = clusters[tl]
            idxs, sents_cl, embs_cl, ws_cl = zip(*items)
            centroid = np.mean(embs_cl, axis=0, keepdims=True)
            cosine_scores = cosine_similarity(np.array(embs_cl), centroid).flatten()
            # Ranking uses cosine * source_weight to select correct sentences
            ranking_scores = cosine_scores * np.array(ws_cl)
            picked   = 0
            for r in np.argsort(ranking_scores)[::-1]:
                if picked >= per_cl: break
                sent     = sents_cl[r]
                words    = sent.lower().split()
                trigrams = set(zip(words, words[1:], words[2:])) if len(words) >= 3 else set()
                if trigrams & used_trigrams: continue
                used_trigrams |= trigrams
                # Store the pure cosine similarity score as the score for display
                selected.append((idxs[r], sent, cosine_scores[r], ws_cl[r]))
                picked += 1

    selected.sort(key=lambda x: x[0])
    
    summary_list = []
    for rank, (idx, sent, score, weight) in enumerate(selected):
        summary_list.append({
            'rank':          rank + 1,
            'kalimat':       sent,
            'score':         round(float(score), 6),
            'source_weight': float(weight),
            'sumber':        valid[idx].get('sumber', ''),
            'doc_id':        valid[idx].get('doc_id', ''),
        })
    return summary_list

def run_bert_textrank(corpus: list[dict], n_summary: int | None = None) -> dict:
    """
    Menjalankan pipeline M2 (BERTopic + Centroid-Cosine Summarizer)
    yang telah diperbarui berdasarkan hyperparameter terbaik dari notebook.
    """
    if not corpus:
        return {'topics': [], 'summary': [], 'n_corpus': 0}

    valid = [
        row for row in corpus
        if len(_clean_token(row.get('kalimat_lemma', '')).split()) >= 4
    ]
    if not valid:
        valid = corpus

    # Normalisasi spasi kalimat asli (tidak lagi menggunakan kalimat_lemma untuk fitting BERT)
    texts_orig  = [normalize_text(row['kalimat_original']) for row in valid]
    weights     = [row.get('source_weight', 1.0) for row in valid]

    # Menentukan jumlah ringkasan kalimat (default = 8 dari Optuna LOTO)
    num_sentences = 8 if n_summary is None else n_summary

    # 1. Encoding menggunakan IndoBERT fine-tuned dengan batch_size = 32
    st_model = get_model()
    embeddings = st_model.encode(
        texts_orig,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    # 2. BERTopic dengan hyperparameter terbaik (LOTO Tuning)
    topics_list = []
    topic_labels = None
    topic_model = None

    if len(texts_orig) < 5:
        print("⚠️ Data terlalu sedikit untuk melakukan clustering topik.")
        topic_labels = [-1] * len(texts_orig)
    else:
        try:
            # Hyperparameter dari notebook M2 Best Config (LOTO)
            nr_topics = 5
            min_topic_size = 60
            min_cluster_size = 75
            n_neighbors = 30
            n_components = 10
            min_dist = 0.0

            topic_model = build_bertopic(
                nr_topics=nr_topics,
                min_topic_size=min_topic_size,
                min_cluster_size=min_cluster_size,
                n_neighbors=n_neighbors,
                n_components=n_components,
                min_dist=min_dist,
                seed=42
            )
            topic_labels, _ = topic_model.fit_transform(texts_orig, embeddings=embeddings)
        except Exception as ex:
            print(f"   ⚠️  Fallback ({ex})")
            topic_labels = [-1] * len(texts_orig)
            topic_model = None

    # --- PERUBAHAN: BLOK UNTUK MENAMPILKAN TOPIK ---
    if topic_model is not None:
        t_info = topic_model.get_topic_info()
        # Hitung jumlah topik aktif (kurangi 1 jika ada topik outlier -1)
        n_topics = len(t_info) - (1 if -1 in t_info['Topic'].values else 0)
        print(f'  Topik BERTopic : {n_topics} Topik Ditemukan')
        
        for t_id in t_info['Topic']:
            if t_id == -1: 
                continue # Skip topik -1 (Outliers/Noise)
            # Ambil 5 kata dengan probabilitas tertinggi untuk topik ini
            top_words = [word for word, _ in topic_model.get_topic(t_id)][:5]
            # Cetak topik beserta count/jumlah kalimatnya
            count = t_info[t_info['Topic'] == t_id]['Count'].values[0]
            print(f'    > Topik {t_id} (N={count}): {", ".join(top_words)}')
            
            # Buat topics_list untuk return API
            topics_list.append({
                'id': int(t_id),
                'name': f"Topik {t_id + 1}",
                'top_words': top_words,
                'weight': round(float(count) / len(texts_orig), 4)
            })
    else:
        print(f'  Topik BERTopic : Fallback digunakan (BERTopic tidak menghasilkan model)')
        topic_labels = [-1] * len(texts_orig)
    # -----------------------------------------------

    # 3. Ekstraksi Ringkasan dengan centroid-cosine dan trigram blocking
    summary_list = extract_summary(
        valid=valid,
        sentences=texts_orig,
        embeddings=embeddings,
        weights=weights,
        topic_labels=topic_labels,
        num_sentences=num_sentences
    )

    return {
        'topics':   topics_list,
        'summary':  summary_list,
        'n_corpus': len(valid),
    }
