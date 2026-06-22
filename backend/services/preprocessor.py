"""
Preprocessing Pipeline v4 — ported from 02_preprocessing_v4 with cosine dataset.ipynb
Semua fungsi dan konstanta identik dengan notebook.
"""
import re
import nltk
from functools import lru_cache
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import (
    StopWordRemoverFactory, StopWordRemover, ArrayDictionary
)

# ── Lazy-init heavy objects ───────────────────────────────────────
_stemmer = None
_sw_remover = None

def get_stemmer():
    global _stemmer
    if _stemmer is None:
        _stemmer = StemmerFactory().create_stemmer()
    return _stemmer

def get_sw_remover():
    global _sw_remover
    if _sw_remover is None:
        # Base Sastrawi
        sw_factory = StopWordRemoverFactory()
        sw_list = sw_factory.get_stop_words()
        
        # Custom BI — domain kurikulum PLN
        sw_domain_id = [
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
        sw_domain_en = [
            'the', 'and', 'of', 'in', 'to', 'is', 'for', 'with', 'that',
            'this', 'from', 'are', 'or', 'on', 'an', 'as', 'at', 'be', 'by',
            'it', 'not', 'was', 'but', 'we', 'all', 'can', 'has', 'have',
            'will', 'one', 'if', 'each', 'so', 'its', 'do', 'no', 'up',
            'out', 'about', 'into', 'then', 'than', 'also', 'other', 'which',
            'learning', 'steering', 'commitee', 'corporate', 'university',
            'mandatori', 'program',
        ]
        
        all_sw = list(set(sw_list + sw_domain_id + sw_domain_en))
        _sw_remover = StopWordRemover(ArrayDictionary(all_sw))
    return _sw_remover


# ── Regex Patterns ────────────────────────────────────────────────

_RE_URL           = re.compile(r'https?://\S+|www\.\S+|\S+@\S+\.\S+')
_RE_TAGS          = re.compile(
    r'\[(JUDUL/SUB|TABEL START|TABEL END|NUMBERING|GAMBAR|HEADER|FOOTER|TABEL PDF|TABEL PPTX)\]',
    re.IGNORECASE
)
_RE_SLIDE         = re.compile(r'-{2,}\s*Slide\s*\d+\s*-{2,}', re.IGNORECASE)
_RE_TABLE_BORDER  = re.compile(r'[|\-]{3,}|[_]{3,}')
_RE_TABLE_HEADER  = re.compile(
    r'\b(no|nomor)\s*[\|:]?\s*(pokok\s*bahasan|rincian|uraian|kriteria|hasil\s*belajar)',
    re.IGNORECASE
)
_RE_CTRL_CHARS    = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_RE_SPECIAL       = re.compile(r'[^\w\s]')
_RE_SERIAL_NUM    = re.compile(r'\b\d{7,}\b')
_RE_LONE_DIGIT    = re.compile(r'\b\d{1,2}\b')
_RE_WHITESPACE    = re.compile(r'[\t\r\n]+|\s{2,}')

# v4 Noise Patterns
_RE_META_HEADER = re.compile(
    r'^(referensi|daftar\s*(gambar|tabel|isi|singkatan)|lampiran'
    r'|kata\s*pengantar|sambutan|pendahuluan\s*:'
    r'|persyaratan\s*peserta|hubungan\s*dengan\s*profesi'
    r'|strategi\s*pelaksanaan|level\s*evaluasi'
    r'|sertifikat\s*:'
    r'|lingkup\s*bahasan|metode\s*pembelajaran)',
    re.IGNORECASE
)
_RE_SILABUS_TEMPLATE = re.compile(
    r'^(hasil belajar\s*:'
    r'|setelah mengikuti.{0,50}(peserta|mampu)'
    r'|setelah menyelesaikan.{0,50}(peserta|mampu)'
    r'|tujuan\s*(pembelajaran|pelatihan|instruksional)\s*:'
    r'|pembelajaran ini disusun'
    r'|sertifikat (pembelajaran|diberikan)'
    r'|evaluasi (yang dilakukan|level)'
    r'|instruktur.{0,30}(soal|evaluasi)'
    r'|peserta menyelesaikan tugas)',
    re.IGNORECASE
)
_RE_CAPTION = re.compile(
    r'^(gambar|tabel|grafik|diagram|foto|bagan|skema|ilustrasi)\s+\d+[\.\s]',
    re.IGNORECASE
)
_RE_JABATAN = re.compile(
    r'(supervisor|asisten|manager|kepala|staf|officer|engineer|teknisi)'
    r'(\s+\w+){1,4}'
    r'(\s+(supervisor|asisten|manager|kepala|staf|officer|engineer|teknisi))',
    re.IGNORECASE
)


# ── Functions ─────────────────────────────────────────────────────

def regex_cleaning(teks: str) -> str:
    if not isinstance(teks, str) or not teks.strip():
        return ''
    teks = _RE_URL.sub(' ', teks)
    teks = _RE_TAGS.sub(' ', teks)
    teks = _RE_SLIDE.sub(' ', teks)
    teks = _RE_TABLE_BORDER.sub(' ', teks)
    teks = _RE_CTRL_CHARS.sub(' ', teks)
    teks = teks.lower()
    teks = _RE_TABLE_HEADER.sub(' ', teks)
    teks = _RE_SPECIAL.sub(' ', teks)
    teks = _RE_SERIAL_NUM.sub(' ', teks)
    teks = _RE_LONE_DIGIT.sub(' ', teks)
    teks = _RE_WHITESPACE.sub(' ', teks)
    return teks.strip()

def remove_stopwords(teks: str) -> str:
    return get_sw_remover().remove(teks)

PROTECTED_WORDS = {
    'bertegangan', 'tegangannya', 'penegang', 'tegangan', 'arus', 'daya', 
    'frekuensi', 'impedansi', 'resistansi', 'kapasitansi', 'induktansi',
    'transformator', 'distribusi', 'transmisi', 'instalasi', 'konstruksi', 
    'isolasi', 'konduksi', 'konduktor', 'busbar', 'grounding', 'fasa', 'netral',
    'pengukuran', 'perlindungan', 'pembatas', 'pengaman', 'penyambungan', 
    'pemasangan', 'pemeriksaan', 'pengoperasian', 'pemeliharaan',
    'inspeksi', 'evaluasi', 'analisis', 'analisa', 'direktori', 'berbasis', 
    'berbeda', 'tersebut', 'terhadap', 'terdapat', 'merupakan', 'dilakukan', 
    'digunakan', 'trafo', 'gardu', 'relay', 'fuse', 'kabel', 'panel',
}

def stemming_protected(teks: str) -> str:
    stemmer = get_stemmer()
    hasil = []
    for tok in teks.split():
        if tok.lower() in PROTECTED_WORDS:
            hasil.append(tok)
        else:
            hasil.append(stemmer.stem(tok))
    return ' '.join(hasil)

LEMMA_DICT = {
    'membaca': 'baca', 'membuat': 'buat', 'menggunakan': 'guna',
    'melakukan': 'laku', 'menjelaskan': 'jelas',
    'menganalisa': 'analisis', 'menganalisis': 'analisis',
    'mengevaluasi': 'evaluasi', 'menghitung': 'hitung',
    'memasang': 'pasang', 'memeriksa': 'periksa',
    'menghubungkan': 'hubung', 'menyambungkan': 'sambung',
    'mengoperasikan': 'operasi', 'memahami': 'paham',
    'mengetahui': 'tahu', 'menerapkan': 'terapkan',
    'mengukur': 'ukur', 'memelihara': 'pelihara',
    'menginspeksi': 'inspeksi', 'mengidentifikasi': 'identifikasi',
    'mendeteksi': 'deteksi', 'memperbaiki': 'perbaiki',
    'merawat': 'rawat', 'mengganti': 'ganti',
    'pengukuran': 'ukur', 'pemasangan': 'pasang',
    'pemeriksaan': 'periksa', 'pengoperasian': 'operasi',
    'penganalisaan': 'analisis', 'pendistribusian': 'distribusi',
    'penghubungan': 'hubung', 'pemahaman': 'paham',
    'pengetahuan': 'tahu', 'penerapan': 'terapkan',
    'berpendidikan': 'didik', 'kelistrikan': 'listrik',
    'keamanan': 'aman', 'ketentuan': 'tentu',
    'perlengkapan': 'lengkap', 'perlindungan': 'lindung',
    'persyaratan': 'syarat', 'pertimbangan': 'timbang',
    'pemeliharaan': 'pelihara', 'perawatan': 'rawat',
    'penggantian': 'ganti', 'perbaikan': 'perbaiki',
    'pembebanan': 'beban', 'pembangkitan': 'bangkit',
    'pelaksanaan': 'laksana', 'pelayanan': 'layanan',
    'trafo': 'transformator', 'travo': 'transformator',
    'jardis': 'jaringan distribusi',
    'kwh': 'kilowatthour', 'kvar': 'kilovar',
    'mva': 'megavoltampere', 'kva': 'kilovoltampere',
    'kv': 'kilovolt',
    'app': 'alat_pembatas_pengaman',
    'amr': 'automatic_meter_reading',
    'dlpd': 'daya_listrik_pencurian_distribusi',
    'spln': 'standar_pln',
    'sop': 'standar_operasional_prosedur',
    'k3': 'keselamatan_kesehatan_kerja',
}

def lemmatization(teks: str) -> str:
    if not isinstance(teks, str) or not teks.strip():
        return ''
    return ' '.join(LEMMA_DICT.get(tok, tok) for tok in teks.split())

def _is_noise_sentence(kalimat: str) -> tuple:
    k = str(kalimat).strip()
    if '[NUMBERING]' in k or '[JUDUL' in k:
        return True, 'tag_sisa'
    if k.count('|') >= 2:
        return True, 'baris_tabel'
    if re.search(r'^[\[\(]?\s*\d+[\s\.,]*\s*(JP|H|HE)\b', k, re.I):
        return True, 'baris_JP_HE'
    if re.match(r'^[\d\s\.\,\;\:\-]+$', k):
        return True, 'hanya_angka'
    
    # v4 Filters
    if len(k.split()) < 8:
        return True, 'terlalu_pendek'
    if _RE_META_HEADER.match(k):
        return True, 'metadata_admin'
    if _RE_SILABUS_TEMPLATE.match(k):
        return True, 'template_silabus'
    if _RE_CAPTION.match(k):
        return True, 'caption'
    if k.count('\t') >= 2:
        return True, 'dominasi_tab'
    
    tokens = k.split()
    if tokens:
        n_alpha = sum(1 for w in tokens if re.search(r'[a-zA-Z]', w))
        if n_alpha / len(tokens) < 0.5:
            return True, 'dominasi_angka_kode'
            
    if _RE_JABATAN.search(k) and len(k.split()) < 20:
        return True, 'daftar_jabatan'
        
    return False, ''

def tokenisasi_kalimat(teks_original: str) -> list:
    if not isinstance(teks_original, str) or not teks_original.strip():
        return []
    
    teks_tmp = teks_original.replace('\t', ' ')
    teks_tmp = _RE_SLIDE.sub(' ', teks_tmp)
    teks_tmp = _RE_TAGS.sub(' ', teks_tmp)
    teks_tmp = _RE_CTRL_CHARS.sub(' ', teks_tmp)
    teks_tmp = re.sub(r'https?://\S+|www\.\S+', ' ', teks_tmp)

    kalimat_raw = re.split(
        r'(?<=[.!?])\s+(?=[A-Z])|'
        r'(?<=[.!?])\s*\n|'
        r'\n{2,}|'
        r'(?<=\.)\s{2,}|'
        r'\n(?=[A-Z])',
        teks_tmp
    )

    kalimat_bersih = []
    for k in kalimat_raw:
        k = k.strip()
        is_noise, _ = _is_noise_sentence(k)
        if not is_noise:
            kata_valid = [w for w in k.split() if re.search(r'[a-zA-Z]', w)]
            if len(kata_valid) >= 8:
                kalimat_bersih.append(k)
    return kalimat_bersih

def split_to_sentences(teks: str) -> list[str]:
    return tokenisasi_kalimat(teks)

def preprocess_v4(teks: str) -> str:
    clean = regex_cleaning(teks)
    nosw  = remove_stopwords(clean)
    stem  = stemming_protected(nosw)
    return lemmatization(stem)

# ── Detailed Curation Pipeline ──────────────────────────────────

_RE_EXPLICIT_EXCL = re.compile(
    r'^(cover|daftar\s*isi|daftar\s*gambar|daftar\s*tabel|lampiran|kata\s*pengantar|sambutan)', 
    re.IGNORECASE
)

def curate_documents_v4(documents: list[dict], threshold: float = 0.9) -> tuple[list[dict], dict, dict]:
    """
    Detailed curation exactly matched with 01_kurasi_data_cosine.ipynb.
    Returns: (kept_docs, ignored_info, inclusion_info)
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    if not documents:
        return [], {}, {}

    ignored_info = {}
    inclusion_info = {}
    
    # ── Step 1: DUPLIKAT BARIS (exact match content) ──
    unique_texts = {}
    after_exact = []
    for d in documents:
        txt = d.get('teks_lemma', '') or d.get('teks_original', '')
        if txt in unique_texts:
            ignored_info[d['id']] = "⚠️ DUPLIKAT BARIS EXACT — baris identik sudah ada sebelumnya"
        else:
            unique_texts[txt] = d['id']
            after_exact.append(d)

    # ── Step 2: Filter: Metadata & Terlalu Sedikit Kata (< 150) ──
    after_size = []
    for d in after_exact:
        fname = d.get('filename', '').lower()
        txt = d.get('teks_lemma', '') or d.get('teks_original', '')
        words = txt.split()
        sentences = tokenisasi_kalimat(d.get('teks_original', ''))
        
        # Cell 4 notebook specific exclusions
        if _RE_EXPLICIT_EXCL.match(fname) or "cover" in fname:
            ignored_info[d['id']] = "File dikecualikan secara eksplisit (cover/metadata/non-substantif)"
        elif len(words) < 150:
            ignored_info[d['id']] = f"Jumlah kata terlalu sedikit ({len(words)} kata < minimum 150 kata) — tidak substantif"
        elif len(sentences) == 0:
            ignored_info[d['id']] = "Cover / Metadata (Tidak ditemukan kalimat substantif)"
        else:
            after_size.append(d)

    if not after_size:
        return [], ignored_info, inclusion_info

    # ── Step 3: COSINE DUPLIKAT ──
    SUMBER_PRIORITY = {
        '1. kursil': 1,
        '2. handout': 2,
        '1. handout': 2,
        '3. materi tayang': 3,
        '2. materi tayang': 3,
        '3. bahan tayang': 3,
        '2. bahan tayang': 3,
    }

    def get_prio(sumber: str) -> int:
        s = sumber.lower().strip()
        for k, v in SUMBER_PRIORITY.items():
            if k in s: return v
        return 99

    texts = [d.get('teks_lemma', '') or d.get('teks_original', '') for d in after_size]
    try:
        vec = TfidfVectorizer(min_df=1, max_df=1.0)
        tfidf = vec.fit_transform(texts)
        sim = cosine_similarity(tfidf)
    except:
        # Fallback inclusion info if TF-IDF fails
        for d in after_size: inclusion_info[d['id']] = "Lolos seleksi (TF-IDF failure fallback)"
        return after_size, ignored_info, inclusion_info

    n = len(after_size)
    to_drop = set()
    for i in range(n):
        for j in range(i + 1, n):
            score = sim[i, j]
            if score < threshold: continue
            if i in to_drop and j in to_drop: continue
            
            p1, p2 = get_prio(after_size[i]['sumber']), get_prio(after_size[j]['sumber'])
            k1, k2 = after_size[i].get('word_count', 0), after_size[j].get('word_count', 0)

            if p1 > p2: drop_idx, keep_idx = i, j
            elif p2 > p1: drop_idx, keep_idx = j, i
            else:
                if k1 <= k2: drop_idx, keep_idx = i, j
                else: drop_idx, keep_idx = j, i

            if drop_idx not in to_drop:
                nama_keep = after_size[keep_idx]['filename']
                ignored_info[after_size[drop_idx]['id']] = f"COSINE DUPLIKAT (score={round(float(score),3)} >= {threshold}) dengan \"{nama_keep}\""
                to_drop.add(drop_idx)

    final_kept = []
    for idx, d in enumerate(after_size):
        if idx not in to_drop:
            final_kept.append(d)
            # Assign Inclusion Reason based on notebook Cell 5
            s = d['sumber'].lower()
            if 'kursil' in s or 'kurikulum' in s or 'silabus' in s:
                inclusion_info[d['id']] = "Kursil/Silabus — wajib masuk sebagai anchor analisis overlap kurikulum"
            elif 'handout' in s:
                inclusion_info[d['id']] = f"Handout — materi utama untuk extractive summarization"
            else:
                inclusion_info[d['id']] = f"Materi Tayang/Bahan Tayang — konten unik, lolos seleksi cosine"

    return final_kept, ignored_info, inclusion_info


def build_corpus_from_documents(documents: list[dict]) -> list[dict]:
    all_sents = []
    for d in documents:
        weight = 1.0
        if "Kurikulum" in d["sumber"] or "Silabus" in d["sumber"]: weight = 2.0
        elif "Materi" in d["sumber"] or "Bahan" in d["sumber"]: weight = 1.8
        
        kalimat_list = tokenisasi_kalimat(d.get('teks_original', ''))
        for pos, sent in enumerate(kalimat_list):
            if len(sent.strip()) < 25: continue
            all_sents.append((sent, {
                'doc_id': d.get('doc_id'),
                'filename': d.get('filename'),
                'sumber': d['sumber'],
                'source_weight': weight,
                'posisi': pos
            }))
    
    unique_sents = []
    seen_sets = []
    for sent, meta in all_sents:
        words = set(sent.lower().split())
        if len(words) < 5: continue
        is_dup = any(len(words & sw) / len(words | sw) >= 0.85 for sw in seen_sets if len(words | sw) > 0)
        if not is_dup:
            unique_sents.append((sent, meta))
            seen_sets.append(words)

    corpus = []
    for sent, meta in unique_sents:
        corpus.append({
            'kalimat_original': sent,
            'kalimat_lemma':    preprocess_v4(sent),
            'source_weight':    meta['source_weight'],
            'sumber':           meta['sumber'],
            'doc_id':           meta['doc_id'],
            'filename':         meta['filename'],
            'posisi_kalimat':   meta['posisi'],
        })
    return corpus


def compute_overlap(summary_a: list[dict], summary_b: list[dict],
                    name_a: str, name_b: str) -> dict:
    """
    Compute topic overlap between two curriculum summaries.
    Includes sentence similarity and common keywords.
    """
    import re
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

    if not summary_a or not summary_b:
        return {'score': 0, 'pairs': [], 'common_keywords': [], 'desc': 'Salah satu kurikulum belum dianalisis.'}

    # Load stopword list to remove conjunctions and common non-technical words
    sw_list = StopWordRemoverFactory().get_stop_words()
    sw_domain = [
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
        'the', 'and', 'of', 'in', 'to', 'is', 'for', 'with', 'that',
        'this', 'from', 'are', 'or', 'on', 'an', 'as', 'at', 'be', 'by',
        'it', 'not', 'was', 'but', 'we', 'all', 'can', 'has', 'have',
        'will', 'one', 'if', 'each', 'so', 'its', 'do', 'no', 'up',
        'out', 'about', 'into', 'then', 'than', 'also', 'other', 'which',
        'learning', 'steering', 'commitee', 'corporate', 'university',
        'mandatori', 'program', 'dengan', 'yang', 'untuk', 'pada', 'dari',
        'adalah', 'dalam', 'bahwa', 'oleh', 'serta', 'suatu', 'secara', 'sudah',
        'dapat', 'akan', 'telah', 'bisa', 'yaitu', 'yakni', 'sebagai', 'atau', 'dan',
        'lainnya', 'lain'
    ]
    stopwords_set = set(sw_list + sw_domain)

    # Extract sentences and keywords
    # We use a bit more sentences for overlap to be more thorough
    sents_a = [r['kalimat'] for r in summary_a]
    sents_b = [r['kalimat'] for r in summary_b]
    
    # Keyword extraction with stopword filtering
    def get_keywords(sents):
        words = ' '.join(sents).lower()
        words = re.findall(r'\b[a-z]{4,}\b', words) # only words >= 4 chars
        return set(w for w in words if w not in stopwords_set)

    kw_a = get_keywords(sents_a)
    kw_b = get_keywords(sents_b)
    common_kw = sorted(list(kw_a.intersection(kw_b)))[:15] # top 15 common words

    # Calculation: Pure Jaccard Index of keywords (Set A & Set B)
    union_kw = kw_a.union(kw_b)
    intersection_kw = kw_a.intersection(kw_b)
    if len(union_kw) == 0:
        overlap_score = 0.0
    else:
        overlap_score = round((len(intersection_kw) / len(union_kw)) * 100, 1)

    # Jaccard Similarity tingkat kalimat untuk pencarian konsep serupa
    def sentence_jaccard(sent_a, sent_b):
        words_a = set(w for w in re.findall(r'\b[a-z]{3,}\b', sent_a.lower()) if w not in stopwords_set)
        words_b = set(w for w in re.findall(r'\b[a-z]{3,}\b', sent_b.lower()) if w not in stopwords_set)
        union = words_a.union(words_b)
        if not union:
            return 0.0
        return len(words_a.intersection(words_b)) / len(union)

    # Top matching pairs berdasarkan Jaccard Index kalimat
    flat = []
    for i, sent_a in enumerate(sents_a):
        for j, sent_b in enumerate(sents_b):
            sim_val = sentence_jaccard(sent_a, sent_b)
            flat.append((sim_val, i, j))
    flat.sort(key=lambda x: x[0], reverse=True)

    seen_a, seen_b, pairs = set(), set(), []
    for score, i, j in flat:
        if i in seen_a or j in seen_b: continue
        if score < 0.08: break # Threshold kemiripan kata kalimat (min 8%)
        pairs.append({
            'from_a':     sents_a[i],
            'from_b':     sents_b[j],
            'similarity': round(score * 100, 1),
        })
        seen_a.add(i)
        seen_b.add(j)
        if len(pairs) >= 10: break

    desc = (
        f"Analisis menemukan {len(pairs)} pasang konsep serupa dan {len(common_kw)} kata kunci bersama "
        f"antara {name_a} dan {name_b}."
        if pairs or common_kw else
        f"Tidak ditemukan tumpang tindih yang signifikan antara {name_a} dan {name_b}."
    )
    
    return {
        'score': overlap_score, 
        'pairs': pairs, 
        'common_keywords': common_kw,
        'desc': desc
    }
