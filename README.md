# 📚 CurricuLens-PLN

<div align="center">
  <img alt="Status" src="https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge">
  <img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=for-the-badge">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python">
  <img alt="Firebase" src="https://img.shields.io/badge/Firebase-Hosting-yellow?style=for-the-badge&logo=firebase">
</div>

<p align="center">
  <strong>Sistem Analisis Kurikulum Berbasis Model Pembelajaran Mesin (MLM Fine-Tuned IndoBERT + BERTopic)</strong>
</p>

## 🌟 Tentang Proyek

**CurricuLens-PLN** adalah aplikasi modern berbasis web yang dirancang khusus untuk menganalisis dokumen kurikulum kelistrikan PLN. Sistem ini dapat memproses berbagai format dokumen pembelajaran (PDF, DOCX, PPTX), mengekstrak topik utama, membuat ringkasan ekstraktif yang komprehensif, serta menganalisis tingkat tumpang tindih (*overlap*) materi antar kurikulum.

## ✨ Fitur Utama

- 📂 **Manajemen Dokumen Kurikulum**: Unggah dan kelola silabus, materi tayang, dan handout dengan drag-and-drop. Dukungan multi-format (PDF, DOCX, PPTX).
- 🧠 **Topic Modeling (BERTopic)**: Mengekstrak klaster topik dan kata kunci penting secara otomatis dari dokumen menggunakan model `snickerdust/FT-PLN-IndoBERT`.
- 📝 **Extractive Summarization**: Menghasilkan ringkasan kalimat terbaik menggunakan metode *Centroid-Cosine Similarity* dengan *Trigram Blocking* untuk mencegah repetisi.
- 🔄 **Analisis Tumpang Tindih (Overlap)**: Secara otomatis mendeteksi persentase kemiripan materi antar kurikulum berdasarkan *Jaccard Index* dan kesamaan konsep semantik.
- 🎨 **Antarmuka Modern & Responsif**: Dibangun dengan Vanilla JavaScript dan Tailwind CSS, memberikan pengalaman pengguna yang mulus dengan fitur pencarian cerdas, Dark Mode, dan Auto-Wakeup Server Overlay.

## 🛠️ Teknologi yang Digunakan

### Frontend
- HTML5, Vanilla JavaScript (ES6+)
- Tailwind CSS
- Firebase Hosting & Firestore

### Backend
- Python (Flask)
- Sentence-Transformers (`snickerdust/FT-PLN-IndoBERT`)
- BERTopic & UMAP & HDBSCAN
- Docker & Hugging Face Spaces

## 🚀 Panduan Memulai Cepat

### Prasyarat
- [Node.js](https://nodejs.org/) & Firebase CLI
- [Docker](https://www.docker.com/) (Opsional, untuk menjalankan backend secara lokal)
- Akun Firebase (untuk database Firestore)

### Menjalankan Frontend
1. Clone repositori ini:
   ```bash
   git clone https://github.com/snickerdust/CurricuLens-PLN.git
   ```
2. Buka folder proyek dan jalankan server statis lokal (misal menggunakan Live Server di VSCode).

### Menjalankan Backend Lokal
1. Masuk ke direktori `backend`:
   ```bash
   cd backend
   ```
2. Install dependensi (disarankan menggunakan *virtual environment*):
   ```bash
   pip install -r requirements.txt
   ```
3. Tambahkan `firebase_credentials.json` ke dalam folder `backend`.
4. Jalankan aplikasi Flask:
   ```bash
   python app.py
   ```



## 📄 Lisensi
Proyek ini dilisensikan di bawah lisensi Apache-2.0.
