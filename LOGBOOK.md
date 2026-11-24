# Logbook Eksperimen — Evaluasi CPMK 3 (Instance Segmentation)

- **Nama / NRP:** `[Isi Nama / NRP Anda]`
- **Digit Terakhir NRP:** `0` → **Variabel Eksperimen:** Ukuran Input (`320` / `640` / `960`)
- **Dataset Latih:** Panoptic Indoor Segmentation v5 (`panopticindoor/panoptic-indoor-segmentation` v5, 74 kelas)
- **Test Set Sendiri:** 40+ foto ruangan nyata kamera HP sendiri (`cdaaptnia/test-hp-ruangan`, min. 10 kondisi sulit)
- **Platform Komputasi:** Google Colab Free (T4 15GB) & Kaggle Notebooks (T4 15GB)

---

## 0. Hipotesis Sebelum Eksperimen

**Tanggal Entri:** `2026-11-24` *(ditulis sebelum eksperimen variasi resolusi 320 & 960 untuk Mask R-CNN dan Mask2Former)*

### A. Perbandingan 3 Keluarga Arsitektur (One-Stage CNN vs Two-Stage CNN vs Transformer)
1. **Model yang diprediksi unggul secara keseluruhan pada Panoptic Indoor v5 (~922 gambar, 74 kelas):**
   - **Prediksi:** **YOLOv8s-seg (One-Stage CNN)** dan **Mask R-CNN ResNet50-FPN-v2 (Two-Stage CNN)** akan mengungguli **Mask2Former Swin-Tiny (Transformer)** pada *low-data regime* (~700 gambar latih untuk 74 kelas).
   - **Alasan arsitektural:** Arsitektur berbasis CNN (YOLOv8 & Mask R-CNN) memiliki *inductive bias* spasial yang kuat (*translation equivariance* & *locality*), sehingga lebih cepat konvergen pada dataset berukuran kecil. Sebaliknya, Mask2Former (Transformer + *Hungarian matching* pada 100 *object queries*) membutuhkan volume data dan jumlah epoch jauh lebih besar untuk memetakan 74 kelas sekaligus.
2. **Keunggulan per tipe kelas objek ruangan:**
   - **Objek berukuran sedang–besar dan kaku** (seperti `bed`, `chair`, `couch`, `dining table`, `tv`, `door-stuff`): diprediksi kuat pada **YOLOv8s-seg** dan **Mask R-CNN** karena fitur FPN/PANet menangkap kontur besar dengan jelas.
   - **Objek kecil atau saling menutupi (*occlusion*)** (seperti `bottle`, `cup`, `mouse`, `cell phone`, `remote`, `book`): diprediksi lebih sulit bagi **YOLOv8s-seg** pada resolusi rendah karena *prototype mask* global hanya beresolusi 1/4 dari ukuran input, sedangkan **Mask R-CNN** memotong tiap proposal lewat RoIAlign `28×28` per objek sehingga batas antar-objek yang berdekatan lebih terisolasi (meski tepi kurva halus tetap dibatasi grid `28×28`).
   - **Domain Shift (`Test_Asli` → `Test_HP_Sendiri`):** Seluruh model diprediksi mengalami penurunan mAP pada foto HP sendiri akibat perbedaan pencahayaan (*low-light/backlight*), sudut ekstrem, serta variasi ruangan nyata di Indonesia/ITS dibanding dataset latih.

### B. Hipotesis Variabel NRP 0: Ukuran Input (`320` vs `640` vs `960`)
1. **`320px`:** Diprediksi menghasilkan mAP terendah dan *domain shift* terbesar di ketiga model. Pada YOLOv8s-seg, *prototype mask* hanya `80×80` sehingga objek kecil (`bottle`, `cup`, `mouse`) hilang atau mask-nya bocor. Pada Mask R-CNN & Mask2Former, fitur *backbone* terlalu kasar untuk membedakan 74 kelas indoor.
2. **`640px`:** Diprediksi menjadi titik keseimbangan (*sweet spot*) terbaik antara akurasi deteksi/segmentasi, ukuran *batch* yang stabil (BatchNorm tidak *noisy*), dan waktu komputasi GPU.
3. **`960px`:** Diprediksi meningkatkan presisi tepi mask pada **IoU ketat (`Mask mAP@50-95`)** terutama pada objek kecil/detail (karena *prototype mask* YOLO naik ke `240×240` dan *feature map* FPN/Swin lebih padat). Namun, karena keterbatasan VRAM 15GB memaksa `batch_size` diperkecil (`batch=4` di YOLO, `batch=2` di Mask R-CNN, `batch=1` di Mask2Former), stabilitas gradien/BatchNorm berpotensi membuat `mAP@50` sedikit fluktuatif dibanding `640px`.

*(Catatan evaluasi pasca-eksperimen: hasil aktual dari ketiga model pada `320 / 640 / 960` akan dibandingkan secara objektif terhadap hipotesis di atas pada bagian Analisis Laporan.)*

---

## 1. Catatan Waktu GPU & Bukti Log Training (3 Model × 3 Resolusi — NRP 0)

| No | Model | Keluarga Arsitektur | Ukuran Input | Batch Size | Epoch | Platform / GPU | Waktu Training (menit) | File Log Asli (`results.csv`) |
|----|-------|---------------------|--------------|------------|-------|----------------|------------------------|-------------------------------|
| 1 | YOLOv8s-seg | One-Stage CNN | 320 | 16 | 50 | Colab Free T4 | | `logs/yolov8_320_results.csv` |
| 2 | YOLOv8s-seg | One-Stage CNN | 640 | 16 | 50 | Colab Free T4 | | `logs/yolov8_640_results.csv` |
| 3 | YOLOv8s-seg | One-Stage CNN | 960 | 4 | 50 | Colab Free T4 | | `logs/yolov8_960_results.csv` |
| 4 | Mask R-CNN (R50-FPN-v2) | Two-Stage CNN | 320 | 4 | 15 | Kaggle T4 | | `logs/maskrcnn_320_results.csv` |
| 5 | Mask R-CNN (R50-FPN-v2) | Two-Stage CNN | 640 | 4 | 15 | Kaggle T4 | | `logs/maskrcnn_640_results.csv` |
| 6 | Mask R-CNN (R50-FPN-v2) | Two-Stage CNN | 960 | 2 | 15 | Kaggle T4 | | `logs/maskrcnn_960_results.csv` |
| 7 | Mask2Former (Swin-Tiny) | Transformer | 320 | 2×4 (eff. 8) | 15 | Kaggle T4 | | `logs/mask2former_320_results.csv` |
| 8 | Mask2Former (Swin-Tiny) | Transformer | 640 | 2×4 (eff. 8) | 15 | Kaggle T4 | | `logs/mask2former_640_results.csv` |
| 9 | Mask2Former (Swin-Tiny) | Transformer | 960 | 1×8 (eff. 8) | 15 | Kaggle T4 | | `logs/mask2former_960_results.csv` |

---

## 2. Logbook Kendala & Keputusan Teknis

| Tanggal | Tahap | Kendala yang Ditemui | Keputusan / Solusi yang Diambil | Dampak |
|---------|-------|----------------------|---------------------------------|--------|
| `2026-11-24` | Persiapan Dataset | Sempat menyiapkan foto pakaian (baju/jaket/kaos), padahal soal mewajibkan *Panoptic Indoor Segmentation v5* (74 kelas objek ruangan) | Beralih penuh ke *Panoptic Indoor Segmentation v5* dan mengambil 40+ foto ruangan nyata dengan kamera HP sendiri (≥10 kondisi sulit) | 100% sesuai ketentuan tugas dan 74 kelas dataset |
| `2026-11-24` | Anotasi Test Set | Roboflow hanya mengekspor 27 kelas yang muncul di test set HP (bukan 74 kelas penuh), berisiko *class ID mismatch* pada YOLOv8 | Menulis skrip remap otomatis ID kelas `custom_hp_yolo` ke daftar resmi 74 kelas v5 dan memverifikasi kecocokan 100% | Evaluasi YOLOv8 pada `Test_HP_Sendiri` berjalan akurat tanpa error ID kelas |
| `2026-11-24` | Training YOLOv8 960px | Potensi *CUDA Out of Memory* di T4 15GB pada resolusi `960` | Menurunkan `batch=4` + mengaktifkan `amp=True` (FP16) | Training `320 / 640 / 960` YOLOv8s-seg selesai stabil tanpa OOM |
| `2026-11-24` | Kuota GPU Colab Free | Muncul batas pemakaian GPU (*usage limits*) di Google Colab Free setelah training 3 ukuran YOLOv8s-seg | Bermigrasi ke **Kaggle Notebooks (Tesla T4 15GB, 30 jam/minggu)** untuk training Mask R-CNN & Mask2Former | Eksperimen berlanjut lancar dengan spesifikasi GPU identik (Tesla T4) |
| `2026-11-24` | Training Mask2Former | `KeyError: 0` pada `convert_segmentation_map_to_binary_masks_fast` di `transformers` versi baru Kaggle karena pixel background `0` tidak diabaikan | Mengubah `Mask2FormerCocoDataset` agar membangun binary tensor `mask_labels` dan `class_labels` langsung dari poligon COCO (tanpa melewati `segmentation_maps`) | Error teratasi 100% sekaligus mencegah mask objek *overlap* saling menimpa |
| `2026-11-24` | Keamanan Repo Git | Notebook Colab/Kaggle membutuhkan `ROBOFLOW_API_KEY` | Tidak pernah menaruh API Key asli di file `.ipynb` yang di-commit ke GitHub (memakai placeholder / `getpass`) + menambahkan `.gitignore` | Repositori aman untuk dipublikasikan |

---

## 3. Tabel Analisis 5 Kasus Kegagalan Berbasis Bukti

| No | Nama File Foto | Jenis Kegagalan | Kondisi Foto | Hipotesis Penyebab (Kaitkan dengan Mekanisme Model) | Eksperimen Kecil Uji Hipotesis | Hasil Uji (Terbukti / Tidak) |
|----|----------------|-----------------|--------------|-----------------------------------------------------|--------------------------------|------------------------------|
| 1 | | Mask bocor / tepi kasar | | Resolusi prototype mask YOLOv8 (`1/4` input: `80×80` di `320px`) vs grid `28×28` RoIAlign Mask R-CNN | Bandingkan inferensi `320px` vs `960px` pada foto yang sama | |
| 2 | | Objek tidak terdeteksi | Cahaya redup / backlight | Kontras tepi rendah di layer awal backbone menghambat ekstraksi fitur foreground | Terapkan preprocessing CLAHE lalu inferensi ulang | |
| 3 | | Salah kelas | Oklusi / sudut ekstrem | Fitur lokal terpotong; kelas mirip di 74 kelas v5 | Bandingkan prediksi antar 3 arsitektur + cek confidence | |
| 4 | | Satu objek terpecah | | Proposal ganda / NMS IoU threshold kurang tepat | Uji perubahan NMS IoU / confidence threshold | |
| 5 | | | | | |
