# Panduan Pengerjaan CPMK 3 — Instance Segmentation di Google Colab 

Proyek ini berisi **4 notebook Google Colab** yang dirancang khusus agar hemat kuota GPU **Colab Free (Tesla T4 15GB)**, menyimpan seluruh hasil/checkpoint otomatis ke Google Drive, dan memenuhi seluruh komponen tugas:
- Perbandingan **3 model dari 3 keluarga arsitektur berbeda**:
  1. `YOLOv8s-seg` (One-Stage CNN)
  2. `Mask R-CNN ResNet50-FPN-v2` (Two-Stage CNN)
  3. `Mask2Former Swin-Tiny` (Transformer)
- **Variabel NRP akhiran 0**: Eksperimen ukuran input **`320` / `640` / `960`**.
- **Test set sendiri (≥ 40 foto HP, min. 10 kondisi sulit)** + ekstraksi EXIF otomatis.
- **Evaluasi 2 test set** (*Test Asli* vs *Test HP Sendiri*) untuk analisis *Domain Shift*.
- **Bukti proses otomatis**: `results.csv` asli, `gpu_time_log.csv`, dan template logbook.

---

## Urutan Eksekusi di Google Colab Free

1. **Upload folder `notebooks/`** ke Google Drive Anda (atau buka masing-masing file `.ipynb` langsung di [Google Colab](https://colab.research.google.com/)).
2. **Upload 40+ foto asli dari HP** (jangan lewat WhatsApp agar EXIF tidak terhapus) ke Google Drive di:
   `MyDrive/CPMK3_Instance_Segmentation/raw_hp_photos/`
3. **Jalankan [`01_setup_and_exif.ipynb`](notebooks/01_setup_and_exif.ipynb)** *(pakai Runtime CPU saja)*:
   - Mengekstrak EXIF semua foto ke `logs/exif_metadata.csv`.
   - Mengunduh dataset utama dan dataset test set HP Anda dari Roboflow (format YOLOv8 & COCO Segmentation).
   - Memverifikasi bahwa nama & urutan kelas antara dataset utama dan test set HP **100% sama**.
4. **Jalankan [`02_train_yolov8_seg_nrp0.ipynb`](notebooks/02_train_yolov8_seg_nrp0.ipynb)** *(Runtime T4 GPU)*:
   - Melatih `YOLOv8s-seg` untuk ukuran input **320, 640, dan 960**.
   - Mengevaluasi ketiga ukuran di *Test Asli* dan *Test HP Sendiri*.
5. **Jalankan [`03_train_maskrcnn_and_mask2former.ipynb`](notebooks/03_train_maskrcnn_and_mask2former.ipynb)** *(Runtime T4 GPU)*:
   - Melatih `Mask R-CNN` dan `Mask2Former` dengan AMP (`fp16`) agar aman di VRAM 15GB.
   - Mengevaluasi dengan `pycocotools` pada *Test Asli* dan *Test HP Sendiri*.
   - *Tips Colab Free*: Anda bisa menjalankan Bagian 1 (Mask R-CNN) di sesi pertama, lalu Bagian 2 (Mask2Former) di sesi berikutnya karena checkpoint otomatis tersimpan di Google Drive.
6. **Jalankan [`04_domain_shift_and_failure_analysis.ipynb`](notebooks/04_domain_shift_and_failure_analysis.ipynb)**:
   - Membuat grafik *Domain Shift* dan grafik pengaruh ukuran input `320 / 640 / 960`.
   - Menjalankan eksperimen kecil uji hipotesis kegagalan (resolusi 320→960 & CLAHE untuk *low-light/backlight*) pada 5 foto gagal pilihan Anda.
7. **Lengkapi [`LOGBOOK_TEMPLATE.md`](LOGBOOK_TEMPLATE.md)** menggunakan file `logs/gpu_time_log.csv` dan `exports/summary_all_models.csv`.
