## Draf Laporan Ujian Tengah Semester (UTS)

## Judul: Analisis Komparatif Model Supervised dan Unsupervised Learning dalam Segmentasi dan Prediksi Harga Produk Elektronik

## I. Pendahuluan

Latar Belakang: Jelaskan bahwa Anda menggunakan dataset penjualan iPhone untuk memahami dinamika pasar di berbagai lokasi toko.

Tujuan Proyek: Mengimplementasikan dua jenis Machine Learning (ML) untuk mendapatkan wawasan bisnis ganda:

Memprediksi harga jual (menggunakan Supervised Learning).

Mengelompokkan produk berdasarkan nilai untuk segmentasi pasar (menggunakan Unsupervised Learning).

## II. Metodologi (Komparasi Model ML)

Jelaskan model yang Anda gunakan dan metrik perbandingannya:

Jenis ML

Model Digunakan

Tujuan Bisnis

Metrik Kinerja

Interpretasi Metrik

Supervised Learning

Linear Regression (LR)

Prediksi harga spesifik produk terlaris.

$R^2$ Score (dalam %)

Mengukur keandalan prediksi (seberapa akurat harga prediksi mendekati harga aktual). Nilai tinggi (mendekati 100%) menunjukkan model handal untuk penetapan harga.

Unsupervised Learning

K-Means Clustering

Segmentasi produk ke dalam klaster harga (Low, Mid, High Value).

Silhouette Score (Skor 0 hingga 1)

Mengukur kualitas klasterisasi. Nilai mendekati 1 menunjukkan klaster sangat padat dan terpisah dengan baik, menandakan segmentasi pasar yang jelas.

## III. Implementasi dan Analisis Hasil

Pre-processing Data:

Sebutkan bahwa Anda melakukan Label Encoding pada fitur non-numerik (Model, Warna) karena Linear Regression dan K-Means memerlukan input numerik.

Hasil Linear Regression (LR):

Sajikan hasil $R^2$ Score dari dashboard Anda (Contoh: "Akurasi rata-rata di berbagai kota adalah 99%.").

Wawasan: Jelaskan bahwa akurasi yang sangat tinggi mungkin disebabkan oleh harga eceran produk yang stabil. Hasil ini sangat berguna untuk Prediksi Harga Ideal (seperti yang terlihat di Canvas Anda).

Hasil K-Means Clustering:

Sajikan hasil Silhouette Score (Contoh: "Silhouette Score rata-rata adalah 0.85").

Wawasan: Jelaskan hasil klasterisasi per lokasi. Fokus pada Klaster yang memiliki produk terbanyak (Misal: "Di lokasi A, klaster Low-Value mendominasi, menandakan pasar sensitif harga"). Ini sangat berguna untuk Strategi Inventaris dan Pemasaran.

## IV. Kesimpulan dan Saran

Kesimpulan Komparatif:

Model Linear Regression berhasil menyediakan alat untuk Pengambilan Keputusan Taktis (menentukan harga jual).

Model K-Means berhasil memberikan alat untuk Analisis Strategis (memahami segmen dan struktur pasar).

Penekanan: Kedua model, meski berbeda jenis, memberikan nilai tambah yang unik dan saling melengkapi dalam wawasan bisnis.

Saran Pengembangan Lanjut:

Menggunakan model Regresi yang lebih kompleks (seperti Random Forest Regressor) jika $R^2$ Score Linear Regression tidak memuaskan.

Menggunakan teknik elbow method untuk menentukan nilai $K$ yang optimal pada K-Means, bukan menetapkan $K=3$ secara statis.

Lampiran: Sertakan screenshot dari Dashboard Canvas Anda yang menampilkan kedua metrik ($R^2$ Score dan Silhouette Score) serta hasil prediksi dan klasterisasi.