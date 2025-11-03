## Laporan Ujian Tengah Semester (UTS) - Analisis dan Prediksi Harga iPhone

## Judul Proyek

Sistem Analisis Penjualan dan Prediksi Harga iPhone Menggunakan Machine Learning Berbasis Regresi

## 1. Tujuan Penelitian / Program

Tujuan utama dari program ini adalah untuk menyediakan dashboard interaktif berbasis web (Flask) yang dapat membantu pemilik toko atau analis pasar dalam membuat keputusan inventaris dan penetapan harga.

## Secara spesifik, program ini bertujuan untuk:

Analisis Pasar Lokal: Mengidentifikasi model iPhone mana yang paling diminati (paling banyak terjual) di lokasi toko tertentu.

Prediksi Harga: Memprediksi harga jual eceran suatu produk (berdasarkan model, warna, dan kapasitas) menggunakan model Machine Learning yang dilatih secara lokal.

Evaluasi Kinerja Model: Membandingkan kinerja dua metode regresi (Linear Regression dan Random Forest) untuk menentukan model mana yang paling efisien dalam memprediksi harga per kota.

## 2. Metode Machine Learning yang Digunakan

Program ini menggunakan dua algoritma utama dari kategori Supervised Learning dengan tipe Regresi (karena targetnya adalah memprediksi nilai numerik, yaitu Harga).

## A. Algoritma Regresi yang Digunakan

Algoritma

Kategori

Kelebihan dalam Konteks Ini

## 1. Regresi Linier (Linear Regression)

Supervised Learning

Sederhana, cepat dilatih, dan mudah diinterpretasikan. Baik untuk memodelkan hubungan harga yang stabil dan linier.

## 2. Random Forest Regressor

Supervised Learning (Ensemble)

Model yang kuat, dapat menangani hubungan non-linier dan data dengan kompleksitas tinggi. Cenderung memberikan akurasi yang lebih stabil. 

## B. Metrik Evaluasi

Kinerja model diukur menggunakan $R^2$ Score (Coefficient of Determination).

Tujuan $R^2$ Score: Mengukur seberapa baik model dapat mereplikasi data hasil observasi. Nilai yang mendekati 1 (atau 100%) menunjukkan bahwa model dapat menjelaskan hampir semua variabilitas harga.

## 3. Data yang Dipakai dan Preprocessing

## A. Data Awal (Dataset)

Nama File: data_penjualan_iphone_1000.xlsx

Jumlah Data: 1000 baris data (setelah proses cleaning).

Fitur (Variabel Input) yang Digunakan:

Model: Nama model iPhone (misalnya, iPhone 11, iPhone 14 Pro Max).

Warna: Varian warna (misalnya, Hitam, Merah, Ungu).

Kapasitas (GB): Ukuran memori.

Lokasi Toko: Kota tempat penjualan terjadi (digunakan untuk filtering).

Target (Variabel Output):

Harga (Rp): Nilai numerik harga jual eceran.

## B. Tahap Pra-pemrosesan Data (Preprocessing)

Program ini menerapkan tiga langkah krusial sebelum pelatihan:

Data Cleaning (Pembersihan Data):

Menggunakan fungsi df_full.drop_duplicates(inplace=True) untuk menghapus setiap baris data yang identik, memastikan integritas dan menghindari bias pelatihan.

## Feature Encoding:

Menggunakan LabelEncoder() dari Scikit-learn untuk mengubah variabel kategori (Model, Warna, Lokasi Toko) menjadi nilai numerik (Model_enc, Warna_enc, Lokasi_enc). Langkah ini wajib karena algoritma ML hanya bekerja dengan angka.

## Pembagian Data:

Data dibagi 80% untuk pelatihan (X_train, y_train) dan 20% untuk pengujian (X_test, y_test) menggunakan train_test_split.

## 4. Pembahasan Hasil dan Kinerja

## A. Kinerja Akurasi yang Sangat Tinggi

Akurasi kedua model (sekitar 99.5% - 99.9%) hampir sempurna.

Interpretasi: Akurasi tinggi ini menunjukkan bahwa harga eceran iPhone sangat stabil dan ditentukan secara ketat oleh fitur produk (Model, Warna, Kapasitas). Model ML tidak perlu bekerja keras untuk "menebak" harga, melainkan hanya menghafal pola harga dasar yang telah ditetapkan.

Implikasi: Karena $R^2$ hampir 100%, ini menunjukkan bahwa untuk prediksi harga eceran stabil, model Linear Regression sudah cukup (lebih cepat dan lebih ringan) dan kompleksitas Random Forest tidak memberikan peningkatan kinerja yang signifikan.

## B. Logika Prediksi Berdasarkan Popularitas

Program ini menerapkan logika filterisasi yang terperinci:

Analisis Popularitas Kumulatif: Program pertama kali mengidentifikasi Model iPhone yang paling banyak terjual secara total di lokasi yang dipilih (misalnya, Model X).

Penentuan Konfigurasi: Kemudian, program mengambil konfigurasi terlaris (Warna dan Kapasitas) dari Model X tersebut.

Prediksi: Harga diprediksi untuk konfigurasi terlaris Model X tersebut. Hal ini memastikan bahwa harga prediksi selalu relevan dengan produk paling diminati di pasar lokal tersebut, bukan sekadar produk yang paling mahal.

## C. Manfaat Aplikasi

Aplikasi ini berhasil menyajikan hasil Machine Learning dalam dashboard yang mudah digunakan, memungkinkan pengguna untuk mendapatkan:

Perbandingan penjualan model iPhone secara visual (Grafik).

Informasi produk terlaris secara global dan lokal.

Prediksi harga yang didasarkan pada model yang paling mewakili tren pasar lokal.