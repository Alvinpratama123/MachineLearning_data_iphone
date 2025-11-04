# from flask import Flask, render_template_string, request
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import r2_score
# import joblib
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import io
# import base64

# app = Flask(__name__)

# # ===== Load dataset dan persiapkan encoding =====
# df = pd.read_excel('data_penjualan_iphone_1000.xlsx')

# le_model = LabelEncoder()
# le_warna = LabelEncoder()
# le_lokasi = LabelEncoder()

# df['Model'] = le_model.fit_transform(df['Model'])
# df['Warna'] = le_warna.fit_transform(df['Warna'])
# df['Lokasi Toko'] = le_lokasi.fit_transform(df['Lokasi Toko'])

# X = df[['Model', 'Warna', 'Kapasitas (GB)', 'Lokasi Toko', 'Jumlah Terjual']]
# y = df['Harga (Rp)']

# # Linear Regression
# lr = LinearRegression().fit(X, y)

# # Random Forest
# try:
#     rf = joblib.load('model_randomforest.pkl')
# except:
#     rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
#     joblib.dump(rf, 'model_randomforest.pkl')

# akurasi_lr = r2_score(y, lr.predict(X)) * 100
# akurasi_rf = r2_score(y, rf.predict(X)) * 100

# html_template = """
# <h1>📱 Analisis Permintaan iPhone di Kota Tertentu</h1>
# <form method="POST">
#     Lokasi Toko: 
#     <select name="lokasi">
#         {% for l in lokasis %}
#             <option value="{{l}}">{{l}}</option>
#         {% endfor %}
#     </select>
#     <input type="submit" value="Analisis">
# </form>

# {% if table_html %}
# <hr>
# <h2>📊 iPhone Paling Diminati di {{ lokasi_terpilih }}</h2>
# {{ table_html | safe }}
# <br>
# <h2>📈 Grafik Popularitas</h2>
# <img src="data:image/png;base64,{{ img_base64 }}" style="max-width:800px;"><br>
# <h2>📌 Prediksi Harga & Akurasi</h2>
# Prediksi Harga (Random Forest) untuk tipe paling diminati: Rp{{ prediksi | round(0) }}<br>
# Akurasi Linear Regression: {{ akurasi_lr | round(2) }}%<br>
# Akurasi Random Forest: {{ akurasi_rf | round(2) }}%<br>
# {% endif %}
# """

# @app.route('/', methods=['GET', 'POST'])
# def home():
#     table_html = None
#     img_base64 = None
#     prediksi = None
#     lokasi_terpilih = None

#     if request.method == 'POST':
#         lokasi_input = request.form['lokasi']
#         lokasi_enc = le_lokasi.transform([lokasi_input])[0]
#         lokasi_terpilih = lokasi_input

#         # Filter dataset berdasarkan lokasi
#         subset = df[df['Lokasi Toko'] == lokasi_enc]

#         # Agregasi untuk tabel: Model, Warna, Kapasitas, Harga rata-rata, Jumlah terjual
#         grouped = subset.groupby(['Model','Warna','Kapasitas (GB)']).agg(
#             Harga_Rata2=('Harga (Rp)','mean'),
#             Jumlah_Terjual=('Jumlah Terjual','sum')
#         ).reset_index()

#         # Decode label kembali ke nama asli
#         grouped['Model'] = le_model.inverse_transform(grouped['Model'])
#         grouped['Warna'] = le_warna.inverse_transform(grouped['Warna'])

#         # Tabel HTML
#         table_html = grouped.sort_values('Jumlah_Terjual', ascending=False).to_html(index=False)

#         # Grafik bar: Model+Warna vs Jumlah Terjual
#         plt.figure(figsize=(10,5))
#         labels = grouped['Model'] + ' ' + grouped['Warna']
#         plt.bar(labels, grouped['Jumlah_Terjual'], color='skyblue')
#         plt.xticks(rotation=45, ha='right')
#         plt.ylabel('Jumlah Terjual')
#         plt.title(f'Popularitas iPhone di {lokasi_input}')
#         plt.tight_layout()

#         buf = io.BytesIO()
#         plt.savefig(buf, format='png')
#         buf.seek(0)
#         img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
#         plt.close()

#         # Prediksi harga Random Forest untuk tipe paling diminati (terjual terbanyak)
#         top = grouped.sort_values('Jumlah_Terjual', ascending=False).iloc[0]
#         model_enc = le_model.transform([top['Model']])[0]
#         warna_enc = le_warna.transform([top['Warna']])[0]
#         kapasitas = top['Kapasitas (GB)']
#         prediksi = rf.predict([[model_enc, warna_enc, kapasitas, lokasi_enc, 1]])[0]

#     return render_template_string(html_template,
#                                   lokasis=le_lokasi.classes_,
#                                   table_html=table_html,
#                                   img_base64=img_base64,
#                                   prediksi=prediksi,
#                                   akurasi_lr=akurasi_lr,
#                                   akurasi_rf=akurasi_rf,
#                                   lokasi_terpilih=lokasi_terpilih)

# if __name__ == '__main__':
#     app.run(debug=True)
# import matplotlib
# matplotlib.use('Agg') # SOLUSI UNTUK MENGHINDARI CRASH DI MACOS

# from flask import Flask, render_template, request
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import r2_score
# import matplotlib.pyplot as plt
# import io
# import base64
# import joblib

# app = Flask(__name__)

# # Load Excel
# df_full = pd.read_excel("data_penjualan_iphone_1000.xlsx")
# lokasis = df_full['Lokasi Toko'].unique()

# # --- LANGKAH DATA CLEANING: Hapus Duplikasi dan Reset Indeks ---
# # 1. Menghapus baris duplikat (jika ada)
# df_full.drop_duplicates(inplace=True)
# # 2. Reset indeks setelah penghapusan
# df_full.reset_index(drop=True, inplace=True)
# # ----------------------------------------------------

# # Encode kategori pada dataset penuh (digunakan untuk prediksi)
# le_model = LabelEncoder()
# le_warna = LabelEncoder()
# le_lokasi = LabelEncoder()
# df_full['Model_enc'] = le_model.fit_transform(df_full['Model'])
# df_full['Warna_enc'] = le_warna.fit_transform(df_full['Warna'])
# df_full['Lokasi_enc'] = le_lokasi.fit_transform(df_full['Lokasi Toko'])

# # Inisialisasi Akurasi Global (Tidak dipakai lagi di luar home())
# # akurasi_lr = akurasi_rf = None

# @app.route('/', methods=['GET','POST'])
# def home():
#     # Menginisialisasi variabel dengan nilai default aman
#     akurasi_lr = 0.00
#     akurasi_rf = 0.00
#     prediksi = 0
#     table_html = img_base64 = lokasi_terpilih = info_data_check = top_selling_info = produk_terpopuler_terpilih = None
    
#     # --- LOGIKA KETIKA PAGE DILUAR POST REQUEST (Initial Load / Global Analysis) ---
#     if request.method == 'GET':
#         # 1. Tentukan Model Terpopuler Global
#         global_popular = df_full.groupby(['Model', 'Kapasitas (GB)']).agg(
#             {'Jumlah Terjual': 'sum', 'Harga (Rp)': 'mean'}
#         ).reset_index().sort_values('Jumlah Terjual', ascending=False)
        
#         # Ambil 3 model terpopuler
#         top_3_models = global_popular.head(3)
        
#         top_selling_info = "Model Terlaris (Global):\n"
#         for index, row in top_3_models.iterrows():
#             # Menggunakan f-string untuk memformat harga Rupiah dengan pemisah ribuan
#             # Format: Rp12.345.678
#             harga_formatted = f"Rp{row['Harga (Rp)']:,.0f}".replace(",", "_").replace(".", ",").replace("_", ".") 
#             top_selling_info += f"- {row['Model']} ({row['Kapasitas (GB)']} GB): {row['Jumlah Terjual']} unit terjual (Rata-rata Harga: {harga_formatted})\n"

#         # 2. Grafik Popularitas Global
#         plt.figure(figsize=(8,5))
#         popular_global_plot = df_full.groupby('Model')['Jumlah Terjual'].sum()
#         popular_global_plot.plot(kind='bar', color='#1e90ff')
#         plt.ylabel('Jumlah Terjual (Global)')
#         plt.title('Popularitas Model iPhone (Keseluruhan Pasar)')
#         buf = io.BytesIO()
#         plt.tight_layout()
#         plt.savefig(buf, format='png')
#         buf.seek(0)
#         plt.close()
#         img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

#     # --- LOGIKA KETIKA ADA POST REQUEST (Location-Specific Analysis) ---
#     elif request.method == 'POST':
#         lokasi_terpilih = request.form['lokasi']
        
#         # 1. Filter data untuk lokasi yang dipilih
#         df_lokasi = df_full[df_full['Lokasi Toko']==lokasi_terpilih].copy()
        
#         # 2. Fitur dan target per lokasi
#         X = df_lokasi[['Model_enc','Warna_enc','Kapasitas (GB)']] 
#         y = df_lokasi['Harga (Rp)']

#         # Pengecekan, hanya lanjutkan jika data lokasi memadai (misal, > 5 baris)
#         if len(df_lokasi) > 5:
#             # 3. Split dan Latih Model Per Lokasi
#             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#             # Latih Linear Regression
#             lr = LinearRegression().fit(X_train, y_train)
#             pred_lr_test = lr.predict(X_test)
#             akurasi_lr = r2_score(y_test, pred_lr_test) * 100

#             # Latih Random Forest
#             rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
#             pred_rf_test = rf.predict(X_test)
#             akurasi_rf = r2_score(y_test, pred_rf_test) * 100

#             # 4. Pengecekan Variasi Harga (untuk info akurasi)
#             price_std_check = df_lokasi.groupby(['Model', 'Kapasitas (GB)'])['Harga (Rp)'].std().dropna()
#             if (price_std_check > 1000).any():
#                 info_data_check = "Model Prediksi (LR & RF) berhasil menemukan variasi harga. Akurasi tinggi karena variasi harga di kota ini kecil."
#             else:
#                 info_data_check = "Harga eceran sangat stabil di kota ini. Akurasi tinggi (>99%) adalah hasil dari hafalan harga dasar produk."
                
#             # 5. Tabel popularitas
#             table_html = df_lokasi.groupby(['Model','Warna','Kapasitas (GB)']).agg({
#                 'Jumlah Terjual':'sum',
#                 'Harga (Rp)':'mean'
#             }).reset_index().sort_values('Jumlah Terjual',ascending=False).to_html(classes='table', index=False)

#             # 6. Grafik popularitas (Lokasi)
#             plt.figure(figsize=(8,5))
#             popular = df_lokasi.groupby('Model')['Jumlah Terjual'].sum()
#             popular.plot(kind='bar', color='#fbbf24') # Warna emas untuk chart per lokasi
#             plt.ylabel('Jumlah Terjual')
#             plt.title(f'Popularitas Model iPhone di {lokasi_terpilih}')
#             buf = io.BytesIO()
#             plt.tight_layout()
#             plt.savefig(buf, format='png')
#             buf.seek(0)
#             plt.close() 
#             img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')


#             # 7. LOGIKA KRITIS: PREDIKSI BERDASARKAN MODEL TERLARIS KUMULATIF
#             popular_model = df_lokasi.groupby(['Model']).agg({'Jumlah Terjual':'sum'}).sort_values('Jumlah Terjual',ascending=False).reset_index().iloc[0]
#             model_terlaris_name = popular_model['Model']
            
#             df_model_terlaris = df_lokasi[df_lokasi['Model']==model_terlaris_name]
            
#             # Menemukan konfigurasi TERLARIS di DALAM MODEL TERLARIS tersebut
#             top_row = df_model_terlaris.groupby(['Model','Warna','Kapasitas (GB)']).sum(numeric_only=True).sort_values('Jumlah Terjual',ascending=False).reset_index().iloc[0]

#             produk_terpopuler_terpilih = f"Model Terlaris Kumulatif: {model_terlaris_name} (Menggunakan konfigurasi terlaris: {top_row['Warna']} {top_row['Kapasitas (GB)']} GB)"

#             X_pred = pd.DataFrame({
#                 'Model_enc':[le_model.transform([top_row['Model']])[0]],
#                 'Warna_enc':[le_warna.transform([top_row['Warna']])[0]],
#                 'Kapasitas (GB)':[top_row['Kapasitas (GB)']],
#             })
            
#             prediksi = round(rf.predict(X_pred)[0], 0) 
#         else:
#             info_data_check = "Data di lokasi ini tidak cukup untuk melatih model (kurang dari 5 baris)."
#             lokasi_terpilih = None # Kembali ke tampilan global

#     # Baris return yang benar:
#     return render_template('index.html', lokasis=lokasis, table_html=table_html,
#                            img_base64=img_base64, prediksi=prediksi, lokasi_terpilih=lokasi_terpilih,
#                            akurasi_lr=akurasi_lr, akurasi_rf=akurasi_rf, info_data_check=info_data_check,
#                            top_selling_info=top_selling_info, produk_terpopuler_terpilih=produk_terpopuler_terpilih)

# if __name__=='__main__':
#     app.run(debug=True)
import matplotlib
matplotlib.use('Agg')

from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans # Import K-Means
from sklearn.metrics import r2_score, silhouette_score # BARU: Import silhouette_score
import matplotlib.pyplot as plt
import io
import base64
import numpy as np # Import numpy untuk perhitungan

app = Flask(__name__)

# Load Excel
df_full = pd.read_excel("data_penjualan_iphone_1000.xlsx")
lokasis = df_full['Lokasi Toko'].unique()

# --- LANGKAH DATA CLEANING: Hapus Duplikasi dan Reset Indeks ---
df_full.drop_duplicates(inplace=True)
df_full.reset_index(drop=True, inplace=True)
# ----------------------------------------------------

# Encode kategori pada dataset penuh
le_model = LabelEncoder()
le_warna = LabelEncoder()
le_lokasi = LabelEncoder()
df_full['Model_enc'] = le_model.fit_transform(df_full['Model'])
df_full['Warna_enc'] = le_warna.fit_transform(df_full['Warna'])
df_full['Lokasi_enc'] = le_lokasi.fit_transform(df_full['Lokasi Toko'])

@app.route('/', methods=['GET','POST'])
def home():
    # Menginisialisasi variabel dengan nilai default aman
    akurasi_lr = 0.00
    prediksi = 0
    table_html = img_base64 = lokasi_terpilih = info_data_check = top_selling_info = produk_terpopuler_terpilih = None
    cluster_analysis = None
    silhouette_score_val = 0.0 # BARU: Inisialisasi Silhouette Score
    
    # --- LOGIKA KETIKA PAGE DILUAR POST REQUEST (Initial Load / Global Analysis) ---
    if request.method == 'GET':
        # 1. Tentukan Model Terpopuler Global
        global_popular = df_full.groupby(['Model', 'Kapasitas (GB)']).agg(
            {'Jumlah Terjual': 'sum', 'Harga (Rp)': 'mean'}
        ).reset_index().sort_values('Jumlah Terjual', ascending=False)
        
        top_3_models = global_popular.head(3)
        
        top_selling_info = "Model Terlaris (Global):\n"
        for index, row in top_3_models.iterrows():
            # Mengganti karakter pemisah ribuan agar formatting rupiah tetap aman
            harga_formatted = f"Rp{row['Harga (Rp)']:,.0f}".replace(",", "_").replace(".", ",").replace("_", ".") 
            top_selling_info += f"- {row['Model']} ({row['Kapasitas (GB)']} GB): {row['Jumlah Terjual']} unit terjual (Rata-rata Harga: {harga_formatted})\n"

        # 2. Grafik Popularitas Global
        plt.figure(figsize=(8,5))
        popular_global_plot = df_full.groupby('Model')['Jumlah Terjual'].sum()
        popular_global_plot.plot(kind='bar', color='#1e90ff')
        plt.ylabel('Jumlah Terjual (Global)')
        plt.title('Popularitas Model iPhone (Keseluruhan Pasar)')
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # --- LOGIKA KETIKA ADA POST REQUEST (Location-Specific Analysis) ---
    elif request.method == 'POST':
        lokasi_terpilih = request.form['lokasi']
        
        # 1. Filter data untuk lokasi yang dipilih
        df_lokasi = df_full[df_full['Lokasi Toko']==lokasi_terpilih].copy()
        
        # 2. Fitur dan target per lokasi
        X = df_lokasi[['Model_enc','Warna_enc','Kapasitas (GB)']] 
        y = df_lokasi['Harga (Rp)']

        if len(df_lokasi) > 5:
            # 3. Split dan Latih Model Regresi Linier
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Latih Linear Regression
            lr = LinearRegression().fit(X_train, y_train)
            pred_lr_test = lr.predict(X_test)
            akurasi_lr = r2_score(y_test, pred_lr_test) * 100

            # --------------------------------------------------------
            # 4. IMPLEMENTASI K-MEANS UNTUK SEGMENTASI PRODUK
            # Tentukan K=3 untuk 3 klaster (Misal: Low, Mid, High Price)
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X)
            df_lokasi['Klaster'] = kmeans.labels_

            # Menghitung Silhouette Score (memerlukan minimal 2 klaster dan 2 sampel)
            if len(np.unique(kmeans.labels_)) > 1:
                silhouette_score_val = silhouette_score(X, kmeans.labels_)
            else:
                silhouette_score_val = 0.0

            # Hitung rata-rata harga untuk setiap klaster
            cluster_summary = df_lokasi.groupby('Klaster').agg(
                Rata_rata_Harga=('Harga (Rp)', 'mean'),
                Jumlah_Produk=('Klaster', 'size')
            ).sort_values('Rata_rata_Harga').reset_index()

            # Tentukan label klaster berdasarkan harga rata-rata
            cluster_labels = ["Rendah (Low-Value)", "Menengah (Mid-Value)", "Tinggi (High-Value)"]
            
            cluster_analysis = "Analisis Segmentasi Harga (K-Means, K=3):\n"
            for i, row in cluster_summary.iterrows():
                harga_formatted = f"Rp{row['Rata_rata_Harga']:,.0f}".replace(",", "_").replace(".", ",").replace("_", ".") 
                cluster_analysis += f"- Klaster {i} ({cluster_labels[i]}): Rata-rata Harga {harga_formatted}, Total Produk: {row['Jumlah_Produk']} unit\n"
            # --------------------------------------------------------
                
            # 5. Pengecekan Variasi Harga (untuk info akurasi)
            price_std_check = df_lokasi.groupby(['Model', 'Kapasitas (GB)'])['Harga (Rp)'].std().dropna()
            if (price_std_check > 1000).any():
                info_data_check = "Model Prediksi (Linear Regression) berhasil menemukan variasi harga. Akurasi tinggi karena variasi harga di kota ini kecil."
            else:
                info_data_check = "Harga eceran sangat stabil di kota ini. Akurasi tinggi (>99%) adalah hasil dari hafalan harga dasar produk."
                
            # 6. Tabel popularitas
            table_html = df_lokasi.groupby(['Model','Warna','Kapasitas (GB)']).agg({
                'Jumlah Terjual':'sum',
                'Harga (Rp)':'mean'
            }).reset_index().sort_values('Jumlah Terjual',ascending=False).to_html(classes='table', index=False)

            # 7. LOGIKA KRITIS: PREDIKSI BERDASARKAN MODEL TERLARIS KUMULATIF
            popular_model = df_lokasi.groupby(['Model']).agg({'Jumlah Terjual':'sum'}).sort_values('Jumlah Terjual',ascending=False).reset_index().iloc[0]
            model_terlaris_name = popular_model['Model']
            
            df_model_terlaris = df_lokasi[df_lokasi['Model']==model_terlaris_name]
            
            top_row = df_model_terlaris.groupby(['Model','Warna','Kapasitas (GB)']).sum(numeric_only=True).sort_values('Jumlah Terjual',ascending=False).reset_index().iloc[0]

            produk_terpopuler_terpilih = f"Model Terlaris Kumulatif: {model_terlaris_name} (Menggunakan konfigurasi terlaris: {top_row['Warna']} {top_row['Kapasitas (GB)']} GB)"

            X_pred = pd.DataFrame({
                'Model_enc':[le_model.transform([top_row['Model']])[0]],
                'Warna_enc':[le_warna.transform([top_row['Warna']])[0]],
                'Kapasitas (GB)':[top_row['Kapasitas (GB)']],
            })
            
            # Prediksi menggunakan Linear Regression (lr)
            prediksi = round(lr.predict(X_pred)[0], 0) 
        else:
            info_data_check = "Data di lokasi ini tidak cukup untuk melatih model (kurang dari 5 baris)."
            lokasi_terpilih = None 

    return render_template('index.html', lokasis=lokasis, table_html=table_html,
                           img_base64=img_base64, prediksi=prediksi, lokasi_terpilih=lokasi_terpilih,
                           akurasi_lr=akurasi_lr, info_data_check=info_data_check,
                           top_selling_info=top_selling_info, produk_terpopuler_terpilih=produk_terpopuler_terpilih,
                           cluster_analysis=cluster_analysis,
                           silhouette_score_val=silhouette_score_val) # BARU: Mengirim Silhouette Score ke HTML

if __name__=='__main__':
    app.run(debug=True)
