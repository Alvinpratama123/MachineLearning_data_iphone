import matplotlib
matplotlib.use('Agg') # Solusi untuk lingkungan tanpa display

from flask import Flask, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans 
from sklearn.metrics import r2_score, silhouette_score 
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
import random
import warnings

# Mengabaikan peringatan K-Means
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

df_full = pd.DataFrame()
le_merk = LabelEncoder()
le_type = LabelEncoder()
MIN_DATA_FOR_MODEL = 5 # Minimum baris data untuk melatih model

def format_rupiah(price):
    """Mengubah float/numerik menjadi format Rupiah string yang aman."""
    try:
        if price is None or pd.isna(price) or not (isinstance(price, (int, float, np.number))):
            return "-"
        return f"Rp{price:,.0f}".replace(",", "_").replace(".", ",").replace("_", ".")
    except Exception:
        return "-"

# --- Load dan Persiapan Data ---
try:
    # Memastikan file diakses dengan aman
    df_raw = pd.read_csv("Data Penjualan Toko HP.csv")
    df_full = df_raw.copy()
    
    # Rename kolom dan membersihkan spasi di nama kolom
    df_full.rename(columns={'Harga': 'Harga (Rp)', 'type': 'Type Produk', 'nama pembeli': 'Nama Pembeli'}, inplace=True)
    
    if not df_full.empty:
        df_full.drop_duplicates(inplace=True)
        df_full['Harga (Rp)'] = pd.to_numeric(df_full['Harga (Rp)'], errors='coerce')
        df_full.dropna(subset=['Harga (Rp)', 'merk', 'Type Produk', 'Nama Pembeli'], inplace=True)
        df_full = df_full[df_full['Harga (Rp)'] > 1000] 
        df_full['merk'] = df_full['merk'].astype(str).str.strip()
        df_full['Type Produk'] = df_full['Type Produk'].astype(str).str.strip()
        df_full['Nama Pembeli'] = df_full['Nama Pembeli'].astype(str).str.strip().str.replace('"', '')
        
        # Penanganan Outlier
        if len(df_full) > 50:
            harga_mean = df_full['Harga (Rp)'].mean()
            harga_std = df_full['Harga (Rp)'].std()
            lower_bound = harga_mean - 3 * harga_std
            upper_bound = harga_mean + 3 * harga_std
            df_full = df_full[(df_full['Harga (Rp)'] >= lower_bound) & (df_full['Harga (Rp)'] <= upper_bound)]
        
        df_full.reset_index(drop=True, inplace=True)

        # Pembuatan Data Sintetik: Jumlah Terjual (Jika belum ada)
        if 'Jumlah Terjual' not in df_full.columns:
            np.random.seed(42)
            def generate_sales(price):
                if price < 2000000: return random.randint(3, 10)
                elif price < 4000000: return random.randint(2, 5)
                else: return random.randint(1, 3)
            df_full['Jumlah Terjual'] = df_full['Harga (Rp)'].apply(generate_sales)
        
        # Encoding Data 
        if len(df_full) > 0:
            le_merk.fit(df_full['merk'])
            le_type.fit(df_full['Type Produk'])
            df_full['Merk_enc'] = le_merk.transform(df_full['merk'])
            df_full['Type_enc'] = le_type.transform(df_full['Type Produk'])
        else:
             print("Warning: Data kosong setelah pembersihan.")
    
except Exception as e:
    print(f"FATAL ERROR saat memuat atau membersihkan data: {e}")
    df_full = pd.DataFrame()
# -------------------------------------------------------------------

def generate_plot(df, title, ylabel, color):
    """Fungsi pembantu untuk membuat plot bar 10 merk teratas."""
    if df.empty or 'merk' not in df.columns or 'Jumlah Terjual' not in df.columns:
        return None
        
    plt.figure(figsize=(8,5))
    plot_data = df.groupby('merk')['Jumlah Terjual'].sum().sort_values(ascending=False).head(10)
    
    plt.style.use('dark_background')
    
    if plot_data.empty: 
        plt.text(0.5, 0.5, "Data tidak cukup untuk grafik.", 
                 horizontalalignment='center', verticalalignment='center', 
                 transform=plt.gca().transAxes, color='white')
        plt.title(title, color='white')
        
    else:
        plot_data.plot(kind='bar', color=color)
        plt.ylabel(ylabel, color='white')
        plt.xlabel('Merk', color='white')
        plt.title(title, color='white')
        plt.xticks(rotation=45, ha='right', color='white')
        plt.yticks(color='white')
        
    plt.gca().set_facecolor('black')
    plt.gcf().set_facecolor('black')
    
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', transparent=True, bbox_inches='tight') 
    buf.seek(0)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def find_optimal_k_and_plot(X_scaled, max_k=10):
    """Menghitung optimal K menggunakan Elbow Method dan menghasilkan plot."""
    inertias = []
    
    # Pastikan data setidaknya memiliki 2 baris untuk klasterisasi
    max_k = min(max_k, len(X_scaled) - 1)
    if max_k < 2:
        return 3, None 
        
    K_range = range(1, max_k + 1)
    
    try:
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto', max_iter=300)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
    except Exception:
        return 3, None
        
    optimal_k = 3
    if len(K_range) > 3:
        diff = np.diff(inertias)
        diff_2 = np.diff(diff)
        if len(diff_2) > 0:
            idx_elbow = np.argmin(diff_2) 
            optimal_k = idx_elbow + 3 
            optimal_k = max(2, min(optimal_k, max_k))
        else:
            optimal_k = max(2, min(3, max_k))

    plt.figure(figsize=(6, 4))
    plt.style.use('dark_background') 
    plt.plot(K_range, inertias, marker='o', color='#fdba74') 
    plt.title('Metode Elbow untuk Optimal K', color='white')
    plt.xlabel('Jumlah Klaster (K)', color='white')
    plt.ylabel('Inersia', color='white')
    plt.xticks(color='white')
    plt.yticks(color='white')
    
    if optimal_k > 1 and optimal_k <= max_k:
        plt.axvline(x=optimal_k, color='#ef4444', linestyle='--', label=f'Optimal K = {optimal_k}')
        plt.legend(facecolor='black', edgecolor='gray', labelcolor='white')
        
    plt.gca().set_facecolor('black')
    plt.gcf().set_facecolor('black')
        
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', transparent=True)
    buf.seek(0)
    plt.close()
    elbow_img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    return optimal_k, elbow_img_base64


def run_global_analysis(df_input):
    """Menjalankan regresi dan klasterisasi pada SELURUH data global."""
    
    # Inisialisasi hasil default
    results = {
        'akurasi_lr': 0.00,
        'prediksi': df_full['Harga (Rp)'].mean() if not df_full.empty else 0,
        'produk_terpopuler_terpilih': "N/A - N/A",
        'cluster_analysis': "Data tidak cukup untuk analisis klasterisasi dan prediksi.",
        'silhouette_score_val': 0.0,
        'elbow_img_base64': None,
        'info_data_check': f"Total data stabil: {len(df_input)} baris."
    }

    if len(df_input) < MIN_DATA_FOR_MODEL:
        results['info_data_check'] = "Data terlalu sedikit. Analisis Model diabaikan."
        return results, pd.DataFrame()
    
    df_analysis = df_input.copy()

    # 1. Linear Regression (Prediksi Harga) - GLOBAL
    X = df_analysis[['Merk_enc','Type_enc']] 
    y = df_analysis['Harga (Rp)']

    # Menggunakan Test Split minimal untuk mendapatkan Akurasi Global
    test_size = max(0.2, 5 / len(X)) if len(X) >= 5 else 0.0 # memastikan setidaknya 5 data tes
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    lr = LinearRegression().fit(X_train, y_train)
    
    if test_size > 0.0: 
        pred_lr_test = lr.predict(X_test)
        akurasi_lr = r2_score(y_test, pred_lr_test) * 100
        results['akurasi_lr'] = max(0.0, akurasi_lr) 
        
    # 2. K-MEANS UNTUK SEGMENTASI PRODUK GLOBAL
    X_cluster = df_analysis[['Harga (Rp)', 'Merk_enc']].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    optimal_k, elbow_img_base64 = find_optimal_k_and_plot(X_scaled)
    results['elbow_img_base64'] = elbow_img_base64
    
    n_clusters = max(2, min(optimal_k, len(df_analysis) - 1))
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto').fit(X_scaled)
    df_analysis['Klaster'] = kmeans.labels_
    
    if len(np.unique(kmeans.labels_)) > 1:
        results['silhouette_score_val'] = silhouette_score(X_scaled, kmeans.labels_)

    cluster_summary = df_analysis.groupby('Klaster').agg(
        Rata_rata_Harga=('Harga (Rp)', 'mean'),
        Jumlah_Produk=('Klaster', 'size')
    ).sort_values('Rata_rata_Harga').reset_index()

    default_labels = ["Low-Value", "Mid-Value", "High-Value", "Premium"]
    cluster_text = f"Analisis Segmentasi Harga Global (K-Means, K={n_clusters}):\n"
    for i, row in cluster_summary.iterrows():
        label = default_labels[i] if i < len(default_labels) else f"Klaster {i+1}"
        cluster_text += f"- {label}: Rata-rata Harga {format_rupiah(row['Rata_rata_Harga'])}, Total Produk: {row['Jumlah_Produk']} unit\n"
    results['cluster_analysis'] = cluster_text
    
    # 3. Prediksi Produk Paling Diminati (GLOBAL)
    if not df_analysis.empty:
        # Produk yang paling banyak terjual
        top_row_group = df_analysis.groupby(['merk','Type Produk']).agg({
            'Jumlah Terjual':'sum', 
            'Harga (Rp)':'mean'
        }).reset_index().sort_values('Jumlah Terjual',ascending=False).head(1)
        
        if not top_row_group.empty:
            merk_terlaris = top_row_group['merk'].iloc[0]
            type_terlaris = top_row_group['Type Produk'].iloc[0]
            avg_price = top_row_group['Harga (Rp)'].iloc[0]

            results['produk_terpopuler_terpilih'] = f"{merk_terlaris} - {type_terlaris}"

            try:
                # Prediksi harga untuk produk terlaris menggunakan Model Regresi Global
                if merk_terlaris in le_merk.classes_ and type_terlaris in le_type.classes_:
                    X_pred = pd.DataFrame({
                        'Merk_enc':[le_merk.transform([merk_terlaris])[0]],
                        'Type_enc':[le_type.transform([type_terlaris])[0]],
                    })
                    results['prediksi'] = round(lr.predict(X_pred)[0], 0) 
                else:
                    results['prediksi'] = avg_price
                    results['info_data_check'] += " (Gagal prediksi, menggunakan rata-rata harga produk terlaris.)"
                    
            except Exception:
                results['prediksi'] = avg_price
                results['info_data_check'] += " (Gagal prediksi, menggunakan rata-rata harga produk terlaris.)"

    # Hapus kolom Merk_enc dan Type_enc sebelum dikirim ke HTML
    df_analysis.drop(columns=['Merk_enc', 'Type_enc', 'Jumlah Terjual'], errors='ignore', inplace=True)
    df_analysis['Harga (Rp)'] = df_analysis['Harga (Rp)'].apply(format_rupiah)
    
    return results, df_analysis[['Nama Pembeli', 'merk', 'Type Produk', 'Harga (Rp)', 'Klaster']]


@app.route('/', methods=['GET'])
def home():
    
    # Cek Kondisi Data Awal
    if df_full.empty or len(df_full) < MIN_DATA_FOR_MODEL:
        return render_template('index.html', info_data_check="Error: Data tidak valid atau terlalu sedikit data yang tersisa setelah pembersihan.")

    # --- ANALISIS GLOBAL OTOMATIS ---
    analysis_results, df_table = run_global_analysis(df_full)
    
    # 4. Tabel Data HP dan Nama Pengguna
    if not df_table.empty:
        # Pastikan kolom Klaster ada sebelum encoding jika data terlalu sedikit
        if 'Klaster' not in df_table.columns:
             df_table['Klaster'] = 'N/A'
             
        # Ganti nama klaster menjadi label yang lebih bermakna di tabel
        df_table['Klaster'] = df_table['Klaster'].apply(lambda x: 
            {0: 'Low-Value', 1: 'Mid-Value', 2: 'High-Value', 3: 'Premium'}.get(x, f'Klaster {x+1}') if isinstance(x, (int, np.integer)) else x
        )
             
        df_table.rename(columns={'Klaster': 'Klaster Harga', 'merk': 'Merk', 'Type Produk': 'Tipe Produk'}, inplace=True)
        table_html = df_table.to_html(classes='table', index=False)
    else:
        table_html = None

    # 5. Grafik Popularitas Merk Global
    img_base64 = generate_plot(df_full, 'Top 10 Popularitas Merk HP (Global Market)', 'Total Unit Terjual (Estimasi)', '#10b981')
    
    # Gabungkan semua hasil untuk dikirim ke template
    return render_template('index.html', 
                           table_html=table_html, 
                           img_base64=img_base64,
                           global_analysis=True, # Flag untuk template
                           **analysis_results # Kirim semua metrik analisis
                           )

if __name__=='__main__':
    if not df_full.empty and len(df_full) >= MIN_DATA_FOR_MODEL:
        plt.style.use('dark_background')
        app.run(debug=False)
    else:
        print("Aplikasi tidak dapat dimulai karena data tidak valid atau terlalu sedikit data yang tersisa setelah pembersihan.")