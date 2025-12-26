from flask import Flask, render_template
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io, base64
import warnings

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# Nonaktifkan warning agar log bersih
warnings.filterwarnings("ignore")
plt.switch_backend('Agg')

app = Flask(__name__)

@app.route("/")
def index():
    try:
        # ===== 1. LOAD DATA =====
        df = pd.read_csv("Mobile-Phones.csv")
        df.columns = df.columns.str.lower()
        # Mengambil kolom yang diperlukan saja
        df = df[['brand', 'model']].dropna()

        # ===== 2. LOGIKA POPULARITAS =====
        # Menentukan popularitas berdasarkan total kemunculan brand di seluruh dataset
        brand_counts = df['brand'].value_counts()
        
        def determine_popularity(brand):
            count = brand_counts[brand]
            if count > brand_counts.quantile(0.7): return 'High'
            elif count > brand_counts.quantile(0.3): return 'Medium'
            else: return 'Low'

        df['popularitas'] = df['brand'].apply(determine_popularity)

        # ===== 3. ENCODING UNTUK MODEL ML =====
        le_brand = LabelEncoder()
        le_model = LabelEncoder()
        le_target = LabelEncoder()

        df['brand_enc'] = le_brand.fit_transform(df['brand'])
        df['model_enc'] = le_model.fit_transform(df['model'])
        df['target'] = le_target.fit_transform(df['popularitas'])

        X = df[['brand_enc', 'model_enc']]
        y = df['target']

        # ===== 4. TRAINING MODEL =====
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        nb = GaussianNB()

        rf.fit(X_train, y_train)
        nb.fit(X_train, y_train)

        # ===== 5. EVALUASI MODEL =====
        rf_pred = rf.predict(X_test)
        nb_pred = nb.predict(X_test)

        def get_full_metrics(y_true, y_pred):
            return {
                "Accuracy": f"{accuracy_score(y_true, y_pred)*100:.1f}%",
                "Precision": round(precision_score(y_true, y_pred, average='macro', zero_division=0), 2),
                "Recall": round(recall_score(y_true, y_pred, average='macro', zero_division=0), 2),
                "F1": round(f1_score(y_true, y_pred, average='macro', zero_division=0), 2),
            }

        # Statistik untuk Ringkasan
        results = {
            "rf": get_full_metrics(y_test, rf_pred),
            "nb": get_full_metrics(y_test, nb_pred),
            "total_rows": len(df),
            "total_brands": len(brand_counts),
            "top_brand": brand_counts.index[0],
            "top_count": int(brand_counts.values[0])
        }

        # ===== 6. GRAFIK SEMUA BRAND =====
        plt.figure(figsize=(16, 6))
        plt.style.use('dark_background')
        colors = plt.cm.plasma(np.linspace(0, 1, len(brand_counts)))
        brand_counts.plot(kind='bar', color=colors, edgecolor='white')
        plt.title("Visualisasi Distribusi Semua Brand (Highest to Lowest)", fontsize=14, pad=20)
        plt.ylabel("Jumlah Model")
        plt.xticks(rotation=90)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, facecolor='#1e293b')
        plt.close()
        graph = base64.b64encode(buf.getvalue()).decode()

        # ===== 7. TABEL SEMUA BRAND (DIPERBAIKI) =====
        # Kita buat tabel ringkasan per brand agar brand seperti Nokia, Infinix, dll muncul
        brand_summary = df.groupby('brand').agg({
            'model': 'count',
            'popularitas': 'first'
        }).reset_index().sort_values('model', ascending=False)
        
        # Mengganti nama kolom untuk tampilan tabel
        brand_summary.columns = ['Brand', 'Total Model', 'Kategori Popularitas']
        
        # Konversi ke HTML dengan class styling
        table_html = brand_summary.to_html(
            index=False, 
            classes="min-w-full text-sm text-left border-collapse",
            border=0
        )

        return render_template(
            "index.html",
            results=results,
            graph=graph,
            table=table_html
        )
    except Exception as e:
        return f"Error Aplikasi: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, port=5001)