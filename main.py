# ============================================================
# TUGAS BESAR MACHINE LEARNING - POLITEKNIK NEGERI INDRAMAYU
# Kelompok 1 | Semester Genap 2025/2026
# Deteksi Fraud Pengajuan Reimbursement Perjalanan Dinas
# Algoritma: Decision Tree + Risk Scoring
# VERSI FINAL 
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import json
import os

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay,
                             f1_score, precision_score, recall_score)
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')

os.makedirs('output', exist_ok=True)

print("=" * 60)
print("  SISTEM DETEKSI FRAUD REIMBURSEMENT PERJALANAN DINAS")
print("  Politeknik Negeri Indramayu | Kelompok 1")
print("=" * 60)
print("\nSemua library berhasil diimport!")


# ============================================================
# TAHAP 1: LOAD DATASET
# ============================================================
print("\n[TAHAP 1] LOAD DATASET")
print("-" * 40)

df_raw = pd.read_csv('data/insurance_data.csv')
print(f"Dataset berhasil dimuat : {df_raw.shape[0]} baris, {df_raw.shape[1]} kolom")


# ============================================================
# TAHAP 2: FEATURE ENGINEERING (Transformasi ke Konteks Laporan)
# ============================================================
print("\n[TAHAP 2] FEATURE ENGINEERING")
print("-" * 40)
print("Mentransformasi fitur dataset asli ke konteks reimbursement perjalanan dinas...")

df = pd.DataFrame()

# a. Jenis Perjalanan 
travel_map = {
    'Travel'  : 'luar negeri',
    'Health'  : 'dalam kota',
    'Motor'   : 'luar kota',
    'Life'    : 'luar kota',
    'Property': 'dalam kota',
    'Mobile'  : 'dalam kota'
}
df['jenis_perjalanan'] = df_raw['INSURANCE_TYPE'].map(travel_map)
print(f"  [OK] jenis_perjalanan  <- INSURANCE_TYPE")

# b. Nominal Klaim (dari CLAIM_AMOUNT)
df['nominal_klaim'] = df_raw['CLAIM_AMOUNT']
print(f"  [OK] nominal_klaim     <- CLAIM_AMOUNT")

# c. Nominal Standar = rata-rata klaim per jenis perjalanan (benchmark historis)
df['nominal_standar'] = df.groupby('jenis_perjalanan')['nominal_klaim'].transform('mean')
print(f"  [OK] nominal_standar   <- groupby mean per jenis_perjalanan")

# d. Rasio Klaim = nominal_klaim / nominal_standar
df['rasio_klaim'] = df['nominal_klaim'] / df['nominal_standar']
print(f"  [OK] rasio_klaim       <- nominal_klaim / nominal_standar")

# e. Kelengkapan Dokumen (dari POLICE_REPORT_AVAILABLE, 1 = lengkap, 0.75 = kurang lengkap)
df['kelengkapan_dokumen'] = df_raw['POLICE_REPORT_AVAILABLE'].apply(
    lambda x: 1.0 if x == 1 else 0.75
)
print(f"  [OK] kelengkapan_dokumen <- POLICE_REPORT_AVAILABLE")

# f. Frekuensi Klaim Bulan (dari TXN_DATE_TIME + CUSTOMER_ID)
df_raw['_month'] = pd.to_datetime(df_raw['TXN_DATE_TIME'], errors='coerce').dt.month
df['frekuensi_klaim_bulan'] = df_raw.groupby(
    ['CUSTOMER_ID', '_month']
)['TRANSACTION_ID'].transform('count')
df['frekuensi_klaim_bulan'] = df['frekuensi_klaim_bulan'].fillna(1).astype(int)
print(f"  [OK] frekuensi_klaim_bulan <- groupby CUSTOMER_ID + bulan")

# g. Selisih Hari Pengajuan (dari REPORT_DT - LOSS_DT)
df_raw['_report_dt'] = pd.to_datetime(df_raw['REPORT_DT'], errors='coerce')
df_raw['_loss_dt']   = pd.to_datetime(df_raw['LOSS_DT'],   errors='coerce')
df['selisih_hari_pengajuan'] = (df_raw['_report_dt'] - df_raw['_loss_dt']).dt.days
df['selisih_hari_pengajuan'] = df['selisih_hari_pengajuan'].fillna(
    df['selisih_hari_pengajuan'].median()
).clip(lower=0)
print(f"  [OK] selisih_hari_pengajuan <- REPORT_DT - LOSS_DT")

# h. Jabatan Pegawai (dari RISK_SEGMENTATION)
jabatan_map = {'L': 'Staff', 'M': 'Supervisor', 'H': 'Manager'}
df['jabatan_pegawai'] = df_raw['RISK_SEGMENTATION'].map(jabatan_map).fillna('Staff')
print(f"  [OK] jabatan_pegawai   <- RISK_SEGMENTATION")

# i. Riwayat Fraud (dari ANY_INJURY sebagai proxy)
df['riwayat_fraud'] = df_raw['ANY_INJURY'].astype(int)
print(f"  [OK] riwayat_fraud     <- ANY_INJURY (proxy)")

# j. TARGET: fraud_label (Kombinasi Historis & Aturan Bisnis)
# Fraud jika CLAIM_STATUS == 'D' ATAU rasio_klaim > 1.8
df['fraud_label'] = np.where(
    (df_raw['CLAIM_STATUS'] == 'D') | (df['rasio_klaim'] > 1.8), 1, 0
)
print(f"  [OK] fraud_label       <- CLAIM_STATUS=='D' OR rasio_klaim > 1.8")

print(f"\nDistribusi target awal:")
print(f"  Normal (0): {(df['fraud_label']==0).sum()} data")
print(f"  Fraud  (1): {(df['fraud_label']==1).sum()} data")
print(f"  Rasio fraud: {(df['fraud_label']==1).mean()*100:.2f}%")


# ============================================================
# TAHAP 3: SELEKSI FITUR
# ============================================================
print("\n[TAHAP 3] SELEKSI FITUR")
print("-" * 40)

# PENTING: rasio_klaim dihapus dari fitur agar tidak data leakage 100%,
# model dipaksa menebak pola dari nominal_klaim dan nominal_standar
FITUR = [
    'jenis_perjalanan',       # Kategorikal
    'nominal_klaim',          # Numerik
    'nominal_standar',        # Numerik
    'kelengkapan_dokumen',    # Numerik (0–1)
    'frekuensi_klaim_bulan',  # Numerik
    'selisih_hari_pengajuan', # Numerik
    'jabatan_pegawai',        # Kategorikal
    'riwayat_fraud',          # Biner
]
TARGET = 'fraud_label'

print(f"Fitur yang digunakan ({len(FITUR)} fitur):")
for i, f in enumerate(FITUR, 1):
    print(f"  {i:2d}. {f}")
print(f"\nTarget: {TARGET} (0 = Tidak Fraud, 1 = Fraud)")


# ============================================================
# TAHAP 4: PREPROCESSING (Pipeline Anti Data Leakage)
# ============================================================
print("\n[TAHAP 4] PREPROCESSING")
print("-" * 40)

# Step 4.1: Handle missing value
df.fillna(df.median(numeric_only=True), inplace=True)

# Step 4.2: Encode fitur kategorikal
KOLOM_KATEGORIKAL = ['jenis_perjalanan', 'jabatan_pegawai']
le_encoders = {}
for col in KOLOM_KATEGORIKAL:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_encoders[col] = le

# Step 4.3: Pisah X dan y
X = df[FITUR].copy()
y = df[TARGET].copy()

# Step 4.4: Train/Test Split SEBELUM scaling (anti data leakage)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Step 4.3 - Train/Test Split (80:20, stratified):")
print(f"  Train: {X_train.shape[0]} data | Fraud: {y_train.sum()} ({y_train.mean()*100:.1f}%)")
print(f"  Test : {X_test.shape[0]} data | Fraud: {y_test.sum()} ({y_test.mean()*100:.1f}%)")

# Step 4.5: Normalisasi — fit HANYA di train
KOLOM_NUMERIK = [
    'nominal_klaim', 'nominal_standar',
    'frekuensi_klaim_bulan', 'selisih_hari_pengajuan'
]
scaler = MinMaxScaler()
X_train = X_train.copy()
X_test  = X_test.copy()
X_train[KOLOM_NUMERIK] = scaler.fit_transform(X_train[KOLOM_NUMERIK])
X_test[KOLOM_NUMERIK]  = scaler.transform(X_test[KOLOM_NUMERIK])
print(f"Step 4.4 - MinMaxScaler: fit pada train, transform pada keduanya")

# Step 4.6: SMOTE — hanya pada train set
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"Step 4.5 - SMOTE (hanya pada train set):")
print(f"  Sebelum: {len(X_train)} data | Fraud: {y_train.sum()}")
print(f"  Setelah: {len(X_train_sm)} data | Fraud: {y_train_sm.sum()}")


# ============================================================
# TAHAP 5: EKSPLORASI DATA (EDA) + VISUALISASI
# ============================================================
print("\n[TAHAP 5] EKSPLORASI DATA (EDA)")
print("-" * 40)

# --- Viz 1: Distribusi Label Fraud ---
plt.figure(figsize=(7, 5))
ax = sns.countplot(x=TARGET, data=df, palette='Set2')
ax.set_xticklabels(['Normal (0)', 'Fraud (1)'])
ax.set_title('Distribusi Label: Normal vs Fraud', fontsize=14, fontweight='bold')
ax.set_xlabel('Kategori Klaim')
ax.set_ylabel('Jumlah Data')
for p in ax.patches:
    pct = p.get_height() / len(df) * 100
    ax.annotate(f'{int(p.get_height())}\n({pct:.1f}%)',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('output/viz1_distribusi_label.png', dpi=150)
plt.close()
print("  [1] viz1_distribusi_label.png -> disimpan")

# --- Viz 2: Histogram Nominal Klaim ---
plt.figure(figsize=(9, 5))
df_plot = df[['nominal_klaim', 'fraud_label']].copy()
df_plot['Label'] = df_plot['fraud_label'].map({0: 'Normal', 1: 'Fraud'})
sns.histplot(data=df_plot, x='nominal_klaim', hue='Label',
             bins=40, palette=['#2196F3', '#F44336'], alpha=0.7)
plt.title('Histogram Nominal Klaim: Normal vs Fraud', fontsize=14, fontweight='bold')
plt.xlabel('Nominal Klaim')
plt.ylabel('Frekuensi')
plt.tight_layout()
plt.savefig('output/viz2_histogram_nominal_klaim.png', dpi=150)
plt.close()
print("  [2] viz2_histogram_nominal_klaim.png -> disimpan")

# --- Viz 3: Heatmap Korelasi ---
plt.figure(figsize=(12, 9))
corr_df = X_train[FITUR].copy()
corr_df[TARGET] = y_train.values
corr_matrix = corr_df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
            cmap='coolwarm', center=0,
            linewidths=0.5, annot_kws={'size': 7})
plt.title('Heatmap Korelasi Antar Fitur (Train Set)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('output/viz3_heatmap_korelasi.png', dpi=150)
plt.close()
print("  [3] viz3_heatmap_korelasi.png -> disimpan")

# --- Viz 4: Boxplot Rasio Klaim per Label (Menggunakan df agar tetap bisa plot rasio_klaim) ---
plt.figure(figsize=(8, 5))
df_box = df[['rasio_klaim', 'fraud_label']].copy()
df_box['Label'] = df_box['fraud_label'].map({0: 'Normal', 1: 'Fraud'})
sns.boxplot(x='Label', y='rasio_klaim', data=df_box,
            palette=['#2196F3', '#F44336'])
plt.title('Boxplot Rasio Klaim per Label', fontsize=14, fontweight='bold')
plt.xlabel('Kategori Klaim')
plt.ylabel('Rasio Klaim (nominal_klaim / nominal_standar)')
plt.ylim(0, 5)
plt.tight_layout()
plt.savefig('output/viz4_boxplot_rasio_klaim.png', dpi=150)
plt.close()
print("  [4] viz4_boxplot_rasio_klaim.png -> disimpan")


# ============================================================
# TAHAP 6: TRAINING MODEL (Decision Tree)
# ============================================================
print("\n[TAHAP 6] TRAINING MODEL - DECISION TREE")
print("-" * 40)

model = DecisionTreeClassifier(
    max_depth=5,              # Cukup dalam untuk menemukan pola
    criterion='gini',
    class_weight='balanced',  # Menangani ketidakseimbangan kelas
    random_state=42
)

model.fit(X_train_sm, y_train_sm)
print(f"\nModel berhasil dilatih!")
print(f"  Kedalaman pohon aktual : {model.get_depth()}")
print(f"  Jumlah leaf nodes      : {model.get_n_leaves()}")

# Cek overfitting
train_pred = model.predict(X_train_sm)
test_pred  = model.predict(X_test)
train_acc  = accuracy_score(y_train_sm, train_pred)
test_acc   = accuracy_score(y_test, test_pred)
gap        = train_acc - test_acc

print(f"\n[CEK OVERFITTING]")
print(f"  Train Accuracy : {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"  Test  Accuracy : {test_acc:.4f}  ({test_acc*100:.2f}%)")
print(f"  Gap            : {gap:.4f}")


# ============================================================
# TAHAP 7: PREDIKSI + RISK SCORING
# ============================================================
print("\n[TAHAP 7] PREDIKSI & RISK SCORING")
print("-" * 40)

y_pred      = model.predict(X_test)
risk_scores = model.predict_proba(X_test)[:, 1]

def kategorikan_risiko(score):
    if score <= 0.30:
        return 'Low Risk'
    elif score <= 0.60:
        return 'Medium Risk'
    else:
        return 'High Risk'

def rekomendasi_sistem(score):
    if score <= 0.30:
        return 'Auto Approved'
    elif score <= 0.60:
        return 'Manual Review'
    else:
        return 'Auto Rejected'

hasil = pd.DataFrame({
    'Label Aktual'      : y_test.values,
    'Label Prediksi'    : y_pred,
    'Risk Score'        : np.round(risk_scores, 4),
    'Kategori Risiko'   : [kategorikan_risiko(s) for s in risk_scores],
    'Rekomendasi Sistem': [rekomendasi_sistem(s) for s in risk_scores]
})

print("\nDistribusi Kategori Risiko:")
dist_risiko = hasil['Kategori Risiko'].value_counts()
for kategori, jumlah in dist_risiko.items():
    pct = jumlah / len(hasil) * 100
    print(f"  {kategori:<15}: {jumlah:4d} data ({pct:.1f}%)")

hasil.to_csv('output/hasil_prediksi.csv', index=False)
print("\nHasil prediksi disimpan: output/hasil_prediksi.csv")


# ============================================================
# TAHAP 8: EVALUASI MODEL
# ============================================================
print("\n[TAHAP 8] EVALUASI MODEL")
print("-" * 40)

acc       = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall    = recall_score(y_test, y_pred, zero_division=0)
f1        = f1_score(y_test, y_pred, zero_division=0)

# Cek target laporan
print(f"\n[CEK TARGET LAPORAN]")
print(f"  Accuracy  >= 80% : {'PASS' if acc >= 0.80 else 'BELUM'} ({acc*100:.1f}%)")
print(f"  Precision >= 75% : {'PASS' if precision >= 0.75 else 'BELUM'} ({precision*100:.1f}%)")
print(f"  Recall    >= 70% : {'PASS' if recall >= 0.70 else 'BELUM'} ({recall*100:.1f}%)")
print(f"  F1-Score  >= 72% : {'PASS' if f1 >= 0.72 else 'BELUM'} ({f1*100:.1f}%)")

print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=['Normal', 'Fraud'], zero_division=0))

# Simpan metrics ke JSON
metrics = {
    "accuracy"  : round(float(acc), 4),
    "precision" : round(float(precision), 4),
    "recall"    : round(float(recall), 4),
    "f1_score"  : round(float(f1), 4),
    "train_accuracy" : round(float(train_acc), 4),
    "test_accuracy"  : round(float(test_acc), 4),
    "overfit_gap"    : round(float(gap), 4),
    "total_data"     : int(len(df)),
    "total_fraud"    : int((df['fraud_label']==1).sum()),
    "total_normal"   : int((df['fraud_label']==0).sum()),
    "distribusi_risiko": {k: int(v) for k, v in dist_risiko.items()}
}
with open('output/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# --- Viz 5: Confusion Matrix ---
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['Normal', 'Fraud'])
disp.plot(cmap='Blues', colorbar=False)
plt.title('Confusion Matrix - Decision Tree', fontsize=13, fontweight='bold')
tn, fp, fn, tp = cm.ravel()
plt.figtext(0.5, -0.05,
    f'TN={tn} | FP={fp} | FN={fn} | TP={tp}\n'
    f'FPR: {fp/(fp+tn)*100:.1f}% | FNR: {fn/(fn+tp)*100:.1f}%',
    ha='center', fontsize=9, style='italic')
plt.tight_layout()
plt.savefig('output/viz5_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [5] viz5_confusion_matrix.png -> disimpan")

# --- Viz 6: Feature Importance ---
feat_imp = pd.Series(model.feature_importances_, index=FITUR).sort_values(ascending=True)
feat_imp_nonzero = feat_imp[feat_imp > 0]

plt.figure(figsize=(9, max(5, len(feat_imp_nonzero) * 0.55)))
colors = ['#F44336' if v == feat_imp_nonzero.max() else
          '#FF9800' if v >= feat_imp_nonzero.quantile(0.75) else
          '#2196F3' for v in feat_imp_nonzero]
feat_imp_nonzero.plot(kind='barh', color=colors)
plt.title('Feature Importance - Decision Tree', fontsize=12, fontweight='bold')
plt.xlabel('Importance Score (Gini)')
red_patch    = mpatches.Patch(color='#F44336', label='Fitur Terpenting')
orange_patch = mpatches.Patch(color='#FF9800', label='Fitur Penting (Top 25%)')
blue_patch   = mpatches.Patch(color='#2196F3', label='Fitur Lainnya')
plt.legend(handles=[red_patch, orange_patch, blue_patch], loc='lower right', fontsize=8)
plt.tight_layout()
plt.savefig('output/viz6_feature_importance.png', dpi=150)
plt.close()
print("  [6] viz6_feature_importance.png -> disimpan")

# --- Viz 7: Decision Tree (max 3 level) ---
plt.figure(figsize=(24, 10))
plot_tree(model, feature_names=FITUR, class_names=['Normal', 'Fraud'],
          filled=True, rounded=True, fontsize=8, max_depth=3)
plt.title('Visualisasi Decision Tree (3 Level Pertama)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('output/viz7_decision_tree.png', dpi=150)
plt.close()
print("  [7] viz7_decision_tree.png -> disimpan")

# --- Viz 8: Distribusi Risk Score ---
plt.figure(figsize=(9, 5))
bins = np.linspace(0, 1, 30)
plt.hist(risk_scores[y_test == 0], bins=bins, alpha=0.6, color='#2196F3', label='Normal (Aktual)', density=True)
plt.hist(risk_scores[y_test == 1], bins=bins, alpha=0.6, color='#F44336', label='Fraud (Aktual)', density=True)
plt.axvline(0.30, color='orange', linestyle='--', linewidth=1.5, label='Threshold Low/Medium (0.30)')
plt.axvline(0.60, color='red',    linestyle='--', linewidth=1.5, label='Threshold Medium/High (0.60)')
plt.title('Distribusi Risk Score per Kelas Aktual', fontsize=13, fontweight='bold')
plt.xlabel('Risk Score (Probabilitas Fraud)')
plt.ylabel('Density')
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig('output/viz8_distribusi_risk_score.png', dpi=150)
plt.close()
print("  [8] viz8_distribusi_risk_score.png -> disimpan")

rules = export_text(model, feature_names=FITUR, max_depth=3)
with open('output/aturan_pohon_keputusan.txt', 'w') as f:
    f.write("ATURAN POHON KEPUTUSAN (3 Level Pertama)\n")
    f.write("=" * 60 + "\n\n")
    f.write(rules)


# ============================================================
# RINGKASAN AKHIR
# ============================================================
print("\n" + "=" * 60)
print("  RINGKASAN HASIL AKHIR")
print("=" * 60)
print(f"\n  Dataset     : {len(df)} data total")
print(f"  Fraud       : {(df['fraud_label']==1).sum()} data ({(df['fraud_label']==1).mean()*100:.2f}%)")
print(f"  Fitur       : {len(FITUR)} fitur (sesuai laporan Bab 2.3)")
print(f"  Algoritma   : Decision Tree (Gini, max_depth=5)")
print(f"  Imbalance   : SMOTE (train set only)")
print(f"\n  HASIL EVALUASI:")
print(f"  Accuracy    : {acc*100:.2f}%  {'[PASS >= 80%]' if acc >= 0.80 else '[BELUM]'}")
print(f"  Precision   : {precision*100:.2f}%  {'[PASS >= 75%]' if precision >= 0.75 else '[BELUM]'}")
print(f"  Recall      : {recall*100:.2f}%  {'[PASS >= 70%]' if recall >= 0.70 else '[BELUM]'}")
print(f"  F1-Score    : {f1*100:.2f}%  {'[PASS >= 72%]' if f1 >= 0.72 else '[BELUM]'}")
print(f"\n  OVERFITTING CHECK:")
print(f"  Train Acc   : {train_acc*100:.2f}%")
print(f"  Test  Acc   : {test_acc*100:.2f}%")
print(f"  Gap         : {gap*100:.2f}% {'[OK]' if gap < 0.05 else '[Perlu perhatian]'}")
print(f"\n  DISTRIBUSI RISK SCORING:")
for kategori, jumlah in dist_risiko.items():
    print(f"  {kategori:<15}: {jumlah} data ({jumlah/len(hasil)*100:.1f}%)")
print(f"\n  Kelompok 1 - Tugas Besar Machine Learning")
print(f"  Politeknik Negeri Indramayu | 2025/2026")
print("=" * 60)


# ============================================================
# EKSPOR MODEL UNTUK DASHBOARD
# ============================================================
import joblib

joblib.dump(model, 'output/model.joblib')
joblib.dump(scaler, 'output/scaler.joblib')
joblib.dump(le_encoders, 'output/encoders.joblib')

print("\n[EKSPOR] Model, Scaler, Encoder berhasil disimpan ke output/")
print("  -> output/model.joblib")
print("  -> output/scaler.joblib")
print("  -> output/encoders.joblib")
