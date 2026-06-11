# ============================================================
# DASHBOARD DETEKSI FRAUD REIMBURSEMENT PERJALANAN DINAS
# Streamlit Executive Dashboard - Kelompok 1
# Politeknik Negeri Indramayu | 2025/2026
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# --- Page Config ---
st.set_page_config(
    page_title="Sistem Audit Fraud Reimbursement",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS: Executive Corporate Theme ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Global */
    html, body, .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #f4f6f9;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a5f 0%, #2c5282 100%);
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    [data-testid="stSidebar"] .stNumberInput label,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4 {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] input {
        color: #1a202c !important;
        background-color: #ffffff !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.2);
    }

    /* KPI Cards */
    .kpi-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 1px 6px rgba(0,0,0,0.07);
        border-left: 4px solid #2c5282;
        margin-bottom: 8px;
    }
    .kpi-card .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1a202c;
        line-height: 1.2;
    }
    .kpi-card .kpi-label {
        font-size: 0.85rem;
        color: #718096;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .kpi-card.green { border-left-color: #38a169; }
    .kpi-card.amber { border-left-color: #dd6b20; }
    .kpi-card.red   { border-left-color: #c53030; }

    /* Badge styles */
    .badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-approved {
        background: #c6f6d5; color: #22543d;
    }
    .badge-review {
        background: #fefcbf; color: #7b341e;
    }
    .badge-investigation {
        background: #fed7d7; color: #742a2a;
    }

    /* Result card for single audit */
    .result-card {
        border-radius: 16px;
        padding: 32px;
        text-align: center;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    }
    .result-card.approved {
        background: linear-gradient(135deg, #f0fff4 0%, #c6f6d5 100%);
        border: 2px solid #38a169;
    }
    .result-card.review {
        background: linear-gradient(135deg, #fffff0 0%, #fefcbf 100%);
        border: 2px solid #dd6b20;
    }
    .result-card.investigation {
        background: linear-gradient(135deg, #fff5f5 0%, #fed7d7 100%);
        border: 2px solid #c53030;
    }
    .result-card .result-icon { font-size: 3rem; }
    .result-card .result-title {
        font-size: 1.5rem; font-weight: 700; margin: 8px 0;
    }
    .result-card .result-score {
        font-size: 1.1rem; color: #4a5568;
    }

    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2c5282 100%);
        color: white;
        padding: 24px 32px;
        border-radius: 12px;
        margin-bottom: 24px;
    }
    .main-header h1 {
        color: white !important;
        font-size: 1.6rem;
        margin: 0;
    }
    .main-header p {
        color: #bee3f8;
        font-size: 0.9rem;
        margin: 4px 0 0 0;
    }

    /* Section titles */
    .section-title {
        color: #1e3a5f;
        font-weight: 600;
        font-size: 1.1rem;
        border-bottom: 2px solid #2c5282;
        padding-bottom: 8px;
        margin-bottom: 16px;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #ffffff;
        padding: 4px;
        border-radius: 10px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        font-weight: 500;
        padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2c5282 !important;
        color: white !important;
    }

    /* Table container */
    .dataframe-container {
        background: white;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 1px 6px rgba(0,0,0,0.06);
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL & ARTIFACTS
# ============================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load('output/model.joblib')
    scaler = joblib.load('output/scaler.joblib')
    encoders = joblib.load('output/encoders.joblib')
    with open('output/metrics.json', 'r') as f:
        metrics = json.load(f)
    return model, scaler, encoders, metrics

try:
    model, scaler, encoders, metrics = load_artifacts()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"⚠️ Gagal memuat model. Pastikan `main.py` sudah dijalankan terlebih dahulu.\n\nDetail: {e}")
    st.stop()


# ============================================================
# CONSTANTS
# ============================================================
FITUR = [
    'jenis_perjalanan', 'nominal_klaim', 'nominal_standar',
    'kelengkapan_dokumen', 'frekuensi_klaim_bulan',
    'selisih_hari_pengajuan', 'jabatan_pegawai', 'riwayat_fraud'
]
KOLOM_NUMERIK = ['nominal_klaim', 'nominal_standar',
                 'frekuensi_klaim_bulan', 'selisih_hari_pengajuan']
KOLOM_KATEGORIKAL = ['jenis_perjalanan', 'jabatan_pegawai']

JENIS_PERJALANAN_OPTIONS = ['dalam kota', 'luar kota', 'luar negeri']
JABATAN_OPTIONS = ['Staff', 'Supervisor', 'Manager']


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def kategorikan_risiko(score):
    if score <= 0.30:
        return 'Risiko Rendah'
    elif score <= 0.60:
        return 'Risiko Sedang'
    else:
        return 'Risiko Tinggi'

def rekomendasi_sistem(score):
    if score <= 0.30:
        return 'Disetujui Otomatis'
    elif score <= 0.60:
        return 'Tinjauan Manual'
    else:
        return 'Investigasi'

def predict_fraud(df_input, nominal_standar):
    """Preprocess and predict fraud for input dataframe."""
    df = df_input.copy()
    df['nominal_standar'] = nominal_standar

    # Encode categoricals
    for col in KOLOM_KATEGORIKAL:
        df[col] = encoders[col].transform(df[col].astype(str))

    # Ensure column order
    df_pred = df[FITUR].copy()

    # Scale numerics
    df_pred[KOLOM_NUMERIK] = scaler.transform(df_pred[KOLOM_NUMERIK])

    # Predict
    predictions = model.predict(df_pred)
    probabilities = model.predict_proba(df_pred)[:, 1]

    return predictions, probabilities

def render_kpi(label, value, css_class=""):
    return f"""
    <div class="kpi-card {css_class}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
    </div>
    """

def get_badge_html(rekomendasi):
    if rekomendasi == 'Disetujui Otomatis':
        return f'<span class="badge badge-approved">✅ {rekomendasi}</span>'
    elif rekomendasi == 'Tinjauan Manual':
        return f'<span class="badge badge-review">⚠️ {rekomendasi}</span>'
    else:
        return f'<span class="badge badge-investigation">🚨 {rekomendasi}</span>'


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### 🛡️ Sistem Audit Fraud")
    st.markdown("**Reimbursement Perjalanan Dinas**")
    st.markdown("Politeknik Negeri Indramayu")
    st.divider()

    st.markdown("#### ⚙️ Konfigurasi Audit")
    nominal_standar = st.number_input(
        "Plafon Kebijakan (Nominal Standar)",
        min_value=0,
        value=5000000,
        step=100000,
        format="%d",
        help="Batas wajar nominal klaim yang ditetapkan kebijakan instansi. "
             "Digunakan sebagai acuan evaluasi kewajaran pengajuan."
    )
    st.divider()

    st.markdown("#### 📋 Status Sistem")
    st.markdown(f"🟢 **Model Aktif**")
    st.markdown(f"📊 Data Pelatihan: **{metrics.get('total_data', '-'):,}** pengajuan")
    st.markdown(f"🎯 Tingkat Keandalan: **{metrics.get('accuracy', 0)*100:.1f}%**")
    st.markdown(f"🔍 Cakupan Deteksi: **{metrics.get('recall', 0)*100:.1f}%**")
    st.divider()
    st.caption("© 2026 Kelompok 1 — Politeknik Negeri Indramayu")


# ============================================================
# MAIN HEADER
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>🛡️ Dashboard Audit Fraud Reimbursement</h1>
    <p>Sistem pendukung keputusan untuk evaluasi pengajuan reimbursement perjalanan dinas</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3 = st.tabs([
    "📁  Audit Batch (Upload CSV)",
    "📝  Audit Tunggal",
    "📊  Transparansi & Analitik"
])


# ============================================================
# TAB 1: BATCH AUDIT
# ============================================================
with tab1:
    st.markdown('<div class="section-title">Unggah Data Pengajuan untuk Audit Massal</div>',
                unsafe_allow_html=True)

    st.info(
        "**Format CSV yang diharapkan (7 kolom):** "
        "`jenis_perjalanan`, `nominal_klaim`, `kelengkapan_dokumen`, "
        "`frekuensi_klaim_bulan`, `selisih_hari_pengajuan`, "
        "`jabatan_pegawai`, `riwayat_fraud`\n\n"
        "Kolom `nominal_standar` diambil dari **Plafon Kebijakan** di sidebar."
    )

    uploaded_file = st.file_uploader(
        "Pilih file CSV", type=["csv"], key="batch_upload"
    )

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.markdown(f"**📄 Data diterima:** {len(df_upload)} baris, {len(df_upload.columns)} kolom")

            required_cols = [c for c in FITUR if c != 'nominal_standar']
            missing = [c for c in required_cols if c not in df_upload.columns]
            if missing:
                st.error(f"❌ Kolom tidak ditemukan: **{', '.join(missing)}**. "
                         f"Pastikan CSV memiliki kolom: {', '.join(required_cols)}")
            else:
                if st.button("🔍 Jalankan Audit", type="primary", key="btn_batch"):
                    with st.spinner("Memproses audit..."):
                        preds, probs = predict_fraud(df_upload[required_cols], nominal_standar)

                    # Build results
                    hasil = df_upload.copy()
                    hasil['nominal_standar'] = nominal_standar
                    hasil['rasio_klaim'] = hasil['nominal_klaim'] / nominal_standar
                    hasil['Status'] = ['Normal' if p == 0 else 'Terindikasi Anomali' for p in preds]
                    hasil['Skor Risiko'] = np.round(probs, 4)
                    hasil['Kategori Risiko'] = [kategorikan_risiko(s) for s in probs]
                    hasil['Rekomendasi'] = [rekomendasi_sistem(s) for s in probs]

                    st.session_state['hasil_batch'] = hasil

        except Exception as e:
            st.error(f"Gagal memproses file: {e}")

    # Display results if available
    if 'hasil_batch' in st.session_state:
        hasil = st.session_state['hasil_batch']

        # KPI Cards
        total = len(hasil)
        n_approved = (hasil['Rekomendasi'] == 'Disetujui Otomatis').sum()
        n_review = (hasil['Rekomendasi'] == 'Tinjauan Manual').sum()
        n_invest = (hasil['Rekomendasi'] == 'Investigasi').sum()
        potensi_kerugian = hasil.loc[
            hasil['Rekomendasi'] == 'Investigasi', 'nominal_klaim'
        ].sum()

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(render_kpi("Total Pengajuan Diaudit", f"{total:,}"), unsafe_allow_html=True)
        with k2:
            st.markdown(render_kpi("Disetujui Otomatis", f"{n_approved:,}", "green"), unsafe_allow_html=True)
        with k3:
            st.markdown(render_kpi("Perlu Tinjauan", f"{n_review:,}", "amber"), unsafe_allow_html=True)
        with k4:
            st.markdown(render_kpi("Investigasi", f"{n_invest:,}", "red"), unsafe_allow_html=True)

        st.markdown("")

        # Extra KPI row
        e1, e2 = st.columns(2)
        with e1:
            st.markdown(render_kpi(
                "Potensi Kerugian Dicegah",
                f"Rp {potensi_kerugian:,.0f}"
            ), unsafe_allow_html=True)
        with e2:
            pct_aman = (n_approved / total * 100) if total > 0 else 0
            st.markdown(render_kpi(
                "Tingkat Pengajuan Aman",
                f"{pct_aman:.1f}%", "green"
            ), unsafe_allow_html=True)

        st.markdown("")

        # Quick Filter
        st.markdown('<div class="section-title">Hasil Audit Detail</div>', unsafe_allow_html=True)

        filter_col1, filter_col2 = st.columns([1, 3])
        with filter_col1:
            filter_status = st.selectbox(
                "Filter Rekomendasi",
                ["Semua", "Disetujui Otomatis", "Tinjauan Manual", "Investigasi"],
                key="filter_batch"
            )

        if filter_status != "Semua":
            df_display = hasil[hasil['Rekomendasi'] == filter_status].copy()
        else:
            df_display = hasil.copy()

        st.markdown(f"Menampilkan **{len(df_display)}** dari {total} pengajuan")

        # Color-coded dataframe
        def highlight_rekomendasi(val):
            if val == 'Disetujui Otomatis':
                return 'background-color: #c6f6d5; color: #22543d'
            elif val == 'Tinjauan Manual':
                return 'background-color: #fefcbf; color: #7b341e'
            elif val == 'Investigasi':
                return 'background-color: #fed7d7; color: #742a2a'
            return ''

        def highlight_status(val):
            if val == 'Normal':
                return 'color: #38a169; font-weight: 600'
            elif val == 'Terindikasi Anomali':
                return 'color: #c53030; font-weight: 600'
            return ''

        # Pandas Styler API may differ across environments. Attempt styling,
        # but fall back to plain DataFrame if Styler.applymap is unavailable.
        try:
            styled_df = df_display.style.applymap(
                highlight_rekomendasi, subset=['Rekomendasi']
            ).applymap(
                highlight_status, subset=['Status']
            ).format({
                'nominal_klaim': 'Rp {:,.0f}',
                'nominal_standar': 'Rp {:,.0f}',
                'rasio_klaim': '{:.2f}',
                'Skor Risiko': '{:.4f}'
            })

            st.dataframe(styled_df, use_container_width=True, height=400)
        except Exception:
            # Fallback: show plain DataFrame with formatted numeric columns
            df_plain = df_display.copy()
            df_plain['nominal_klaim'] = df_plain['nominal_klaim'].map(lambda x: f"Rp {x:,.0f}")
            df_plain['nominal_standar'] = df_plain['nominal_standar'].map(lambda x: f"Rp {x:,.0f}")
            df_plain['rasio_klaim'] = df_plain['rasio_klaim'].map(lambda x: f"{x:.2f}")
            df_plain['Skor Risiko'] = df_plain['Skor Risiko'].map(lambda x: f"{x:.4f}")
            st.dataframe(df_plain, use_container_width=True, height=400)

        # Download button
        csv_download = hasil.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Unduh Hasil Audit (CSV)",
            data=csv_download,
            file_name="hasil_audit_batch.csv",
            mime="text/csv",
            type="primary"
        )


# ============================================================
# TAB 2: SINGLE AUDIT
# ============================================================
with tab2:
    st.markdown('<div class="section-title">Formulir Audit Pengajuan Tunggal</div>',
                unsafe_allow_html=True)

    st.markdown(
        f"**Plafon Kebijakan aktif:** Rp {nominal_standar:,.0f} "
        f"_(dapat diubah di sidebar)_"
    )

    with st.form("form_single_audit"):
        col_a, col_b = st.columns(2)

        with col_a:
            jenis = st.selectbox("Jenis Perjalanan", JENIS_PERJALANAN_OPTIONS)
            nom_klaim = st.number_input("Nominal Klaim (Rp)", min_value=0,
                                         value=3000000, step=100000, format="%d")
            kelengkapan = st.radio("Kelengkapan Dokumen",
                                   ["Lengkap", "Tidak Lengkap"], horizontal=True)
            kelengkapan_val = 1.0 if kelengkapan == "Lengkap" else 0.75

        with col_b:
            frekuensi = st.number_input("Frekuensi Klaim Bulan Ini",
                                         min_value=1, value=1, step=1)
            selisih = st.number_input("Selisih Hari Pengajuan",
                                       min_value=0, value=3, step=1)
            jabatan = st.selectbox("Jabatan Pegawai", JABATAN_OPTIONS)
            riwayat = st.radio("Riwayat Anomali Sebelumnya",
                                ["Tidak", "Ya"], horizontal=True)
            riwayat_val = 1 if riwayat == "Ya" else 0

        submitted = st.form_submit_button("🔍 Evaluasi Pengajuan", type="primary",
                                           use_container_width=True)

    if submitted:
        # Build input dataframe
        df_single = pd.DataFrame([{
            'jenis_perjalanan': jenis,
            'nominal_klaim': nom_klaim,
            'kelengkapan_dokumen': kelengkapan_val,
            'frekuensi_klaim_bulan': frekuensi,
            'selisih_hari_pengajuan': selisih,
            'jabatan_pegawai': jabatan,
            'riwayat_fraud': riwayat_val
        }])

        with st.spinner("Mengevaluasi pengajuan..."):
            preds, probs = predict_fraud(df_single, nominal_standar)

        pred = preds[0]
        prob = probs[0]
        kategori = kategorikan_risiko(prob)
        rekom = rekomendasi_sistem(prob)
        rasio = nom_klaim / nominal_standar if nominal_standar > 0 else 0

        # Determine card style
        if rekom == 'Disetujui Otomatis':
            card_class = "approved"
            icon = "✅"
            color = "#22543d"
        elif rekom == 'Tinjauan Manual':
            card_class = "review"
            icon = "⚠️"
            color = "#7b341e"
        else:
            card_class = "investigation"
            icon = "🚨"
            color = "#742a2a"

        st.markdown("")

        # Result card
        st.markdown(f"""
        <div class="result-card {card_class}">
            <div class="result-icon">{icon}</div>
            <div class="result-title" style="color: {color};">{rekom}</div>
            <div class="result-score">
                Skor Risiko: <strong>{prob:.4f}</strong> &nbsp;|&nbsp;
                Kategori: <strong>{kategori}</strong>
            </div>
            <div style="margin-top:8px; color: #4a5568; font-size: 0.9rem;">
                Rasio Klaim: <strong>{rasio:.2f}x</strong> dari plafon kebijakan
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        # Detail breakdown
        with st.expander("📋 Detail Evaluasi", expanded=True):
            d1, d2, d3 = st.columns(3)
            d1.metric("Status", "Normal" if pred == 0 else "Terindikasi Anomali")
            d2.metric("Nominal Klaim", f"Rp {nom_klaim:,.0f}")
            d3.metric("Plafon Kebijakan", f"Rp {nominal_standar:,.0f}")

            d4, d5, d6 = st.columns(3)
            d4.metric("Jenis Perjalanan", jenis.title())
            d5.metric("Jabatan", jabatan)
            d6.metric("Kelengkapan Dok.", kelengkapan)


# ============================================================
# TAB 3: EXPLAINABLE AI
# ============================================================
with tab3:
    st.markdown('<div class="section-title">Transparansi & Analitik Model</div>',
                unsafe_allow_html=True)

    st.markdown(
        "Bagian ini menjelaskan **kriteria evaluasi** yang digunakan sistem "
        "dalam menentukan status pengajuan, serta performa historis sistem."
    )

    st.markdown("")

    # Metrics cards
    st.markdown("#### 📈 Performa Sistem")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tingkat Keandalan", f"{metrics.get('accuracy',0)*100:.1f}%")
    m2.metric("Ketepatan Deteksi", f"{metrics.get('precision',0)*100:.1f}%")
    m3.metric("Cakupan Deteksi", f"{metrics.get('recall',0)*100:.1f}%")
    m4.metric("Skor Keseimbangan", f"{metrics.get('f1_score',0)*100:.1f}%")

    st.markdown("")
    st.divider()

    # Feature Importance
    col_viz1, col_viz2 = st.columns(2)

    with col_viz1:
        st.markdown("#### 🎯 Kriteria Evaluasi Utama")
        st.markdown(
            "Grafik di bawah menunjukkan **kriteria mana yang paling berpengaruh** "
            "terhadap keputusan sistem saat mengevaluasi pengajuan."
        )
        viz6_path = 'output/viz6_feature_importance.png'
        if os.path.exists(viz6_path):
            st.image(viz6_path, use_container_width=True)
        else:
            st.warning("Visualisasi belum tersedia. Jalankan `main.py` terlebih dahulu.")

    with col_viz2:
        st.markdown("#### 🧪 Akurasi Deteksi Historis")
        st.markdown(
            "**Confusion Matrix** menunjukkan seberapa akurat sistem "
            "membedakan pengajuan normal dari yang terindikasi anomali "
            "berdasarkan data pengujian historis."
        )
        viz5_path = 'output/viz5_confusion_matrix.png'
        if os.path.exists(viz5_path):
            st.image(viz5_path, use_container_width=True)
        else:
            st.warning("Visualisasi belum tersedia. Jalankan `main.py` terlebih dahulu.")

    st.divider()

    # Data summary
    st.markdown("#### 📊 Ringkasan Data Pelatihan")
    s1, s2, s3 = st.columns(3)
    s1.metric("Total Data Historis", f"{metrics.get('total_data',0):,}")
    s2.metric("Pengajuan Normal", f"{metrics.get('total_normal',0):,}")
    s3.metric("Pengajuan Anomali", f"{metrics.get('total_fraud',0):,}")
