import streamlit as st
import pandas as pd
import random
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(page_title="Simulasi Piket - Model Baru", layout="wide")

st.title("📊 Simulasi Sistem Piket IT Del - Model Pipeline Baru")

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("⚙️ Parameter Sistem")

JUMLAH_MEJA = st.sidebar.number_input("Jumlah Meja", 20, 120, 60)
MIN_TIME = st.sidebar.slider("Durasi Minimum (detik)", 10, 60, 30)
MAX_TIME = st.sidebar.slider("Durasi Maksimum (detik)", 30, 120, 60)

st.sidebar.markdown("---")
st.sidebar.write("👥 3 Petugas Lauk")
st.sidebar.write("🚚 2 Petugas Angkat")
st.sidebar.write("🍚 2 Petugas Nasi")
st.sidebar.write("🕖 Mulai 07:00")

START_TIME = datetime.strptime("07:00:00", "%H:%M:%S")

PETUGAS = {
    "lauk": 3,
    "angkat": 2,
    "nasi": 2
}

# =========================================================
# MODEL PIPELINE SIMULATION
# =========================================================

def run_simulation():

    meja = list(range(1, JUMLAH_MEJA + 1))

    waktu_lauk = [START_TIME] * PETUGAS["lauk"]
    waktu_angkat = [START_TIME] * PETUGAS["angkat"]
    waktu_nasi = [START_TIME] * PETUGAS["nasi"]

    selesai_lauk = {}
    selesai_angkat = {}
    selesai_nasi = {}

    # ======================
    # 1️⃣ LAUK
    # ======================
    for m in meja:
        idx = waktu_lauk.index(min(waktu_lauk))
        start = waktu_lauk[idx]
        dur = random.randint(MIN_TIME, MAX_TIME)
        finish = start + timedelta(seconds=dur)

        waktu_lauk[idx] = finish
        selesai_lauk[m] = finish

    # ======================
    # 2️⃣ ANGKAT (2 meja per trip)
    # ======================
    sorted_ready = sorted(selesai_lauk.items(), key=lambda x: x[1])
    queue = [m[0] for m in sorted_ready]

    while queue:
        batch = queue[:2]
        queue = queue[2:]

        idx = waktu_angkat.index(min(waktu_angkat))
        ready_time = max(selesai_lauk[m] for m in batch)
        start = max(ready_time, waktu_angkat[idx])

        dur = sum(random.randint(MIN_TIME, MAX_TIME) for _ in batch)
        finish = start + timedelta(seconds=dur)

        waktu_angkat[idx] = finish

        for m in batch:
            selesai_angkat[m] = finish

    # ======================
    # 3️⃣ NASI
    # ======================
    for m in meja:
        idx = waktu_nasi.index(min(waktu_nasi))
        start = max(selesai_angkat[m], waktu_nasi[idx])
        dur = random.randint(MIN_TIME, MAX_TIME)
        finish = start + timedelta(seconds=dur)

        waktu_nasi[idx] = finish
        selesai_nasi[m] = finish

    df = pd.DataFrame({
        "Meja": meja,
        "Waktu Selesai": [selesai_nasi[m] for m in meja]
    })

    df["Durasi Total (detik)"] = (
        (df["Waktu Selesai"] - START_TIME)
        .dt.total_seconds()
    )

    return df

# =========================================================
# TOMBOL JALANKAN
# =========================================================

if st.button("🚀 Jalankan Simulasi"):

    df = run_simulation()

    total_waktu = df["Durasi Total (detik)"].max()
    selesai_jam = START_TIME + timedelta(seconds=total_waktu)

    # =====================================================
    # KPI
    # =====================================================

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Meja", JUMLAH_MEJA)
    col2.metric("Jam Selesai", selesai_jam.strftime("%H:%M:%S"))
    col3.metric("Durasi Total (menit)", round(total_waktu / 60, 2))

    st.divider()

    # =====================================================
    # 1️⃣ Diagram Batang 60 Meja
    # =====================================================

    st.subheader("📊 Durasi Penyelesaian per Meja")

    fig_bar = px.bar(
        df,
        x="Meja",
        y="Durasi Total (detik)",
        title="Durasi Penyelesaian Setiap Meja"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # =====================================================
    # 2️⃣ Histogram Distribusi
    # =====================================================

    st.subheader("📈 Distribusi Durasi")

    fig_hist = px.histogram(
        df,
        x="Durasi Total (detik)",
        nbins=20,
        title="Distribusi Waktu Penyelesaian"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # =====================================================
    # 3️⃣ Cumulative Completion Curve
    # =====================================================

    st.subheader("📈 Kurva Penyelesaian Kumulatif")

    df_sorted = df.sort_values("Waktu Selesai")
    df_sorted["Jumlah Selesai"] = range(1, len(df_sorted) + 1)

    fig_line = px.line(
        df_sorted,
        x="Waktu Selesai",
        y="Jumlah Selesai",
        markers=True,
        title="Progress Penyelesaian Meja"
    )
    st.plotly_chart(fig_line, use_container_width=True)

    # =====================================================
    # 4️⃣ Analisis Bottleneck
    # =====================================================

    rata = df["Durasi Total (detik)"].mean()
    maks = df["Durasi Total (detik)"].max()

    st.info(f"""
Rata-rata durasi penyelesaian meja adalah {round(rata,2)} detik.
Meja paling lambat selesai pada {round(maks,2)} detik.
""")

    st.balloons()