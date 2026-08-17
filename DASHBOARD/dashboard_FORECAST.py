"""
Dashboard Interaktif — Peramalan Nilai Tukar USD/IDR (ARIMA vs LSTM)
=====================================================================
Cara menjalankan:
    1. Pastikan folder `dashboard_data/` (hasil ekspor dari notebook SKRIPSI)
       berada satu direktori dengan file ini.
    2. Install dependensi:
           pip install streamlit plotly pandas
    3. Jalankan:
           streamlit run streamlit_app.py
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------
# KONFIGURASI HALAMAN
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Prediksi Kurs USD/IDR",
    page_icon="📈",
    layout="wide",
)

DATA_DIR = "dashboard_data"


# --------------------------------------------------------------------------
# LOAD DATA (hasil ekspor dari notebook, tidak mengubah pipeline aslinya)
# --------------------------------------------------------------------------
@st.cache_data
def load_data():
    df_hist = pd.read_csv(f"{DATA_DIR}/dashboard_actual.csv", parse_dates=["Date"])
    df_test = pd.read_csv(f"{DATA_DIR}/dashboard_test_predictions.csv", parse_dates=["Date"])
    df_future = pd.read_csv(f"{DATA_DIR}/dashboard_future_forecast.csv", parse_dates=["Date"])
    df_compare = pd.read_csv(f"{DATA_DIR}/dashboard_model_comparison.csv")
    meta = pd.read_csv(f"{DATA_DIR}/dashboard_meta.csv").iloc[0]
    return df_hist, df_test, df_future, df_compare, meta


try:
    df_hist, df_test, df_future, df_compare, meta = load_data()
except FileNotFoundError:
    st.error(
        "File data belum ditemukan. Jalankan seluruh sel di notebook `SKRIPSI.ipynb` "
        "(termasuk sel ekspor data dashboard di bagian paling akhir) terlebih dahulu, "
        "lalu pastikan folder `dashboard_data/` berada di direktori yang sama dengan "
        "`streamlit_app.py` ini."
    )
    st.stop()

best_order = meta["best_order"]
best_model_name = meta["best_model_name"]


# --------------------------------------------------------------------------
# HEADER
# --------------------------------------------------------------------------
st.title("📈 Dashboard Peramalan Kurs USD/IDR")
st.caption("Perbandingan Model ARIMA vs LSTM")

min_date = df_hist["Date"].min().date()
max_date = df_future["Date"].max().date()

# --------------------------------------------------------------------------
# SIDEBAR — KONTROL INTERAKTIF
# --------------------------------------------------------------------------
st.sidebar.header("⚙️ Pengaturan Tampilan")

default_start = max(min_date, (df_hist["Date"].max() - pd.Timedelta(days=180)).date())
date_range = st.sidebar.date_input(
    "Rentang tanggal grafik",
    value=(default_start, max_date),
    min_value=min_date,
    max_value=max_date,
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_d, end_d = date_range
else:
    start_d, end_d = min_date, max_date

st.sidebar.markdown("**Tampilkan garis:**")
show_actual = st.sidebar.checkbox("Data Aktual", value=True)
show_arima = st.sidebar.checkbox("Prediksi ARIMA (data uji)", value=True)
show_lstm = st.sidebar.checkbox("Prediksi LSTM (data uji)", value=True)
show_future = st.sidebar.checkbox("Peramalan ke Depan", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("🔎 Cek Nilai pada Tanggal Tertentu")
selected_date = st.sidebar.date_input(
    "Pilih tanggal",
    value=max_date,
    min_value=min_date,
    max_value=max_date,
)

st.sidebar.markdown("---")
st.sidebar.subheader("🏆 Model Terbaik")
st.sidebar.success(f"{best_model_name}\n\nMAPE terkecil pada data uji")


# --------------------------------------------------------------------------
# FILTER DATA SESUAI RENTANG TANGGAL
# --------------------------------------------------------------------------
def filter_range(df):
    return df[(df["Date"].dt.date >= start_d) & (df["Date"].dt.date <= end_d)]


hist_f = filter_range(df_hist)
test_f = filter_range(df_test)
future_f = filter_range(df_future)


# --------------------------------------------------------------------------
# GRAFIK UTAMA (interaktif — zoom, hover, pan)
# --------------------------------------------------------------------------
fig = go.Figure()

if show_actual:
    fig.add_trace(go.Scatter(
        x=hist_f["Date"], y=hist_f["USD_IDR"],
        name="Aktual", line=dict(color="black", width=2),
    ))
if show_arima:
    fig.add_trace(go.Scatter(
        x=test_f["Date"], y=test_f["ARIMA"],
        name=f"ARIMA{best_order}", line=dict(color="orange", width=2, dash="dash"),
    ))
if show_lstm:
    fig.add_trace(go.Scatter(
        x=test_f["Date"], y=test_f["LSTM"],
        name="LSTM (univariat)", line=dict(color="green", width=2, dash="dash"),
    ))
if show_future:
    fig.add_trace(go.Scatter(
        x=future_f["Date"], y=future_f["Forecast"],
        name="Peramalan ke Depan", line=dict(color="red", width=2, dash="dot"),
        mode="lines+markers", marker=dict(size=4),
    ))

sel_ts = pd.Timestamp(selected_date)
if start_d <= selected_date <= end_d:
    fig.add_vline(x=sel_ts, line_width=1, line_dash="dot", line_color="gray")

fig.update_layout(
    height=520,
    xaxis_title="Tanggal",
    yaxis_title="Kurs USD/IDR",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=10, r=10, t=30, b=10),
)

st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------
# NILAI PADA TANGGAL TERPILIH (mencari tanggal bursa terdekat)
# --------------------------------------------------------------------------
st.subheader(f"📅 Detail Nilai — {selected_date.strftime('%d %B %Y')}")

all_dates = pd.concat([df_hist["Date"], df_test["Date"], df_future["Date"]]).drop_duplicates()
nearest_date = all_dates.iloc[(all_dates - sel_ts).abs().argsort().iloc[0]]

if nearest_date != sel_ts:
    st.caption(
        f"Tanggal {selected_date.strftime('%d %b %Y')} bukan hari bursa / di luar data. "
        f"Menampilkan tanggal terdekat: **{nearest_date.strftime('%d %b %Y')}**."
    )

row_hist = df_hist[df_hist["Date"] == nearest_date]
row_test = df_test[df_test["Date"] == nearest_date]
row_future = df_future[df_future["Date"] == nearest_date]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Aktual", f"{row_hist['USD_IDR'].values[0]:,.0f}" if not row_hist.empty else "—")
c2.metric("Prediksi ARIMA", f"{row_test['ARIMA'].values[0]:,.0f}" if not row_test.empty and not np.isnan(row_test['ARIMA'].values[0]) else "—")
c3.metric("Prediksi LSTM", f"{row_test['LSTM'].values[0]:,.0f}" if not row_test.empty and not np.isnan(row_test['LSTM'].values[0]) else "—")
c4.metric("Peramalan ke Depan", f"{row_future['Forecast'].values[0]:,.0f}" if not row_future.empty else "—")


# --------------------------------------------------------------------------
# TABEL PERBANDINGAN MODEL
# --------------------------------------------------------------------------
st.subheader("🏆 Perbandingan Performa Model (Data Uji)")
st.dataframe(df_compare, use_container_width=True, hide_index=True)

st.caption(
    "Pedoman interpretasi MAPE (Lewis, 1982): <10% sangat akurat · 10–20% baik · "
    "20–50% layak/cukup · >50% tidak akurat."
)

# --------------------------------------------------------------------------
# DATA MENTAH (opsional, bisa dilihat & diunduh)
# --------------------------------------------------------------------------
with st.expander("📄 Lihat data mentah pada rentang tanggal terpilih"):
    tab1, tab2, tab3 = st.tabs(["Aktual", "Prediksi Data Uji", "Peramalan ke Depan"])
    with tab1:
        st.dataframe(hist_f, use_container_width=True, hide_index=True)
    with tab2:
        st.dataframe(test_f, use_container_width=True, hide_index=True)
    with tab3:
        st.dataframe(future_f, use_container_width=True, hide_index=True)
