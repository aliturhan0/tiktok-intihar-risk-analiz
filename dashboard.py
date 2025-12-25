import streamlit as st
import pandas as pd
import plotly.express as px

# ============================
#   DASHBOARD SETTINGS
# ============================
st.set_page_config(
    page_title="TikTok Multimodal Risk Dashboard",
    page_icon="",
    layout="wide"
)

st.title("TikTok Multimodal Psikolojik Risk Analizi ")


# ============================
#   LOAD CSV
# ============================
@st.cache_data
def load_data():
    df = pd.read_csv("tiktok_results/tiktok_data_with_transcript_risk.csv")

    # MULTIMODAL risk skorları CSV’den direkt geliyor
    df["mm_risk_score"] = df["mm_risk"]
    df["mm_level"] = df["mm_risk_level"]

    # Text risk (rule-based)
    df["text_risk_score"] = df["risk_score_v2"]

    # RF risk
    df["rf_risk"] = df["rf_risk_score"]

    # Video Risk
    df["video_risk_score"] = df["video_risk"]

    return df


df = load_data()


# ============================
#   SIDEBAR FILTERS
# ============================
st.sidebar.header("🔎 Filtreler")

hashtags = df["hashtag"].dropna().unique().tolist()
selected_tag = st.sidebar.multiselect("Hashtag filtrele:", hashtags, default=hashtags)

search_text = st.sidebar.text_input("Metin içinde ara:", "")

df_filtered = df[df["hashtag"].isin(selected_tag)]

if search_text.strip() != "":
    df_filtered = df_filtered[
        df_filtered["transcript"].str.contains(search_text, case=False, na=False)
    ]


# ============================
#   METRIC CARDS
# ============================
st.subheader("📊 Özet Metrikler")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("📝 Text Risk (Avg)", f"{df_filtered['text_risk_score'].mean():.3f}")
col2.metric("🌲 RF Risk (Avg)", f"{df_filtered['rf_risk'].mean():.3f}")
col3.metric("🎥 Video Risk (Avg)", f"{df_filtered['video_risk_score'].mean():.3f}")
col4.metric("🎯 Multimodal Risk (Avg)", f"{df_filtered['mm_risk_score'].mean():.3f}")
col5.metric("📦 Toplam Video", len(df_filtered))


# ============================
#   PIE CHART (Multimodal Levels)
# ============================
st.subheader("📌 Multimodal Risk Seviyeleri Dağılımı")

pie = px.pie(
    df_filtered,
    names="mm_level",
    title="Multimodal Risk Dağılımı",
    color="mm_level",
    color_discrete_map={"HIGH": "red", "MEDIUM": "orange", "LOW": "green"},
)
st.plotly_chart(pie, use_container_width=True)


# ============================
#   SCATTER: TEXT vs VIDEO
# ============================
st.subheader("🎥 Text Risk vs Video Risk Scatter Plot")

scatter = px.scatter(
    df_filtered,
    x="text_risk_score",
    y="video_risk_score",
    color="mm_level",
    size="mm_risk_score",
    hover_data=["caption", "transcript", "video_url"],
    labels={
        "text_risk_score": "Text Risk",
        "video_risk_score": "Video Risk"
    }
)
st.plotly_chart(scatter, use_container_width=True)


# ============================
#   BAR CHART: HASHTAG
# ============================
st.subheader("📊 Hashtag Bazında Multimodal Risk")

tag_bar = px.bar(
    df_filtered,
    x="hashtag",
    y="mm_risk_score",
    color="mm_level",
    title="Hashtag → Multimodal Risk Dağılımı",
    color_discrete_map={"HIGH": "red", "MEDIUM": "orange", "LOW": "green"},
)
st.plotly_chart(tag_bar, use_container_width=True)


# ============================
#   DATA TABLE
# ============================
st.subheader("📄 Filtrelenmiş Tüm Veriler")

st.dataframe(
    df_filtered[
        [
            "hashtag", "caption", "transcript",
            "text_risk_score", "rf_risk", "video_risk_score",
            "mm_risk_score", "mm_level", "video_url"
        ]
    ],
    use_container_width=True
)
