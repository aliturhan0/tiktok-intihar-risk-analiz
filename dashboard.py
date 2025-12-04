import streamlit as st
import pandas as pd
import plotly.express as px


# ============================
#   DASHBOARD SETTINGS
# ============================
st.set_page_config(
    page_title="TikTok Risk Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 TikTok Psikolojik Risk Analizi Dashboard")

# ============================
#   LOAD CSV
# ============================
@st.cache_data
def load_data():
    df = pd.read_csv("tiktok_results/tiktok_data_with_transcript_risk.csv")

    # Final Hybrid Score
    df["final_score"] = (df["risk_score_v2"] * 0.6) + (df["rf_risk_score"] * 0.4)

    # Final Level
    def hybrid_level(x):
        if x >= 0.70: return "HIGH"
        if x >= 0.40: return "MEDIUM"
        return "LOW"

    df["final_level"] = df["final_score"].apply(hybrid_level)

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
    df_filtered = df_filtered[df_filtered["transcript"].str.contains(search_text, case=False, na=False)]

# ============================
#   METRIC CARDS
# ============================
col1, col2, col3, col4 = st.columns(4)

col1.metric("Ortalama Rule Risk", f"{df_filtered['risk_score_v2'].mean():.3f}")
col2.metric("Ortalama RF Risk", f"{df_filtered['rf_risk_score'].mean():.3f}")
col3.metric("Hybrid Final Skor", f"{df_filtered['final_score'].mean():.3f}")
col4.metric("Toplam Video", len(df_filtered))

# ============================
#   RISK LEVEL PIE CHART
# ============================
st.subheader("📌 Final Risk Seviyeleri Dağılımı")

pie = px.pie(
    df_filtered,
    names="final_level",
    title="Risk Seviyesi Dağılımı",
    color="final_level",
    color_discrete_map={"HIGH": "red", "MEDIUM": "orange", "LOW": "green"},
)
st.plotly_chart(pie, use_container_width=True)

# ============================
#   SCATTER PLOT (Rule vs RF)
# ============================
st.subheader("📈 Rule Risk vs RF Risk Scatter Plot")

scatter = px.scatter(
    df_filtered,
    x="risk_score_v2",
    y="rf_risk_score",
    color="final_level",
    size="final_score",
    hover_data=["caption", "transcript", "video_url"],
    labels={"risk_score_v2": "Rule-Based Risk", "rf_risk_score": "Random Forest Risk"}
)
st.plotly_chart(scatter, use_container_width=True)

# ============================
#   BAR CHART BY HASHTAG
# ============================
st.subheader("📊 Hashtag Bazında Risk Karşılaştırması")

tag_bar = px.bar(
    df_filtered,
    x="hashtag",
    y="final_score",
    color="final_level",
    title="Hashtag → Final Risk Dağılımı",
    color_discrete_map={"HIGH": "red", "MEDIUM": "orange", "LOW": "green"},
)
st.plotly_chart(tag_bar, use_container_width=True)

# ============================
#   DATA TABLE
# ============================
st.subheader("📄 Tüm Veriler (Filtrelenmiş)")
st.dataframe(df_filtered, use_container_width=True)

