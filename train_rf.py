import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from joblib import dump
import os
import sys

# ============================
#  VERİ SETİ YOLU
# ============================
DF_PATH = "tiktok_results/tiktok_data_with_transcript_risk.csv"

print("📥 Veri yükleniyor:", DF_PATH)

if not os.path.exists(DF_PATH):
    print(f"❌ Veri seti bulunamadı: {DF_PATH}")
    sys.exit(1)

df = pd.read_csv(DF_PATH)

# ============================
#  GEREKLİ KOLONLAR
# ============================
REQUIRED_COLUMNS = ["transcript", "rf_risk_label"]

for col in REQUIRED_COLUMNS:
    if col not in df.columns:
        print(f"❌ Veri setinde '{col}' kolonu yok!")
        print("Mevcut kolonlar:", df.columns.tolist())
        sys.exit(1)

# ============================
#  EĞİTİM VERİLERİ
# ============================

# text → transcript
df["text"] = df["transcript"].astype(str)

# label → string olduğu için doğrudan alıyoruz
df["label"] = df["rf_risk_label"].astype(str)

print("📊 Örnek satır:")
print(df[["text", "label"]].head())

# ============================
#  TF-IDF + LABEL ENCODER
# ============================
print("🔤 TF-IDF oluşturuluyor...")

tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df["text"])

print("🏷️ Label Encoder oluşturuluyor...")

le = LabelEncoder()
y = le.fit_transform(df["label"])

print("🧾 Label Mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

# ============================
#  RANDOM FOREST EĞİT
# ============================
print("🌲 Random Forest eğitiliyor...")

rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

rf.fit(X, y)

# ============================
#  MODELLERİ KAYDET
# ============================
os.makedirs("models", exist_ok=True)
dump(tfidf, "models/tfidf_vectorizer.pkl")
dump(rf, "models/random_forest_model.pkl")
dump(le, "models/label_encoder.pkl")

print("\n✅ MODEL EĞİTİMİ TAMAMLANDI!")
print("📦 Kaydedilen dosyalar:")
print(" - models/tfidf_vectorizer.pkl")
print(" - models/random_forest_model.pkl")
print(" - models/label_encoder.pkl")
