import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack
import joblib
import os

INPUT_CSV = "tiktok_output.csv"
MODEL_PATH = "models/rf_risk_model.pkl"

# ============================
#  RISK LABEL OLUŞTURMA
# ============================
def risk_percent_to_label(x):
    if x >= 70:
        return "HIGH"
    elif x >= 40:
        return "MEDIUM"
    return "LOW"

print("📥 CSV okunuyor...")
df = pd.read_csv(INPUT_CSV)

# === Eksik transcript/caption temizle
df["caption"] = df["caption"].fillna("")
df["transcript"] = df["transcript"].fillna("")

# === Etiket kolonunu oluştur
df["risk_label"] = df["risk_yuzdesi"].apply(risk_percent_to_label)

# === Modelde kullanılacak metin
df["full_text"] = df["caption"] + " " + df["transcript"]

# ============================
#  TF-IDF TEXT VEKTÖRLEME
# ============================
print("🔤 TF-IDF hazırlanıyor...")
tfidf = TfidfVectorizer(max_features=5000)
X_text = tfidf.fit_transform(df["full_text"])

# ============================
#  LABEL ENCODER
# ============================
print("🏷 Etiketler encode ediliyor...")
le = LabelEncoder()
y = le.fit_transform(df["risk_label"])

# ============================
#  TRAIN / TEST
# ============================
print("📊 Train/Test bölünüyor...")
X_train, X_test, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42
)

# ============================
#  RANDOM FOREST EĞİTİMİ
# ============================
print("🌲 Random Forest eğitiliyor...")
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

acc = rf.score(X_test, y_test)
print(f"🎯 Test Accuracy: {acc:.4f}")

# ============================
#  KAYDET
# ============================
os.makedirs("models", exist_ok=True)

joblib.dump({
    "model": rf,
    "tfidf": tfidf,
    "label_encoder": le
}, MODEL_PATH)

print(f"📦 Model kaydedildi → {MODEL_PATH}")
