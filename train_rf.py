import os, sys, re
import pandas as pd
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier

DF_PATH = "tiktok_results/tiktok_data_with_transcript_risk.csv"
OUT_DIR = "models"

# Öncelik sırası: önce gerçek etiket, yoksa pseudo-label
LABEL_PRIORITY = ["true_label", "mm_risk_level", "risk_level_v2", "rf_risk_label"]

TEXT_COLS = ["caption", "transcript", "summary"]  # varsa hepsini kullanacağız

def clean_text(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.lower()

    # url / kullanıcı / hashtag temizliği (aşırı temizleme yapmıyoruz)
    s = re.sub(r"http\S+", " ", s)
    s = re.sub(r"@\w+", " ", s)
    s = re.sub(r"#\w+", " ", s)

    # emoji/punct kalsın diye agresif silme yok
    s = re.sub(r"\s+", " ", s).strip()
    return s

def pick_label_col(df: pd.DataFrame):
    for c in LABEL_PRIORITY:
        if c in df.columns:
            return c
    return None

def main():
    print("📥 Veri yükleniyor:", DF_PATH)
    if not os.path.exists(DF_PATH):
        print(f"❌ Veri seti bulunamadı: {DF_PATH}")
        sys.exit(1)

    df = pd.read_csv(DF_PATH)

    # text kolonları kontrol (olanları al)
    available_text_cols = [c for c in TEXT_COLS if c in df.columns]
    if not available_text_cols:
        print("❌ caption/transcript/summary kolonlarından hiçbiri yok!")
        print("Mevcut kolonlar:", df.columns.tolist())
        sys.exit(1)

    label_col = pick_label_col(df)
    if not label_col:
        print("❌ Label kolonu bulunamadı. true_label/mm_risk_level/risk_level_v2/rf_risk_label yok.")
        print("Mevcut kolonlar:", df.columns.tolist())
        sys.exit(1)

    # text birleştir
    def merge_row(r):
        parts = []
        for c in available_text_cols:
            parts.append(clean_text(r.get(c, "")))
        return " ".join([p for p in parts if p]).strip()

    df["text"] = df.apply(merge_row, axis=1)

    # label normalize
    df["label"] = df[label_col].astype(str).str.upper().str.strip()
    df = df[df["text"].str.len() > 10].copy()
    df = df[df["label"].isin(["LOW", "MEDIUM", "HIGH"])].copy()

    print(f"\n✅ Kullanılan label: {label_col}")
    print("✅ Kullanılan text kolonları:", available_text_cols)
    print("\n📊 Label dağılımı:")
    print(df["label"].value_counts())

    # Minimum şart
    if len(df) < 100:
        print("\n❌ Veri çok az. 100+ öneririm, 'çok doğru' için 300–1000+ lazım.")
        sys.exit(1)

    # Encode
    le = LabelEncoder()
    y = le.fit_transform(df["label"].values)
    print("\n🧾 Label Mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

    # Split (dengeli)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].values,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # --- Multilingual Feature Set ---
    # 1) Word TF-IDF: anlam yakalar (TR/EN karışık idare eder)
    word_tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=60000,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )

    # 2) Char TF-IDF: dil bağımsız, argo/yanlış yazım/emoji çevresi yakalar
    char_tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=80000,
        min_df=2,
        max_df=0.98,
        sublinear_tf=True
    )

    features = FeatureUnion([
        ("word", word_tfidf),
        ("char", char_tfidf),
    ])

    # RF (balanced)
    rf = RandomForestClassifier(
        n_estimators=800,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1
    )

    model = Pipeline([
        ("features", features),
        ("rf", rf)
    ])

    print("\n🌲 RF eğitiliyor (word+char TF-IDF)...")
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("\n✅ CONFUSION MATRIX")
    print(confusion_matrix(y_test, y_pred))

    print("\n✅ CLASSIFICATION REPORT")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save: pipeline + encoder
    os.makedirs(OUT_DIR, exist_ok=True)
    dump(model, os.path.join(OUT_DIR, "random_forest_model.pkl"))
    dump(le, os.path.join(OUT_DIR, "label_encoder.pkl"))

    # Not: eskiden ayrı tfidf kaydediyordun.
    # Bu yeni yöntemde TF-IDF pipeline içinde. Yine de uyumluluk için ayrıca kaydedelim:
    dump(model.named_steps["features"], os.path.join(OUT_DIR, "tfidf_vectorizer.pkl"))

    print("\n✅ MODEL EĞİTİMİ TAMAMLANDI!")
    print("📦 Kaydedilen dosyalar:")
    print(" - models/random_forest_model.pkl   (pipeline: word+char tfidf + RF)")
    print(" - models/tfidf_vectorizer.pkl      (FeatureUnion)")
    print(" - models/label_encoder.pkl")

if __name__ == "__main__":
    main()
