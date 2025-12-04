import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import numpy as np
from joblib import load

# ================================
#   DOSYA YOLLARI
# ================================
INPUT_CSV = "tiktok_output.csv"
OUTPUT_CSV = "tiktok_results/tiktok_data_with_transcript_risk.csv"

# RANDOM FOREST MODEL & TF-IDF
RF_MODEL_PATH = "models/random_forest_model.pkl"
TFIDF_PATH = "models/tfidf_vectorizer.pkl"
ENCODER_PATH = "models/label_encoder.pkl"

# ================================
#   MODEL
# ================================
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================
#   RİSKLİ KELİMELER
# ================================
RISK_KEYWORDS = [
    "kill myself", "kys", "suicide", "end it all", "cutting", "self harm",
    "self-harm", "depressed", "depression", "worthless", "i wanna die",
    "i want to die", "can't do this", "cant do this", "hopeless", "alone",
    "nobody cares", "no one cares", "help me", "save me", "hate myself",
    "why am i alive", "i give up"
]

# ================================
#   LOAD MODELS
# ================================
def load_transformer():
    print("📥 Roberta sentiment modeli yükleniyor...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    return tokenizer, model


def load_random_forest():
    print("🌲 Random Forest yükleniyor...")
    rf_model = load(RF_MODEL_PATH)
    tfidf = load(TFIDF_PATH)
    label_encoder = load(ENCODER_PATH)
    return rf_model, tfidf, label_encoder

# ================================
#   TRANSFORMER SENTIMENT
# ================================
def predict_sentiment(text, tokenizer, model):
    if not isinstance(text, str) or text.strip() == "":
        text = "neutral"

    inputs = tokenizer(
        text, return_tensors="pt", truncation=True,
        padding=True, max_length=256
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        scores = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    labels = ["NEG", "NEU", "POS"]
    pred_idx = int(np.argmax(scores))
    return labels[pred_idx], float(scores[pred_idx])

# ================================
#   RULE-BASED RISK
# ================================
def base_risk_from_tag(tag: str) -> float:
    tag = (tag or "").lower()
    if tag == "depression": return 0.53
    if tag == "anxiety":    return 0.43
    if tag == "suicide":    return 0.73
    if tag == "psychosis":  return 0.63
    return 0.20


def keyword_risk_boost(text: str) -> float:
    if not isinstance(text, str) or text.strip() == "":
        return 0.0
    t = text.lower()
    hits = sum(1 for kw in RISK_KEYWORDS if kw in t)
    if hits == 0: return 0.0
    if hits == 1: return 0.15
    if hits == 2: return 0.25
    return 0.40


def adjust_risk_with_sentiment(base_risk: float, sentiment: str) -> float:
    if sentiment == "NEG": base_risk += 0.10
    elif sentiment == "POS": base_risk -= 0.05
    return max(0.0, min(1.0, base_risk))


def compute_final_risk(tag: str, sentiment: str, caption: str, transcript: str) -> float:
    merged = (caption + " " + transcript).lower()
    base = base_risk_from_tag(tag)
    base = adjust_risk_with_sentiment(base, sentiment)
    boost = keyword_risk_boost(merged)
    final = base + boost
    return max(0.0, min(1.0, final))


def risk_level(score: float) -> str:
    if score >= 0.70: return "HIGH"
    if score >= 0.40: return "MEDIUM"
    return "LOW"

# ================================
#   RANDOM FOREST PREDICT
# ================================
def rf_predict(text, rf_model, tfidf, label_encoder):
    X_vec = tfidf.transform([text])

    # sınıf olasılıkları
    proba = rf_model.predict_proba(X_vec)[0]

    # LOW, MEDIUM, HIGH
    score_low = proba[label_encoder.transform(["LOW"])[0]]
    score_med = proba[label_encoder.transform(["MEDIUM"])[0]]
    score_high = proba[label_encoder.transform(["HIGH"])[0]]

    # sayısal risk skoru
    rf_score = (
        score_low * 0.1 +
        score_med * 0.5 +
        score_high * 1.0
    )

    # sınıf etiketi
    cls = rf_model.predict(X_vec)[0]
    label = label_encoder.inverse_transform([cls])[0]

    return label, float(rf_score)

# ================================
#   ANA ÇALIŞMA
# ================================
def main():
    if not os.path.exists(INPUT_CSV):
        print(f"❌ Giriş dosyası bulunamadı: {INPUT_CSV}")
        return

    print("📥 CSV okunuyor...")
    df = pd.read_csv(INPUT_CSV)

    tokenizer, model = load_transformer()
    rf_model, tfidf, label_encoder = load_random_forest()

    sentiments = []
    risk_scores = []
    risk_numeric = []
    risk_levels = []

    rf_labels = []
    rf_scores = []

    total = len(df)

    for idx, row in df.iterrows():

        tag = row.get("hashtag", "")
        caption = row.get("caption", "")
        transcript = row.get("transcript", "")

        caption = caption if isinstance(caption, str) else ""
        transcript = transcript if isinstance(transcript, str) else ""
        merged_text = (caption + " " + transcript).strip()

        # TRANSFORMER SENTIMENT
        sent_label, _ = predict_sentiment(merged_text, tokenizer, model)

        # RULE RISK
        final_risk = compute_final_risk(tag, sent_label, caption, transcript)
        level = risk_level(final_risk)

        # RANDOM FOREST RISK
        rf_label, rf_score = rf_predict(merged_text, rf_model, tfidf, label_encoder)

        print(f"[{idx+1}/{total}] TAG={tag} | SENT={sent_label} | RULE={final_risk:.3f} | RF={rf_label} ({rf_score:.3f})")

        sentiments.append(sent_label)
        risk_scores.append(final_risk)
        risk_numeric.append(round(final_risk, 3))  # <<< yüzdesel değil, sayısal
        risk_levels.append(level)

        rf_labels.append(rf_label)
        rf_scores.append(rf_score)

    # CSV'ye yaz
    df["sentiment_v2"] = sentiments
    df["risk_score_v2"] = risk_scores
    df["risk_numeric"] = risk_numeric        # <<< yeni isim
    df["risk_level_v2"] = risk_levels

    df["rf_risk_label"] = rf_labels
    df["rf_risk_score"] = rf_scores

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("\n✅ TÜM ANALİZ TAMAMLANDI!")
    print(f"📄 Yeni kayıt: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
