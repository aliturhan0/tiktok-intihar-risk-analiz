from joblib import load
import pandas as pd

RF_MODEL_PATH = "models/random_forest_model.pkl"
TFIDF_PATH = "models/tfidf_vectorizer.pkl"
ENCODER_PATH = "models/label_encoder.pkl"

INPUT_CSV = "tiktok_output.csv"

# ----------------------------
# Load
# ----------------------------
rf = load(RF_MODEL_PATH)
tfidf = load(TFIDF_PATH)
le = load(ENCODER_PATH)

print("\n=== CLASS ORDER CHECK ===")
print("rf.classes_            :", rf.classes_)
print("label_encoder.classes_ :", le.classes_)

# ----------------------------
# Vocabulary / feature check (NNZ)
# ----------------------------
print("\n=== TF-IDF FEATURE CHECK (NNZ) ===")
feature_tests = [
    "kill myself suicide self harm cutting scars",
    "panic attack anxiety cant breathe",
    "today was amazing happy great",
    "funny video lol hahaha",
]
for s in feature_tests:
    X = tfidf.transform([s])
    print(f"TEXT: {s}")
    print("nnz(nonzero features) =", X.nnz)

# ----------------------------
# Safe RF predict (no class-order bug)
# ----------------------------
def rf_predict_safe(text: str):
    X = tfidf.transform([text])
    proba = rf.predict_proba(X)[0]  # order = rf.classes_

    # map class -> label -> proba
    cls_to_label = {c: le.inverse_transform([c])[0] for c in rf.classes_}
    label_to_proba = {cls_to_label[c]: proba[i] for i, c in enumerate(rf.classes_)}

    rf_score = (
        label_to_proba.get("LOW", 0.0) * 0.1 +
        label_to_proba.get("MEDIUM", 0.0) * 0.5 +
        label_to_proba.get("HIGH", 0.0) * 1.0
    )

    pred_cls = rf.predict(X)[0]
    pred_label = le.inverse_transform([pred_cls])[0]
    return pred_label, float(rf_score), label_to_proba

# ----------------------------
# 1) Quick sanity test
# ----------------------------
print("\n=== SANITY TEST (manual texts) ===")
tests = [
    "i want to die. nobody cares. kill myself",
    "panic attack, anxiety, cant breathe",
    "i feel sad and hopeless, i give up",
    "today was amazing i feel great and happy",
    "funny video lol hahaha",
    "self harm cutting scars help me",
]

for t in tests:
    label, score, probs = rf_predict_safe(t)
    print("\nTEXT:", t)
    print("RF_LABEL:", label, "RF_SCORE:", round(score, 3))
    print("PROBS:", {k: round(float(v), 3) for k, v in probs.items()})

# ----------------------------
# 2) Your real data distribution test
# ----------------------------
print("\n=== DATASET DISTRIBUTION TEST ===")
try:
    df = pd.read_csv(INPUT_CSV)

    labels = []
    scores = []
    nnzs = []

    for _, row in df.iterrows():
        cap = row.get("caption", "")
        tr = row.get("transcript", "")
        text = (str(cap) + " " + str(tr)).strip()

        X = tfidf.transform([text])
        nnzs.append(int(X.nnz))

        label, score, _ = rf_predict_safe(text)
        labels.append(label)
        scores.append(score)

    df["rf_label_test"] = labels
    df["rf_score_test"] = scores
    df["tfidf_nnz"] = nnzs

    print("\nLABEL COUNTS:")
    print(df["rf_label_test"].value_counts(dropna=False))

    print("\nSCORE SUMMARY:")
    print(df["rf_score_test"].describe())

    print("\nTF-IDF NNZ SUMMARY (0 ise TF-IDF metni tanımıyor demek):")
    print(df["tfidf_nnz"].describe())
    print("\nNNZ == 0 COUNT:", int((df["tfidf_nnz"] == 0).sum()))

    df.to_csv("rf_test_output.csv", index=False, encoding="utf-8-sig")
    print("\nSaved: rf_test_output.csv (kontrol için)")

except FileNotFoundError:
    print(f"\n❌ {INPUT_CSV} bulunamadı. Önce scraper çalışıp tiktok_output.csv oluşmalı.")
