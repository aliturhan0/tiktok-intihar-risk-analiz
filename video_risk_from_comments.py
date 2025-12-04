import os
import time
import random
from datetime import datetime

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from playwright.sync_api import sync_playwright

# -----------------------
#  AYARLAR
# -----------------------
DATA_PATH = "tiktok_results/tiktok_data.csv"
OUTPUT_PATH = "tiktok_results/tiktok_video_risk.csv"
PROFILE_DIR = "/Users/aliturhan/tiktok_profile"  # senin kullandığın profil
MAX_COMMENTS_PER_VIDEO = 20                      # her video için en fazla kaç yorum
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
#  MODEL YÜKLEME (RoBERTa)
# -----------------------
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

print("📥 Model yükleniyor...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(DEVICE)
id2label = model.config.id2label  # {0: 'negative', 1: 'neutral', 2: 'positive'}

def clean_text(text: str) -> str:
    text = str(text)
    return " ".join(text.split())

def human_wait(a=0.3, b=1.2):
    time.sleep(random.uniform(a, b))

def predict_label(text: str) -> str:
    text = clean_text(text)
    if not text.strip():
        return "NEU"

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    pred_id = int(probs.argmax())
    label = id2label[pred_id].upper()  # NEGATIVE / NEUTRAL / POSITIVE

    if "NEG" in label:
        return "NEG"
    if "POS" in label:
        return "POS"
    return "NEU"

def label_to_risk(label: str) -> float:
    if label == "NEG":
        return 1.0
    if label == "NEU":
        return 0.5
    return 0.0  # POS

def get_comments_for_video(page, max_comments=20):
    """
    Video sayfası açıkken görünen ilk yorumları toplar.
    TikTok DOM değişirse locator'ları buradan güncelleriz.
    """
    comments = []

    # Yorumların yüklenmesi için biraz bekle
    time.sleep(3)

    for _ in range(6):  # birkaç kez scroll yap
        try:
            # TikTok genelde comment-text için bu tür data-e2e kullanıyor
            loc = page.locator("[data-e2e='comment-text']")
            count = loc.count()

            for i in range(count):
                if len(comments) >= max_comments:
                    break
                try:
                    txt = loc.nth(i).inner_text().strip()
                    if txt and txt not in comments:
                        comments.append(txt)
                except Exception:
                    continue

            if len(comments) >= max_comments:
                break

            # Scroll ile biraz daha yorum yükle
            page.mouse.wheel(0, 1500)
            human_wait(0.5, 1.0)
        except Exception:
            break

    return comments

def main():
    if not os.path.exists(DATA_PATH):
        print(f"❌ Veri dosyası bulunamadı: {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    if "video_url" not in df.columns or "tag" not in df.columns:
        print("❌ CSV'de 'video_url' ve/veya 'tag' kolonları yok.")
        return

    results = []

    with sync_playwright() as p:
        browser = p.chromium.launch_persistent_context(
            user_data_dir=PROFILE_DIR,
            headless=False,
            viewport={"width": 390, "height": 844},  # mobile
            user_agent=(
                "Mozilla/5.0 (iPhone; CPU iPhone OS 15_2 like Mac OS X) "
                "AppleWebKit/605.1.15 (KHTML, like Gecko) "
                "Version/15.2 Mobile/15E148 Safari/604.1"
            ),
            has_touch=True,
            is_mobile=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-infobars",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ]
        )

        page = browser.new_page()

        for idx, row in df.iterrows():
            tag = row["tag"]
            url = row["video_url"]

            print(f"\n🎬 [{idx+1}/{len(df)}] {tag} → {url}")
            try:
                page.goto(url, timeout=90000)
            except Exception as e:
                print(f"❌ Video açılamadı: {e}")
                continue

            human_wait(1.0, 2.0)

            # Yorumları topla
            comments = get_comments_for_video(page, max_comments=MAX_COMMENTS_PER_VIDEO)
            print(f"💬 {len(comments)} yorum toplandı.")

            if not comments:
                # Yorum yoksa risk tanımsız → düşük verelim
                results.append({
                    "tag": tag,
                    "video_url": url,
                    "n_comments": 0,
                    "neg_count": 0,
                    "neu_count": 0,
                    "pos_count": 0,
                    "risk_score": 0.2,
                    "risk_level": "LOW"
                })
                continue

            labels = [predict_label(c) for c in comments]
            risks = [label_to_risk(l) for l in labels]

            neg_count = sum(1 for l in labels if l == "NEG")
            neu_count = sum(1 for l in labels if l == "NEU")
            pos_count = sum(1 for l in labels if l == "POS")

            avg_risk = float(sum(risks) / len(risks))

            if avg_risk >= 0.7:
                level = "HIGH"
            elif avg_risk >= 0.4:
                level = "MEDIUM"
            else:
                level = "LOW"

            results.append({
                "tag": tag,
                "video_url": url,
                "n_comments": len(comments),
                "neg_count": neg_count,
                "neu_count": neu_count,
                "pos_count": pos_count,
                "risk_score": round(avg_risk, 3),
                "risk_level": level
            })

        browser.close()

    out_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✅ VİDEO RİSK ANALİZİ TAMAMLANDI!")
    print(f"📄 Çıktı: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
