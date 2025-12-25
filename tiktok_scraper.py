import sys
import locale
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
locale.getpreferredencoding = lambda: "UTF-8"

from playwright.sync_api import sync_playwright, TimeoutError as PlayTimeout
import pandas as pd
import re
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import requests
from moviepy.editor import VideoFileClip
from faster_whisper import WhisperModel
import os
import cv2
import mediapipe as mp
from deepface import DeepFace

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ===========================
# SENTIMENT MODEL (caption)
# ===========================
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")


def get_sentiment(text: str):
    if not text or len(text.strip()) == 0:
        return "neutral", 0.0, 1.0, 0.0

    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    output = model(**encoded)
    probs = torch.nn.functional.softmax(output.logits, dim=1).detach().numpy()[0]

    labels = ["negative", "neutral", "positive"]
    pred = labels[int(np.argmax(probs))]
    return pred, float(probs[0]), float(probs[1]), float(probs[2])


# ===========================
# RISK METRİKLERİ
# ===========================
RISK_KEYWORDS = [
    "kill myself", "kys", "suicide", "end it all", "cutting", "self harm",
    "self-harm", "depressed", "depression", "worthless", "i wanna die",
    "i want to die", "cant do this", "can't do this", "hopeless", "alone",
    "nobody cares", "no one cares"
]


def temizle(text):
    text = str(text)
    text = re.sub(r"http\\S+", "", text)
    text = re.sub(r"\\s+", " ", text)
    return text.strip()


def keyword_risk(text):
    if not text:
        return 0.0
    t = text.lower()
    hits = sum(1 for kw in RISK_KEYWORDS if kw in t)
    if hits == 0:
        return 0.0
    return min(1.0, hits / 3.0)


def parse_count(s: str) -> int:
    if not s:
        return 0
    s = s.strip().upper()
    try:
        if s.endswith("K"):
            return int(float(s[:-1].replace(",", ".")) * 1000)
        if s.endswith("M"):
            return int(float(s[:-1].replace(",", ".")) * 1_000_000)
        return int(s.replace(",", ""))
    except Exception:
        return 0


def engagement_risk(like_count: int, comment_count: int) -> float:
    total = like_count + comment_count
    if total <= 0:
        return 0.0
    score = total / 100_000.0
    return max(0.0, min(1.0, score))


def compute_risk_score(caption: str, neg: float, like_str: str, comment_str: str, hashtag: str) -> float:
    kw_score = keyword_risk(caption)
    likes = parse_count(like_str)
    comments = parse_count(comment_str)
    eng_score = engagement_risk(likes, comments)

    hashtag_bonus = 0.0
    if hashtag.lower() in ["suicide", "killmyself", "selfharm", "self_harm"]:
        hashtag_bonus = 0.15

    raw = (
        0.5 * neg +
        0.3 * kw_score +
        0.2 * eng_score +
        hashtag_bonus
    )
    raw = max(0.0, min(1.0, raw))
    return round(raw * 100, 2)


# ===========================
# VIDEO DOWNLOAD
# ===========================
def download_video(video_url, out="current_video.mp4"):

    fallback_urls = [
        f"https://tikcdn.io/ssstik/{video_url}",
        f"https://www.tikwm.com/api/?url={video_url}",
        f"https://dl.snaptik.app/v1/?url={video_url}",
        f"https://api.vvmd.cc/tk/?url={video_url}"
    ]

    headers = {"User-Agent": "Mozilla/5.0"}

    for fb in fallback_urls:
        try:
            print(f" Fallback deniyor: {fb}")
            r = requests.get(fb, headers=headers, timeout=20, stream=True)

            if "application/json" in r.headers.get("Content-Type", ""):
                try:
                    data = r.json()
                    mp4 = data.get("data", {}).get("play", "")
                    if mp4:
                        r = requests.get(mp4, headers=headers, timeout=25, stream=True)
                except Exception:
                    pass

            if r.status_code == 200:
                with open(out, "wb") as f:
                    for chunk in r.iter_content(1024 * 64):
                        if chunk:
                            f.write(chunk)
                print(" Video indirildi:", out)
                return out

        except Exception as e:
            print("❌ Fallback hata:", e)

    print("❌ Video indirilemedi")
    return None


# ===========================
# TRANSCRIPT ÇIKAR
# ===========================
def extract_transcript_and_summary(video_path):
    transcript = ""
    summary = ""
    try:
        audio_path = "current_audio.mp3"
        print(" Ses çıkarılıyor...")
        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(audio_path, logger=None, verbose=False)

        print(" Faster-Whisper transcript alınıyor...")
        model = WhisperModel("small", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(audio_path)

        pieces = []
        for seg in segments:
            safe = seg.text.encode("utf-8", "ignore").decode("utf-8", "ignore")
            pieces.append(safe)
        transcript = " ".join(pieces).strip()

        summary = transcript[:300]

    except Exception as e:
        print("[!] Transcript/Summary hatası:", e)

    return transcript, summary


# ===========================
# VIDEO EMOTION ANALYSIS
# ===========================
try:
    mp_face = mp.solutions.face_detection
except AttributeError:
    mp_face = None



def video_emotion_analysis(video_path):
    """Hem yüz varsa mimik analizi, yoksa atmosfer analizi yapar."""

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        return {}, {}

    # --- Yüz kontrolü ---
    has_face = False
    if mp_face is not None:
      with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) as fd:

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        det = fd.process(rgb)
        if det.detections:
            has_face = True

    cap.release()

    # -------------------------
    # 1) Yüz varsa → DeepFace
    # -------------------------
    if has_face:
        try:
            analysis = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
            if isinstance(analysis, list):
                analysis = analysis[0]

            emo = analysis["emotion"]
            dominant = analysis["dominant_emotion"]

            return {
                "face_detected": True,
                "face_emotion": dominant,
                "face_emotion_score": float(emo.get(dominant, 0.0))
            }, {}
        except:
            pass

    # -------------------------
    # 2) Yüz yoksa → Atmosfer
    # -------------------------
    cap = cv2.VideoCapture(video_path)
    blur_list, bright_list, motion_list = [], [], []

    ret, prev_frame = cap.read()
    prev_g = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        blur_list.append(cv2.Laplacian(g, cv2.CV_64F).var())
        bright_list.append(np.mean(g))
        motion_list.append(np.mean(cv2.absdiff(prev_g, g)))

        prev_g = g

    cap.release()

    blur = float(np.mean(blur_list))
    bright = float(np.mean(bright_list))
    motion = float(np.mean(motion_list))

    # Atmosfer-duygu skorları
    sad = (1 - bright / 255) * 0.6 + (blur < 20) * 0.4
    happy = (bright / 255)
    angry = motion / 50
    calm = 1 - angry

    total = happy + sad + angry + calm
    happy /= total
    sad /= total
    angry /= total
    calm /= total

    return {
        "face_detected": False,
        "face_emotion": None,
        "face_emotion_score": 0.0
    }, {
        "visual_happy": float(happy),
        "visual_sad": float(sad),
        "visual_angry": float(angry),
        "visual_calm": float(calm),
        "visual_blur": blur,
        "visual_brightness": bright,
        "visual_motion": motion
    }


# ===========================
# ANA SCRAPER
# ===========================
def scrape_tiktok_hashtags(hashtags, limit=3):
    all_data = []

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False, channel="chrome",
            args=["--disable-blink-features=AutomationControlled",
                  "--disable-infobars", "--no-sandbox",
                  "--disable-dev-shm-usage"]
        )
        context = browser.new_context()
        page = context.new_page()

        for tag in hashtags:
            url = f"https://www.tiktok.com/tag/{tag}"
            print(f"\n🔎 {url} açılıyor...")

            try:
                page.goto(url, timeout=180000, wait_until="domcontentloaded")
            except:
                page.goto(url, timeout=180000, wait_until="load")

            input("👉 Doğrulama varsa çöz → Enter\n")

            print("📜 Scroll yapılıyor...")
            last_height = 0
            for _ in range(20):
                page.mouse.wheel(0, 6000)
                time.sleep(1)
                new_height = page.evaluate("() => document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height

            print(" Video linkleri alınıyor...")
            cards = page.locator("a[href*='/video/']")
            try:
                href_list = cards.evaluate_all("els => els.map(e => e.href)")
            except:
                href_list = []

            href_list = list(dict.fromkeys(href_list))
            print(f"{len(href_list)} link bulundu (#{tag})")

            href_list = href_list[:limit]

            for idx, video_url in enumerate(href_list, start=1):
                print(f"\n▶️ [{idx}/{len(href_list)}] Video: {video_url}")

                try:
                    page.goto(video_url, timeout=60000, wait_until="domcontentloaded")
                    time.sleep(2)
                except:
                    print("❌ Video açılmadı")
                    continue

                # Caption
                caption = ""
                for sel in [
                    'div[data-e2e="browse-video-desc"]',
                    'h1[data-e2e="video-desc"]',
                    'span[data-e2e="video-desc"]'
                ]:
                    try:
                        txt = page.locator(sel).inner_text(timeout=1000)
                        if txt.strip():
                            caption = txt.strip()
                            break
                    except:
                        pass

                like_text = ""
                comment_text = ""
                try:
                    like_text = page.locator('strong[data-e2e="like-count"]').inner_text(timeout=1000).strip()
                except:
                    pass
                try:
                    comment_text = page.locator('strong[data-e2e="comment-count"]').inner_text(timeout=1000).strip()
                except:
                    pass

                sent_label, neg, neu, pos = get_sentiment(caption)
                risk_yuzdesi = compute_risk_score(caption, neg, like_text, comment_text, tag)

                # ---- VIDEO INDIR ----
                video_path = download_video(video_url)

                # ---- YENİ: VIDEO EMOTION ANALYSIS ----
                face_info = {}
                visual_info = {}

                if video_path:
                    print("🎥 Video duygu analizi yapılıyor...")
                    face_info, visual_info = video_emotion_analysis(video_path)

                    print("🎙 Transcript alınıyor...")
                    transcript, summary = extract_transcript_and_summary(video_path)
                else:
                    transcript, summary = "", ""

                # VERIYI BİRLEŞTİR
                row = {
                    "hashtag": tag,
                    "caption": temizle(caption),
                    "like_raw": like_text,
                    "comment_raw": comment_text,
                    "sentiment": sent_label,
                    "neg": neg,
                    "neu": neu,
                    "pos": pos,
                    "risk_yuzdesi": risk_yuzdesi,
                    "transcript": transcript,
                    "summary": summary,
                    "video_url": video_url,
                }

                # Video emotion kolonları ekle
                row.update(face_info)
                row.update(visual_info)

                all_data.append(row)

        browser.close()

    return pd.DataFrame(all_data)


# ===========================
# RUN
# ===========================
if __name__ == "__main__":
    hashtags = ["depression", "anxiety", "suicide"]
    df = scrape_tiktok_hashtags(hashtags, limit=3)
    df.to_csv("tiktok_output.csv", index=False, encoding="utf-8-sig")
    print("\n✅ Kaydedildi: tiktok_output.csv")
    print(df.head())
