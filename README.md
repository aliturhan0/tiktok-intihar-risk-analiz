📌 TikTok Psikolojik Risk Analizi Sistemi
TikTok Videolarından Otomatik Transcript Çıkarma, Sentiment Analizi ve Psikolojik Risk Skoru Üretme
🧠 Projenin Amacı
Bu proje, TikTok üzerindeki videolardan elde edilen metinsel içerikleri (caption/transcript) analiz ederek psikolojik risk içeren davranışları erken tespit etmeyi amaçlar.
Ana hedefler:
Depresyon, anksiyete, intihar, öz kıyım temalarını tespit etmek
RoBERTa ile sentiment skorlarını çıkarmak
Risk anahtar kelimelerini taramak
Risk skorunu hesaplamak
Dashboard’da görselleştirmek
🚀 Sistem Mimarisi
1. TikTok Scraper (Playwright)
Scraper şu işlemleri yapar:
TikTok video sayfasını açar
Caption (transcript) içeriğini DOM üzerinden çeker
Metni temizler (emoji, URL, hashtag, mention)
CSV olarak kaydeder
CSS seçicisi:
strong[data-e2e='browse-video-desc']
2. Transcript → RoBERTa Analizi
Her transcript şu 5 aşamadan geçer:
✔ 1) Tokenization (Byte-Pair Encoding)
Metin subword birimlerine ayrılır.
Örnek:
"I'm tired of everything"
→ ['I', "'", 'm', 'tired', 'of', 'every', 'thing']
✔ 2) Self-Attention
Her kelime, cümlenin diğer tüm kelimeleriyle bağlam ilişkisi kurar.
Model duygusal tonlamayı bu şekilde çözer.
✔ 3) Transformer Encoder (12 Katman)
Multi-Head Attention
LayerNorm
Feed-Forward Network
✔ 4) CLS Embedding
Model, tüm cümlenin anlamını temsil eden [CLS] vektörünü (768 boyut) üretir.
✔ 5) Sentiment Sınıflandırma
CLS → Linear Layer → Softmax yoluyla üç olasılık döner:
negative: 0.68
neutral : 0.25
positive: 0.07
Kullanılan model:
cardiffnlp/twitter-roberta-base-sentiment-latest
3. Risk Anahtar Kelime Tespiti
Transcript, risk kelimeleri sözlüğüyle taranır:
["die", "kill myself", "suicide", "worthless", "tired of life", ...]
Her eşleşen kelime risk puanını artırır.
4. Risk Skoru Hesabı
Risk skoru sentiment + riskli kelime sayısına göre hesaplanır:
risk_score = neg*0.6 + neu*0.2 + pos*0.0
keyword_bonus = risk_keyword_count * 0.15
final_risk = risk_score + keyword_bonus
📊 Dashboard (Streamlit)
Dashboard’ta aşağıdaki özellikler bulunur:
Sentiment değerleri
Risk kelime sayısı
Final risk puanı
Hashtag analizi
Filtreleme / sıralama
Başlatma komutu:
streamlit run dashboard.py
🛠️ Kurulum (Local)
1. Ortamı oluştur
python3 -m venv venv
source venv/bin/activate
2. Bağımlılıkları kur
pip install -r requirements.txt
playwright install
3. Analizi başlat
python sentiment_analysis.py
4. Dashboard
streamlit run dashboard.py
