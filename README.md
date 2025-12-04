📌 TikTok Psikolojik Risk Analizi Sistemi
TikTok Videolarından Otomatik Transcript Çıkarma, Sentiment Analizi ve Psikolojik Risk Skoru Üretme
🧠 Projenin Amacı
Bu proje, TikTok üzerindeki videolardan elde edilen metinsel içerikleri (caption/transcript) otomatik olarak analiz ederek psikolojik risk içeren davranışları erken tespit etmeyi amaçlayan bir NLP ve veri işleme sistemidir.
Projenin hedefleri:
TikTok videolarındaki depresyon, anksiyete, intihar, öz kıyım gibi risk temalarını otomatik tespit etmek
RoBERTa tabanlı sentiment analizi ile duygu skorlarını çıkarmak
Risk anahtar kelimelerini tarayarak risk skoru hesaplamak
Analiz edilen sonuçları CSV olarak kaydetmek
Dashboard üzerinden sonuçları görselleştirmek
🚀 Sistem Mimarisi
1. TikTok Scraper (Playwright)
Scraper şu işlemleri yapar:
TikTok video sayfasını açar
Caption (transcript) içeriğini DOM üzerinden çeker
Metni temizler (emoji, @, #, URL, fazla boşluk)
Sonuçları CSV'ye kaydeder
Kullanılan CSS seçicisi:
strong[data-e2e='browse-video-desc']
2. Transcript → RoBERTa Analizi
Her transcript aşağıdaki adımlardan geçer:
✔ 1) Tokenization (Byte-Pair Encoding)
Metin subword birimlerine ayrılır.
Örnek:
"I'm tired of everything"
→ ['I', "'", 'm', 'tired', 'of', 'every', 'thing']
✔ 2) Self-Attention
Her kelime, cümlenin diğer tüm kelimeleriyle bağlam ilişkisi kurar.
Bu sayede model duygusal tonlamayı çözer.
✔ 3) Transformer Encoder (12 Katman)
Multi-Head Attention
LayerNorm
Feed-Forward Network
Bu katmanlar transcript’in anlamını derinleştirir.
✔ 4) CLS Embedding
Model tüm cümlenin anlamını temsil eden [CLS] vektörünü (768 boyut) üretir.
✔ 5) Sentiment Sınıflandırma
CLS → Linear Layer → Softmax yoluyla üç olasılık döner:
negative: 0.68
neutral : 0.25
positive: 0.07
Kullanılan model:
cardiffnlp/twitter-roberta-base-sentiment-latest
3. Risk Anahtar Kelime Tespiti
Transcript, risk kelimeleri sözlüğüyle taranır.
Örnek liste:
["die", "kill myself", "suicide",
 "worthless", "tired of life", ...]
Eşleşen her kelime risk puanını artırır.
4. Risk Skoru Hesabı
Risk skoru sentiment + riskli kelime sayısına göre hesaplanır:
risk_score = neg*0.6 + neu*0.2 + pos*0.0
keyword_bonus = risk_keyword_count * 0.15
final_risk = risk_score + keyword_bonus
Bu formül başlangıç aşamasında heuristik olarak belirlenmiştir.
İleride Random Forest / SHAP / Grid Search ile optimize edilecektir.
📊 Dashboard (Streamlit)
Dashboard üzerinden:
Sentiment değerleri
Risk kelime sayısı
Final risk puanı
Hashtag bazlı analiz
Filtreleme ve sıralama
gibi özellikler sunulur.
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
