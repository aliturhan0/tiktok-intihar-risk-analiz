📌 TikTok Psikolojik Risk Analizi Sistemi
TikTok Videolarından Otomatik Transcript Çıkarma, Sentiment Analizi ve Psikolojik Risk Skoru Üretme
🧠 Projenin Amacı
Bu proje, TikTok üzerindeki videolardan elde edilen metinsel içerikleri (caption/transcript) otomatik olarak analiz ederek psikolojik risk içeren davranışları erken tespit etmeyi amaçlayan bir NLP ve veri işleme sistemidir.
Ana hedef:
TikTok videolarındaki depresyon, anksiyete, intihar, öz kıyım, kendine zarar verme gibi risk temalarını otomatik olarak tespit etmek
RoBERTa tabanlı sentiment analizi ile duygu skorları çıkarmak
Risk anahtar kelimelerini tarayarak risk skoru hesaplamak
Analiz edilen veri setini CSV formatında saklamak
Dashboard üzerinden sonucu görselleştirmek
🚀 Sistem Mimarisi
1. TikTok Scraper (Playwright)
TikTok video sayfasını açar
Caption (transcript) içeriğini DOM üzerinden çeker
Metni temizler (emoji, @, #, URL, fazla boşluk)
CSV olarak kaydeder
Kullanılan CSS seçicisi:
strong[data-e2e='browse-video-desc']
2. Transcript → RoBERTa Analizi
Her tiranskript şu adımlardan geçer:
✔ 1) Tokenization (Byte-Pair Encoding)
Metin subword birimlerine ayrılır.
Örneğin:
"I'm tired of everything" → ['I', "'", 'm', 'tired', 'of', 'every', 'thing']
✔ 2) Self-Attention
Her kelime diğer tüm kelimelerle bağlam ilişkisi kurar.
Bu sayede model cümlenin duygusal tonunu çözer.
✔ 3) 12 Katmanlı Transformer Encoder
Multi-Head Attention
LayerNorm
Feed-Forward Network
Transcript katmanlar boyunca anlam bakımından derinleştirilir.
✔ 4) CLS Embedding
Model, tüm cümlenin anlamını temsil eden [CLS] vektörünü üretir (768 boyut).
✔ 5) Sentiment Sınıflandırma
CLS → Softmax → [neg, neu, pos] olasılıklarını döndürür.
Örnek:
negative: 0.68
neutral : 0.25
positive: 0.07
Kullanılan model:
cardiffnlp/twitter-roberta-base-sentiment-latest
3. Risk Anahtar Kelime Tespiti
Transcript belirli risk kelimeleri sözlüğü ile taranır.
Örnek:
["die", "kill myself", "suicide", "worthless", "tired of life", ...]
Her eşleşmede risk puanı artırılır.
4. Risk Skoru Hesabı
Risk skoru sentiment ve keyword analizinin birleşimidir:
risk_score = neg*0.6 + neu*0.2 + pos*0.0
keyword_bonus = risk_keyword_count * 0.15
final_risk = risk_score + keyword_bonus
Bu formül proje başlangıcında literatüre uygun olarak belirlenen heuristik bir parametre yapısıdır.
İleride Random Forest / SHAP / Grid Search ile optimize edilecektir.
📊 Dashboard (Streamlit)
Bu sistem ile:
Her videonun sentiment değerleri
Risk kelime sayısı
Final risk puanı
Hashtag bazlı analiz
Filtrelemeler
görsel arayüzde listelenir.
Komut:
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
