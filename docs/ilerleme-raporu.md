# rag_with_oag — İlerleme Raporu

Plan: Claude Docs'taki "rag_with_oag — Kodlama Planı" dokümanı.
Proje dizini: `~/Projects/rag_with_oag`

---

## 1. Tur: M0 ve M1 (iskelet ve RAG)

RAG modu yerel Ollama modelleriyle (`nomic-embed-text` ve `gemma4:e4b`) uçtan uca çalışıyor. M1'in "5 satırla RAG cevabı" kapısı geçti, TestPyPI yayını ise yapılmadı.

Gerçek denemenin çıktısı, iki Türkçe dokümanla:

```
Taraflar, sözleşmeyi otuz gün önceden yazılı bildirimle feshedebilirler [C1].
Fesih bildirimi noter kanalıyla yapılmalıdır [C1].   →  kaynak: docs/hizmet.md · Hizmet Sözleşmesi
```

### Yapılanlar

- **M1 (v0.1), iskelet ve RAG:** Planın paket yapısı kuruldu: config, tipler, `store/` (bağlantı, şema, Cypher sorguları, repository), `llm/`, `embed/`, `ingest/`, `retrieval/`, `generation/`, `KnowledgeBase` (sync ve async) ve CLI.
  - Ingest tekrar çalıştırılınca aynı dokümanları atlıyor.
  - Yarıda kesilen bir ingest sonraki çalıştırmada tamamlanıyor.
  - Farklı bir embedding modeliyle veya boyutuyla açılan bir bilgi tabanı sessizce bozulmak yerine hata veriyor.
- **Testler:** 38 test geçiyor. Gerçek gömülü FalkorDB kullanılıyor, Docker ve network gerekmiyor. Kapsam %91. Ruff ve `mypy --strict` temiz.
- **CI:** Linux ve macOS, Python 3.12 ve 3.13. Tag gelince sadece TestPyPI'a yayın yapıyor; gerçek PyPI'a yayın, plandaki gibi M2 kapısına kadar kapalı. Bu workflow henüz GitHub'da hiç çalışmadı.
- **M0 spike:** `spikes/m0_falkordb.py` recall, gecikme, bellek ve skor yönünü ölçüyor.

### M0 sonuçları (sentetik vektörler)

| | 3k × 384 | 5k × 768 |
|---|---|---|
| ANN recall@10 | 0.995 | 0.974 |
| rag p95 | 1.2 ms | 1.8 ms |
| hybrid prototip p95 | 1.7 ms | 2.1 ms |
| Bellek | 20 MB | 46 MB |

Plan kapı için sayısal eşik vermiyordu. Önerilen eşikler: recall ≥ 0.95, rag p95 ≤ 50 ms ve hybrid p95 ≤ 150 ms; bunların birlikte netleştirilmesi gerekiyor. Ölçümler sentetik vektörlerle yapıldı, gerçek bir korpusla (`--embed fastembed --corpus DIR`) henüz çalıştırılmadı.

### Plana yansıtılması gereken bulgular

1. **FalkorDB'nin varsayılan `efRuntime=10` değeri recall'ı yaklaşık 0.5'e düşürüyor.** Bu, M0'ın yakaladığı asıl risk. İndeks `efRuntime=200` ile oluşturuluyor, böylece recall 0.99'a çıkıyor ve sorgu süresi 1 ms civarında kalıyor.
2. **Vektör skoru mesafe, benzerlik değil** (0 aynı vektör demek). Kütüphane bunu `1 - mesafe` ile benzerliğe çeviriyor.
3. **`adaptive-oci-chunking` boyutu ASCII'ye dayalı bir regex ile sayıyor**, bu yüzden Türkçe karakterli kelimeleri yanlış bölüyor. Tiktoken ile gerçek bir token üst sınırı uygulanıyor.
4. **falkordblite'ın async istemcisi kapanışta veriyi kaydetmeyebiliyor.** Bu yüzden sunucu sync istemciyle başlatılıyor, sorgular ise resmi async istemciyle unix socket üzerinden gönderiliyor.
5. **openai 3.x `httpx2` kullanıyor ve respx bunu yakalamıyor.** Planda "respx ile HTTP mock" yazıyordu; testlerde bunun yerine `httpx2`'nin kendi mock transport'u kullanılıyor.

---

## 2. Tur: M2 (ontoloji, oag ve hybrid)

Oag ve hybrid artık çalışıyor. Golden set'te multi-hop sorularda hybrid modun bağlamı cevabı her soruda içerdi, rag'de ise 10 sorunun 8'inde. Bu, M2 kapısını geçiyor, ama aşağıdaki çekinceyle.

Oag ve hybrid'in ilk turda neden yapılmadığı: bu bir kapsam kararıydı. Plan oag ve hybrid'i M2'ye koyduğu için M1'in sonunda durulmuştu. Oysa planın kendisi "çekirdek değer hybrid moddur" diyordu; ya devam edilmeli ya da baştan sorulmalıydı.

### Gerçek yerel modellerle golden set

Çıkarım ve cevap için `gemma4:e4b`, embedding için `nomic-embed-text` kullanıldı. Korpus 13 doküman ve 30 sorudan oluşuyor.

| Grup | rag | oag | hybrid |
|---|---|---|---|
| Tek hop | 100% | 70% | 100% |
| **Multi-hop** | 80% | 100% | **100%** |
| Serbest metin | 100% | 0% | 100% |
| **Toplam** | 93% | 57% | **100%** |

Tablodaki sayılar bağlam isabeti: beklenen cevap, LLM'e giden bağlamda var mı.

- **Çıkarım:** 13 entity ve 14 ilişki çıktı, beklenen sayılarla birebir aynı. Reddedilen öğe yok, Bulut tek düğümde birleşti.
- **Gecikme:** sorgu başına p50 yaklaşık 50 ms.
- **Ingest yavaş:** yerel modelle 13 doküman 8 dakika sürdü, doküman başına yaklaşık 37 saniye.

Sonuçlar beklentiye uyuyor. Oag serbest metin sorularını cevaplayamıyor, çünkü depozito veya faiz gibi bilgiler ontolojide yok. Hybrid ise iki modun güçlü yanlarını birleştiriyor.

Sonucu iki nedenle temkinli okumak gerekiyor:

- **Korpus çok küçük.** k=6 ile sorgu başına korpusun neredeyse yarısı geri geliyor, bu da rag'in skorunu şişiriyor. Gerçek bir korpusta aradaki farkın daha büyük olması beklenir, ama bu ölçülmedi.
- **Cevap doğruluğu henüz ölçülmedi.** Plan kapıyı cevap kalitesiyle tanımlıyor. Tablodaki metrik bağlam isabeti; cevap doğruluğu (`--answers`) 90 ayrı LLM çağrısı gerektiriyor, bu da yerel modelle yaklaşık 30–60 dakika demek. Ayrıca hybrid'in bağlamı rag'in iki katı token tutuyor.

### Neler eklendi

- **Ontoloji:** YAML veya Pydantic ile tanımlanıyor, tutarlılığı doğrulanıyor ve ontolojiden JSON Schema üretiliyor.
- **Çıkarım:** Yapılandırılmış çıktı için dört adımlı bir fallback zinciri var (json_schema → tool calling → json_object → prompt). Hatalı çıktı, hata mesajıyla modele bir kez geri gönderiliyor. Sonuçlar chunk başına önbelleğe alınıyor. Başarısız çıkarım bir sonraki ingest'te yeniden deneniyor.
- **Entity birleştirme:** Önce anahtar alana bakılıyor (örneğin vergi no), sonra katlanmış isme, sonra embedding benzerliğine.
- **Modlar:** Bir KB'ye ontoloji verilince hybrid varsayılan mod oluyor. Ontoloji sonradan eklenirse mevcut dokümanlar bir sonraki ingest'te çıkarılıyor. Ontoloji değişirse KB açılırken hata veriyor.
- **Testler:** 79 test geçiyor, kapsam %93.

### Testlerin yakaladığı ve düzeltilen hatalar

1. **Eşzamanlı ingest'te aynı şirket ikiye bölünüyordu.** Bu çok dokümanlı zinciri koparıyordu. Resolve ve yazma adımı sıraya alındı; LLM çıkarımı paralel kalmaya devam ediyor.
2. **FalkorDB full-text araması Türkçe büyük/küçük harfi eşleştirmiyor** ("şimşek" araması "Şimşek"i bulmuyor). İsimler artık katlanmış bir alanda (`name_norm`) aranıyor. Ekleri tolere etmek için prefix arama kullanılıyor ("sozles*" hem "sözleşme"yi hem "sözleşmesi"ni buluyor).
3. **Genel kelimeler isim eşleşmesini eziyordu.** "Sözleşme" kelimesi bütün sözleşmelerle eşleştiği için doğru şirket ilk sıraya çıkamıyordu. Linking'de isim eşleşmesine daha yüksek ağırlık verildi.

### Açık kalanlar

- Commit atılmadı. (3. turda yapıldı.)
- Sıradaki adaylar: cevap doğruluğu değerlendirmesi (`--answers`), ilk commit, M3 (`Ontology.suggest()`, streaming, değişen dokümanın yeniden ingest'i).

---

## 3. Tur: Hata düzeltmeleri ve cevap doğruluğu

Kod incelemesinde bulunan hatalar düzeltildi ve golden set ilk kez cevap doğruluğuyla ölçüldü. Çok adımlı sorularda graf işe yarıyor: cevap doğruluğu rag'de %64, hybrid ve oag'da %82. Aynı token bütçesiyle çalışan rag de %82'ye çıkıyor, ama bu korpusta bu karşılaştırma anlamsız; nedeni aşağıda.

### Düzeltilenler

1. **Entity birleştirmede sonsuz döngü.** Grafta anahtarsız duran bir şirket, aynı dokümanda hem anahtarsız hem anahtarlı geçince iki kimlik birbirini gösteriyor ve ingest donuyordu. Aynı yolda farklı vergi numaralı iki şirket de tek düğüme birleşiyordu. İkisi de düzeltildi; golden set'e bu durumu uçtan uca deneyen çok chunk'lı bir doküman eklendi (`kuzey_faaliyet.md`). Kuzey Yatırım artık vergi numarası ve şehriyle tek düğüm.
2. **Sayı dönüşümü.** `integer` alanında `2.5` değeri `25` oluyordu. Türkçe ve İngilizce basamak ayırıcıları artık okunuyor (`1.500.000`, `12,5`, `%40`).
3. **API anahtarı.** `OPENAI_API_KEY` başka sunuculara da gidiyordu; artık yalnızca OpenAI'ye gönderiliyor.
4. **Graf genişletme.** Merkezi bir entity bütçeyi kendi komşularıyla dolduruyordu: 2.000 entity'lik bir testte seed'in 50 doğrudan ilişkisinden yalnızca 1'i bağlama giriyordu. Yollar artık uzunluğa göre sıralanıyor ve 40'ı giriyor; gecikme 1,1 ms'den 2,4 ms'ye çıktı.
5. **Yapılandırılmış çıktı.** Bağlam taşması ve kesik JSON artık "bu yöntem desteklenmiyor" sayılmıyor. Model hiç cevap yazamadan token sınırına takılırsa sessizce boş cevap dönmek yerine `OutputTruncatedError` veriliyor.
6. **Diğer:** CLI kurulum hataları traceback yerine anlaşılır mesaj veriyor, `RetrievalConfig` sert üst sınırı kontrol ediyor, kapatılmış bir KB açık hata veriyor, embedding modellerinin sorgu ve doküman önekleri destekleniyor, eval'e eşit bütçe modu (`rag_eq`) eklendi.
7. **Repo:** `.DS_Store` dosyaları takipten çıkarıldı, ruff ve mypy sürümleri sabitlendi, README'deki kurulum komutu GitHub'dan kuruluma çevrildi (paket henüz PyPI'da değil). CI artık geçiyor.

### Golden set: 14 doküman, 33 soru

Çıkarım ve cevap için `gemma4:e4b` (16k bağlam penceresiyle), embedding için `nomic-embed-text` kullanıldı. Tablodaki değerler cevap doğruluğu; parantez içindekiler ortalama bağlam boyutu.

| Grup | rag | oag | hybrid | rag_eq |
|---|---|---|---|---|
| Tek adım | 91% | 73% | 100% | 100% |
| **Çok adım** | 64% | 82% | **82%** | 82% |
| Serbest metin | 100% | 0% | 100% | 100% |
| **Toplam** | 85% (2.217) | 52% (1.360) | **94%** (3.578) | 94% (3.792) |

- **İlk ölçüm yanıltıcıydı.** Ollama varsayılan olarak 4.096 tokenlık bağlam penceresi açıyor. `gemma4:e4b` bir thinking modeli ve uzun bağlamlarda kalan tokenleri düşünmeye harcayıp boş cevap döndü: 132 cevabın 14'ü boştu, hepsi 4.000 tokenin üstündeki bağlamlarda. Bu, ilk turda hybrid'i %55'e düşürmüştü. 16k pencereyle bu sorun kayboldu.
- **Graf çok adımlı sorularda fark yaratıyor:** rag %64, oag ve hybrid %82.
- **oag aynı başarıya üçte bir bağlamla ulaşıyor** (1.289'a karşı 3.688 token). Serbest metin sorularını ise hiç cevaplayamıyor, çünkü depozito veya faiz gibi bilgiler ontolojide yok.
- **Eşit bütçe karşılaştırması bu korpusta anlamsız.** Korpusun tamamı 3.720 token; `rag_eq` ortalama 3.792 tokenlık bağlamla zaten bütün korpusu modele veriyor. Hybrid'in eşit bütçede rag'den iyi olup olmadığını ölçmek için bağlama sığmayan, gerçek boyutlu bir korpus gerekiyor.

### Graf kalitesi: açık bulgular

Beklenen graf 13 entity ve 14 ilişki; çıkan graf 15 entity ve 15 ilişki.

1. **İsim varyantları ayrı düğüm oluyor.** Faaliyet raporu "Zirve Gıda" ve "Toros Kimya" diye ("A.Ş." olmadan) yazdığı için, ilişkisi olmayan iki fazla düğüm oluştu. Katlanmış isim eşleşmesi ve embedding benzerliği (0,92) bu varyantları birleştirmiyor. Hukuki ekleri (A.Ş., Ltd.) isim eşleştirmesinde yok saymak bir çözüm adayı.
2. **İlişki yönü yanlış çıkarılabiliyor.** `Bulut —ORTAGIDIR→ Delta` ters yazılmış, `Zirve —ORTAGIDIR→ Kuzey` ise doğru yönün yanına fazladan eklenmiş. "Delta'nın ortağı olduğu şirketler" sorusu bu yüzden yarım cevaplandı (yalnızca Manisa). İki ucu aynı tipte olan ilişkilerde doğrulama yönü yakalayamıyor; prompt'ta yön örneği vermek veya daha güçlü bir çıkarım modeli kullanmak gerekiyor.
3. **Oranlar karışık birimde.** Ontoloji `oran` alanını "yüzde olarak" tanımlıyor, model ise 0,6 yazdı.

### Açık kalanlar

- Gerçek boyutlu bir korpusla eşit bütçe karşılaştırması.
- Yukarıdaki üç graf kalitesi bulgusu.
- PyPI yayını: hesaplar ve trusted publisher tanımı bekleniyor; sürüm `0.2.0` olmalı.
- Ollama ile çalışırken bağlam penceresini artırmak (`OLLAMA_CONTEXT_LENGTH` veya `num_ctx`); README'de belirtilmeli.
- M3: `Ontology.suggest()`, streaming, değişen dokümanın yeniden ingest'i, çok değerli ilişkiler.
