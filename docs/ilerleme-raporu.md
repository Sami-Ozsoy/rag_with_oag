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

---

## 4. Tur: Graf kalitesi düzeltmeleri ve DeepSeek ile ölçüm

3. turda bulunan üç graf kalitesi hatası düzeltildi. Golden set artık `gemma4:e4b` yerine Ollama cloud üzerindeki `deepseek-v4.1-flash:1017-cloud` ile ölçülüyor; embedding modeli yine yerel `nomic-embed-text`. Bu modelle hybrid, çok adımlı sorularda eşit bütçeli rag'i de geçiyor. Ama bu sonuç düzeltmelerden değil, modelden geliyor: eski kod da DeepSeek ile aynı sonucu veriyor.

### Düzeltilenler

1. **Hukuki ekler artık ayrı düğüm oluşturmuyor.** İsim eşleştirmesinde sondaki hukuki ekler (A.Ş., Ltd. Şti., Inc., GmbH vb.) yok sayılıyor: "Zirve Gıda A.Ş." ile "Zirve Gıda" aynı şirket sayılıyor. Kimlik, `name_norm` ve iki birleştirme adımı (aynı yazım içinde ve grafa karşı) bu eşleştirmeyi kullanıyor. Ek listesi ontolojideki `name_suffixes` alanıyla genişletilebiliyor. Farklı vergi numaralı iki şirket, isimleri yalnızca ekte farklı olsa da ayrı kalıyor.
2. **Eski bilgi tabanları sessizce bozulmuyor.** Geçerli ek listesi ontoloji fingerprint'ine dahil. Normalizasyonun değiştiği eski bir bilgi tabanı açılırken `OntologyMismatchError` veriliyor; iki farklı normalizasyon aynı grafta karışmıyor.
3. **İlişki yönü:** Çıkarım prompt'u ilişkiyi "<kaynak> <İLİŞKİ> <hedef>" cümlesi olarak okumayı, ters kurulmuş cümlelerde (edilgen yapı, iyelik) bile kaynağı açıklamaya göre seçmeyi ve metin söylemedikçe ters ilişki eklememeyi istiyor. Nötr isimli bir örnek de eklendi. JSON Schema'daki `source` ve `target` alanlarının açıklamaları ilişkinin açıklamasını içeriyor.
4. **Birim:** Prompt, sayıları özellik açıklamasının istediği birimde yazmayı istiyor; yüzdeler 0,4 değil 40 olarak yazılıyor.
5. **Testler:** 136 test geçiyor. Ruff, format ve `mypy --strict` temiz.

### Golden set: DeepSeek ile eski ve yeni kod

| Grup | rag | oag | hybrid | rag_eq |
|---|---|---|---|---|
| Tek adım | 91% | 73% | 100% | 100% |
| **Çok adım** | 73% | 100% | **100%** | 91% |
| Serbest metin | 100% | 0% | 100% | 100% |
| **Toplam** | 88% | 58% | **100%** | 97% |

- Değerler cevap doğruluğu. Eski kodda (`9213f62`) ve yeni kodda **aynı**.
- Graf iki koşuda da beklenenle birebir: 13 entity ve 14 ilişki, ORTAGIDIR yönlerinin dördü de doğru, oranlar 60, 35, 51 ve 100.
- Çok adımlı sorularda M2 kapısı ilk kez eşit bütçeli rag'e karşı da geçiyor: hybrid %100, rag_eq %91.
- Ingest 14 doküman için yaklaşık 35–48 saniye sürüyor (gemma ile 8 dakikaydı).

Sonuçları okurken iki şeye dikkat etmek gerekiyor:

- **Düzeltmelerin etkisi bu sette ölçülemiyor.** Düzeltilen üç hata zayıf modelin hatalarıydı; DeepSeek bunları yapmıyor ve faaliyet raporunda bile isimleri ekleriyle tam yazıyor. Düzeltmelerin doğruluğu birim ve entegrasyon testleriyle gösterildi.
- **Golden set bu model için doydu.** Hybrid %100 aldığından sonraki iyileştirmeler bu sette görünmeyecek. Gerçek boyutlu ve daha zor bir korpus artık daha da gerekli.

### Açık kalanlar

- Gerçek boyutlu ve daha zor bir korpusla eşit bütçe karşılaştırması. Bağlama sığmamalı ve isim varyantları, dolaylı anlatımlar içermeli.
- PyPI yayını: hesaplar ve trusted publisher tanımı bekleniyor.
- M3: `Ontology.suggest()`, streaming, değişen dokümanın yeniden ingest'i ve silinmesi, `reembed()`, gözlemlenebilirlik.

---

## 5. Tur: Sentetik Türkçe korpus, zincir isabeti ve retrieval düzeltmeleri

Doymuş golden set'in yerine, bağlama sığmayan ve doğru cevapları kesin olarak bilinen sentetik bir Türkçe korpus kuruldu. Bu korpusta yapılan ölçümler retrieval'da dört yapısal sorunu ve ölçümün kendisinde bir hatayı ortaya çıkardı. Hepsi düzeltildikten sonra, eşit token bütçesinde **hybrid 160 sorunun %98'ini doğru cevaplıyor, rag %32'sini**. Ölçümler `deepseek-v4.1-flash:1017-cloud` (çıkarım ve cevap) ve yerel `nomic-embed-text` (embedding) ile yapıldı.

### Sentetik korpus

`eval/synthetic/generate.py`, LLM kullanmadan ve seed'e bağlı olarak bir "dünya" (şirketler, ortaklık zincirleri, sözleşmeler) üretiyor ve bu dünyadan belgeleri, soruları ve beklenen grafı yazıyor. Varsayılan set (`eval/synthetic/sets/default`, seed 7):

- 596 belge: 150 ticaret sicili kaydı, 86 faaliyet raporu, 300 sözleşme, 60 gürültü belgesi. Toplam 526 bin token.
- Beklenen graf: 450 entity (150 `Sirket`, 300 `Sozlesme`) ve 749 ilişki (149 `ORTAGIDIR`, 600 `TARAFIDIR`).
- 160 soru, dört grupta 40'ar: tek adım, çok adım, üç adım ve serbest metin (ontolojide olmayan sözleşme maddeleri).
- Zorluklar: hukuki eksiz ve kısa isimler, birbirine karıştırılabilen şirket çiftleri ("Meşe Gıda A.Ş." ve "Meşe Gıda Dağıtım"), edilgen ve devrik ortaklık cümleleri, farklı yüzde yazımları, yalnızca faaliyet raporunda geçen ortaklıklar.
- Her sorunun cevap zinciri (`chain`) ve her halkanın hangi belgede hangi cümleyle geçtiği kayıtlı. Üretici, her halkanın sorunun belgelerinde gerçekten yazdığını kendisi doğruluyor.

Runner'a eklenenler: `--set`, `--limit`, ontoloji değişse de çalışan `--fresh`, beklenen grafla karşılaştırılan graf kalitesi tablosu ve `--seed-cache-from`. Sonuncusu başka bir bilgi tabanının çıkarım önbelleğini kopyalıyor; resolver gibi çıkarım sonrası bir değişiklikten sonra yeniden ingest LLM çağrısı yapmadan bitiyor (596 belge, 82 saniye). LLM'siz ayar taramaları için `eval/sweep_retrieval.py` eklendi.

### Ölçüm hatası ve zincir isabeti

Bağlam isabeti, cevap kelimesinin bağlamda herhangi bir yerde geçmesine bakıyordu. Sentetik sette bu yanıltıcı çıktı: 30 şehir var ve kalabalık bir oag bağlamı bunların 18'ini içeriyor; şirket adları da alakasız yollardan bağlama girebiliyor. Bir örnekte cevap olan "Ayaz Mobilya" bağlamda vardı, ama zincirin son halkası (Ayaz Mobilya → Lodos İnşaat) yoktu.

Yeni ölçüm **zincir isabeti**: zincirin her halkası ya graf olgusu olarak (doğru yön, `rol` ve `oran` ile) ya da bağlama giren pasajda kaynak cümlesi olarak bulunmalı (`eval/chain.py`). Eski ölçüm "gevşek isabet" adıyla tablolarda kalıyor. Rag'in çok adımlı sorulardaki %15'lik gevşek isabeti zincir isabetiyle %0 çıktı. Son koşuda zincir isabeti ile cevap doğruluğu neredeyse birebir örtüşüyor; zincir isabeti ücretsiz ve güvenilir bir ön ölçüm olarak kullanılabiliyor.

### Düzeltilenler

1. **Adım adım graf genişletme.** Tek ve değişken uzunluklu sorgu yerine her adımda bir sorgu çalışıyor. Her düğüm, ilişki tipi ve yön için en fazla `max_fanout=8` kenar alınıyor; 30 sözleşmesi olan bir şirkette ortaklık kenarları sözleşmelerin arasında kaybolmuyor. Olgular yürüyüş sırasında diziliyor ve her yeni entity'nin özellikleri ona ulaşan ilişkinin hemen arkasına geliyor; bütçe dolduğunda en uzak olgular düşüyor. `max_paths` kaldırıldı, `max_facts` 120 oldu.
2. **Birebir eşleşmeyle linking.** Soruda bir anahtar (`DS-2023-008`, vergi numarası) ya da tam bir şirket adı aynen geçiyorsa yalnızca o entity tohum oluyor. Daha önce tam metin araması numarayı parçalarına ayırıp benzer numaralı beş sözleşmeyi daha getiriyordu ve olgu bütçesinin çoğu yanlış tohumların komşularına gidiyordu. "Lale Cam Yatırım Grubu" yazıldığında adı onun içinde kalan "Lale Cam" ayrıca tohum olmuyor. `link_exact_anchors=False` eski davranışı veriyor.
3. **Resolver'da isim koruması.** Grafta "eksik" görünen 22 ilişkinin hepsi aslında doğru çıkarılmış, ama benzer isimli başka bir şirkete bağlanmıştı (Kemer Ambalaj → Bahar Ambalaj, Zeytin Yapı → Zeytin Yazılım). Sebep: sözleşmeler tarafları vergi numarasız yazıyor ve alfabetik yüklemede sözleşmeler sicil kayıtlarından önce geliyor. Bu durumda resolver ismi embedding benzerliğiyle en yakın şirkete bağlıyordu; isimlerin ortak sektör kelimesi benzerliği 0,92 eşiğinin üstüne taşıyordu. Embedding ile birleştirme artık yalnızca isimlerin kelime kümeleri birbirini kapsıyorsa yapılıyor: "Zirve" ile "Zirve Gıda A.Ş." birleşiyor, "Kemer Ambalaj" ile "Bahar Ambalaj" birleşmiyor.
4. **`hops` varsayılanı 3.** Tek tohumla, sözleşme üzerinden giden sorular (şirket → sözleşme → şirket → ortak) üç kenar gerektiriyor. Golden set'te oag'ın çok adımlı isabeti `hops=2` ile %64, `hops=3` ile %100.
5. **Hybrid'de tohum belgesi pasajları ve rezerv.** Serbest metin soruları için 300 sözleşme aynı madde kalıplarını kullanıyor; korpus genelindeki aramada cevap pasajı 40 sorunun yalnızca 8'inde ilk iki sonuç arasındaydı. Arama tohum entity'nin geçtiği belgelerle sınırlanınca bu 40/40 oluyor. Hybrid artık bu pasajları üçüncü bir sıralama listesi olarak füzyona katıyor. `hybrid_text_reserve=4` ile tohum belgesi ve rag sonuçlarından sırayla ilk dördü, olgulardan önce bağlama giriyor.
6. **Bütçe muhasebesi.** Bölüm başlıkları ve ayraçlar bütçeye sayılıyor; üç modda da bağlam `context_token_budget`'ı aşmıyor.
7. **Testler:** 197 test geçiyor. Ruff, format ve `mypy --strict` temiz.

### Sonuçlar: tam sentetik set (160 soru)

Değerler zincir isabeti / cevap doğruluğu. Bütçe 6.000 token; ortalama bağlam hybrid'de 5.739, rag_eq'da 5.746 token.

| Grup | rag | oag | hybrid | rag_eq |
|---|---|---|---|---|
| Tek adım | 75 / 75 | 100 / 100 | **100 / 100** | 80 / 80 |
| Çok adım | 0 / 2 | 100 / 92 | **100 / 100** | 5 / 5 |
| Üç adım | 0 / 0 | 98 / 92 | **92 / 92** | 0 / 0 |
| Serbest metin | 50 / 50 | 0 / 0 | **100 / 100** | 45 / 45 |
| **Genel** | 31 / 32 | 74 / 71 | **98 / 98** | 32 / 32 |

- Rag çok adımlı sorularda bütçe artırılsa da zinciri getiremiyor (rag_eq %0–5).
- Oag'ın bağlam isabeti %98–100 iken cevap doğruluğu %92'de kalıyor: yalnızca olgu satırlarıyla model bazen yanılıyor, hybrid'deki kaynak metin bu boşluğu kapatıyor.
- Üç adımda hybrid'in kalan %8'i bütçe sınırından geliyor; 10.000 token bütçede %98.
- Golden set yeni varsayılanlarla bozulmadı: hybrid her grupta %100.

Hybrid'in zincir isabetinin düzeltmelerle değişimi (6.000 token):

| Aşama | Çok adım | Üç adım | Serbest metin |
|---|---|---|---|
| Eski linking, `hops=2` | 90 | 25 | 32 |
| Birebir eşleşme, `hops=2` | 90 | 60 | 78 |
| Birebir eşleşme, `hops=3` | 90 | 82 | 48 |
| Resolver koruması (yeniden ingest) | 100 | 92 | 42 |
| Tohum belgesi pasajları, rezerv 4 | **100** | **92** | **100** |

### Graf kalitesi

| | Önce | Resolver korumasından sonra |
|---|---|---|
| Entity'ler | 450/450, mükerrer yok | 450/450, mükerrer yok |
| `ORTAGIDIR` hassasiyet / geri çağırma | %95,3 / %96,0 | **%100 / %100** |
| `TARAFIDIR` hassasiyet / geri çağırma | – / %97,3 | **%99,5 / %99,5** |
| `oran` doğruluğu | %100 | %100 |
| `rol` doğruluğu | %93,7 | %93,6 |

### Açık kalanlar

- **Resolver'da yükleme sırası.** Kısa bir ad başka bir şirketin adının içinde kalıyorsa ("Meşe Gıda" ve "Meşe Gıda Dağıtım") ve doğru şirket henüz grafta yoksa yanlış birleşme olabiliyor; kalan 3 `TARAFIDIR` hatası bu. Kalıcı çözüm, yüklemeden sonra çalışan bir yeniden çözümleme adımı.
- **`rol` doğruluğu %94.** Hangi sözleşme türlerinde ve hangi cümle kalıplarında yanıldığı incelenmedi.
- **Anahtar araması** bir tipin tüm düğümlerini tarıyor; büyük graflarda indeksli sorguya çevrilmeli.
- **Sentetik setin sınırları.** Tek bir korpus, tek bir alan ve tek bir model. Sonuçlar gerçek belgelerle doğrulanmalı. Aynı hedefe birden fazla yoldan ulaşılan sorularda üretici yalnızca bir yolu kaydediyor.
- `eval/synthetic/sets/default/README.md` hâlâ `--hops 3` öneriyor; artık varsayılan bu.
- PyPI yayını ve M3 (önceki turdan).

## 6. Tur: Görülmemiş set ve güçlü RAG ile karşılaştırma

5. turdaki sonuç (hybrid %98, RAG %32) iki açıdan itiraza açıktı. Ayarlar aynı 160 soruya bakılarak yapılmıştı ve karşılaştırılan RAG zayıftı: yalnızca vektör araması vardı, İngilizce ağırlıklı bir embedder kullanıyordu, anahtar sözcük araması ve reranker yoktu. Bu tur ikisini de kapattı.

### Görülmemiş (held-out) set

Üreticiye `--variant b` eklendi. Bu varyantta olgu cümleleri, belge iskeletleri, standart maddeler ve soruların hepsi ayrı kalıp havuzlarından geliyor. Ontoloji, dünya modeli ve soru aileleri aynı kaldı. `eval/synthetic/sets/heldout` seed 101 ile üretildi: 593 belge, 450 entity, 749 ilişki, 160 soru. Varsayılan setle ortak cümle sayısı 13.243'te 0. Varsayılan set bayt bayt aynı kaldı.

### Güçlü RAG

- **Çok dilli embedder:** `bge-m3`.
- **BM25.** İlk denemede FalkorDB'nin tam metin indeksi kullanıldı ve RAG'i iyileştireceğine kötüleştirdi (tek adımlı sorularda %65 → %15). Bu indeksin puanı OR sorgularında terimleri nadirliğe göre toplamıyor. Soruda "vergi kimlik numarası sicil" gibi yaygın sözcükler olunca 150 sicil kaydının hepsi aynı puanı alıyordu ve şirket adı hiçbir fark yaratmıyordu. Yerine süreç içinde çalışan bir BM25 (Okapi) indeksi yazıldı. Türkçe sözcükler katlanıp 6 harfe kırpılıyor, `KS-2024-002` gibi anahtarlar tek parça terim oluyor. RRF'de metin ve vektör tarafına eşit ağırlık veriliyor. Tek başına BM25, tek adımlı sorularda doğru belgeyi 40 sorunun 37'sinde 1. sıraya koyuyor.
- **Reranker:** `jinaai/jina-reranker-v2-base-multilingual` (isteğe bağlı).

### Sonuçlar

Bağlam taraması (zincir isabeti; tek adım / çok adım / derin / serbest metin). "En iyi RAG" sütununda her grupta rag ile rag_eq'den yüksek olanı yazıyor.

| Set | RAG ayarı | En iyi RAG | Hybrid |
|---|---|---|---|
| Varsayılan | `nomic`, yalnız vektör | 65 / 0 / 0 / 48 | 100 / 100 / 92 / 100 |
| Varsayılan | `bge-m3`, yalnız vektör | 100 / 5 / 3 / 28 | 100 / 98 / 82 / 100 |
| Varsayılan | `bge-m3` + BM25 | 100 / 2 / 0 / 85 | 100 / 98 / 82 / 100 |
| Varsayılan | `bge-m3` + BM25 + reranker | 100 / 2 / 0 / 62 | 100 / 98 / 82 / 100 |
| Held-out | `nomic` + BM25 | 98 / 12 / 2 / 68 | 100 / 100 / 80 / 100 |
| Held-out | `bge-m3` + BM25 | 100 / 18 / 0 / 57 | 100 / 98 / 85 / 100 |

Held-out set, cevap doğruluğu (`bge-m3` + BM25, DeepSeek v4.1 flash):

| Grup | rag | rag_eq | hybrid |
|---|---|---|---|
| Tek adım | 100 | 100 | 100 |
| Çok adım | 8 | 18 | 98 |
| Derin | 0 | 0 | 78 |
| Serbest metin | 58 | 40 | 100 |
| **Tümü** | **41** | **39** | **94** |

Golden sette (14 belge, 33 soru) BM25 ile RAG'in bağlam isabeti tek adımlı sorularda 91'den 100'e, çok adımlı sorularda 55'ten 82'ye çıktı. Hybrid her grupta %100'de kaldı.

### Çıkarımlar

- **Ezberleme yok.** Hybrid görülmemiş sette de %94–96.
- **RAG'i güçlendirmek farkı kapatmıyor.** BM25 serbest metin sorularında belirgin artış sağlıyor. Çok ve derin adımlı sorular ise ~%0–18'de kalıyor, çünkü cevap birden çok belgeye dağılmış bir zincir.
- **BM25 artık varsayılan** (`rag_lexical=True`). Reranker bu veride serbest metni 85'ten 62'ye düşürdü ve CPU'da çok yavaş; isteğe bağlı kaldı.
- **Derin sorularda cevap kaybı.** Bağlam %85 doğru, cevap %78. Aradaki fark modelin üç halkalı zinciri izlerken yaptığı hata.

### Açık kalanlar

- **rag_eq, serbest metinde rag'den düşük** (58'e karşı 40). Büyük olasılıkla `k` arttıkça fazladan gelen pasajlar, en iyi pasajın cevabı taşıyan komşularını bütçeden itiyor. Bu doğrulanmadı.
- **`bge-m3` ile resolver eşiği.** 0,92 eşiği `nomic`'e göre ayarlanmıştı. `bge-m3` ile `ORTAGIDIR` hassasiyeti %92,4'e düştü.
- **BM25 bellekte tutuluyor** (parça başına bir token listesi). Büyük korpuslarda ölçülmeli.

### Belge güncelleme ve silme

- **`kb.delete(kaynak | belge id)`**, CLI'da `rag-with-oag delete`:
  - Belgenin parçaları ve kendisi silinir.
  - İlişkilerin kanıt listesinden bu belgenin parçaları çıkarılır; kanıtı kalmayan ilişki silinir.
  - Artık hiçbir yerde geçmeyen ve ilişkisi kalmayan entity'ler silinir.
  - BM25 indeksi yenilenir.
  - Sonuç bir `DeleteReport` olarak döner.
- **Değiştirme:** aynı kaynak yolu yeni içerikle gelirse önce yeni sürüm yazılıp çıkarılır, eski sürüm bütün çalıştırma bittikten sonra silinir. Yeni sürüm başarısız olursa eski sürüm kalır. `replace=False` (CLI'da `--no-replace`) eski davranışı korur.
- **Yazma kilidi:** ingest ve delete birbirini bekler; sorgular etkilenmez.
- **Sınır:** entity özellikleri geri alınmıyor, çünkü hangi özelliğin hangi belgeden geldiği kaydedilmiyor.
- **Ölçeklenme:** silme sırasında ilişki taraması bütün ilişkileri geziyor; büyük graflarda ölçülmeli.
- **Doğrulama:** 223 test geçti. Gerçek embedder ile CLI denemesinde düzenlenen dosyanın eski değeri bilgi tabanından çıktı, silinen belge aramadan kayboldu.
