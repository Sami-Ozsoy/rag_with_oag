"""Template types and the original Turkish phrasing pools."""

from dataclasses import dataclass

type Section = tuple[str, str]


@dataclass(frozen=True)
class ClauseText:
    heading: str
    statements: tuple[str, ...]
    question: str


@dataclass(frozen=True)
class TemplateSet:
    contract_title: str
    signature_heading: str
    party_paragraphs: tuple[str, ...]
    signature_after: int
    clause_slots: tuple[int, ...]
    clauses: dict[str, ClauseText]
    boilerplate: tuple[Section, ...]
    contract_notes: tuple[str, ...]
    registry_title: str
    registry_record_position: int
    registry_records: tuple[str, ...]
    registry_name: str
    ownership_passages: tuple[str, ...]
    ownership_list: str
    ownership_entry: str
    ownership_separator: str
    report_title: str
    finance_heading: str
    finance_position: int
    finance_notes: tuple[str, ...]
    report_sections: tuple[Section, ...]
    report_notes: tuple[str, ...]
    portfolio_heading: str
    meetings_heading: str
    meeting_notes: tuple[str, ...]
    noise_title: str
    noise: tuple[Section, ...]
    questions: dict[str, tuple[str, ...]]
    free_text_question: str


CONTRACT_BOILERPLATE = (
    (
        "Uyuşmazlıkların çözümü",
        "Bir uyuşmazlık doğduğunda yetkili temsilciler önce ortak bir görüşme düzenler. "
        "Tarafların sunduğu belgeler ve üzerinde uzlaşılan konular görüşme tutanağında ayrı ayrı "
        "gösterilir. Görüşme sürerken uyuşmazlıkla bağlantısı bulunmayan işler "
        "yürütülmeye devam eder.",
    ),
    (
        "Tebligat usulü",
        "Resmî bildirimler teslim alındığı doğrulanabilen yazılı bir kanaldan yapılır. "
        "Teslim kaydında bildirimin konusu ile gönderim ve kabul bilgileri birlikte tutulur. "
        "Günlük iş akışında kullanılan mesajlaşma kanalları yalnızca çalışma "
        "koordinasyonu içindir.",
    ),
    (
        "Damga vergisi",
        "Damga vergisi ve benzeri mali yükümlülükler yürürlükteki mevzuat uyarınca "
        "yerine getirilir. "
        "İşlem makbuzlarının bir örneği mali kayıtlar arasında saklanır. "
        "Sonradan düzenlenen ek metinler için doğabilecek yükümlülükler ayrıca değerlendirilir.",
    ),
    (
        "Mücbir sebep",
        "Tarafların makul denetimi dışındaki bir olay edimi geçici olarak engellerse olayın etkisi "
        "ve alınan önlemler yazılı olarak bildirilir. Engelin ortadan kalktığı anlaşılınca "
        "çalışmaların yeniden başlama sırası üzerinde birlikte mutabakata varılır.",
    ),
    (
        "Devir yasağı",
        "Sözleşmeden doğan hak ve yükümlülükler yazılı onay alınmadan üçüncü "
        "kişilere devredilemez. "
        "Devir talebinde devredilmek istenen işlerin kapsamı açıkça belirtilir. Başvuru "
        "değerlendirilirken mevcut iş planı yürürlükte kalır.",
    ),
    (
        "Kişisel verilerin korunması",
        "İşlenmesi gereken kişisel veriler ilgili işin gerektirdiği alanlarla sınırlı tutulur. "
        "Erişim yetkileri görev esasına göre verilir ve görev değişikliklerinde gözden geçirilir. "
        "Silme ve anonimleştirme işlemleri kayıt altına alınır.",
    ),
    (
        "Belge sırası",
        "İmzalı metin, ekler ve sonradan onaylanan değişiklikler aynı dosyada saklanır. "
        "Birbiriyle çelişen iki kayıt ortaya çıktığında her belgenin yürürlük kapsamı incelenir. "
        "Taslak sürümler onaylı belgelerden ayrı bir klasörde tutulur.",
    ),
    (
        "İş güvenliği",
        "Sahaya giren ekipler çalışma alanına ilişkin iş güvenliği kurallarını önceden öğrenir. "
        "Tehlikeli görülen bir işlem durdurulur ve risk değerlendirmesi "
        "tamamlandıktan sonra yeniden "
        "başlatılır. Kayıtlar görevli ekipler tarafından birlikte incelenebilir.",
    ),
    (
        "Alt yükleniciler",
        "Dış kaynak kullanılması hâlinde yapılacak iş ve denetim yöntemleri önceden yazılı olarak "
        "belirlenir. İşin bir bölümünün alt yükleniciye bırakılması asıl edimin "
        "takibini değiştirmez. "
        "Saha erişimi için gerekli bilgiler ilgili ekibe iletilir.",
    ),
    (
        "Kalite kontrolü",
        "Teslim edilen çıktılar, önceden paylaşılan kabul ölçütlerine göre değerlendirilir. "
        "Ölçütlerden sapma görülürse bulgu ile düzeltme adımı aynı tutanağa yazılır. "
        "İncelemeyi yapan kişi yeni teslimin hangi sürümle karşılaştırıldığını belirtir.",
    ),
    (
        "İş planı",
        "Uygulama sırası, işleri devralacak kişileri de kapsayan ortak bir çalışma "
        "planında gösterilir. "
        "Güncellemeler yapıldığında eski plan arşive kaldırılır ve ilgililere yeni plan dağıtılır. "
        "Açık konular bir sonraki toplantının gündemine alınır.",
    ),
    (
        "Kayıt doğrulama",
        "İşlem kayıtlarının kaynağı ve kayıt üzerinde yapılan düzeltmeler izlenebilir olmalıdır. "
        "Hatalı kayıtlar silinmeden önce düzeltme gerekçesi yazılır. Denetim sırasında "
        "istenen kayıtlar, değişiklik geçmişiyle birlikte incelemeye açılır.",
    ),
    (
        "Kapsam değişikliği",
        "İş kapsamını etkileyen talepler yazılı biçimde değerlendirilir. Talep metninde işin "
        "gerekçesi, beklenen çıktı ve mevcut plan üzerindeki etkiler açıklanır. Karar verilene "
        "kadar ekipler onaylanmış görev listesi üzerinden çalışır.",
    ),
    (
        "Bağımsız hükümler",
        "Bir hükmün uygulanamaması diğer hükümlerin yürürlüğünü etkilemez. Taraflar, uygulamada "
        "boşluk doğduğunda asıl düzenlemenin amacına uygun yeni bir metin üzerinde görüşür. "
        "Bu görüşmenin sonucu dosyaya yazılı olarak eklenir.",
    ),
    (
        "Toplantı tutanakları",
        "Çalışma toplantılarında alınan kararlar tutanakta, açık kalan konular ise "
        "izleme listesinde "
        "yer alır. Katılımcılar kendilerine iletilen tutanakta maddi hata görürse düzeltme ister. "
        "Onaylanan tutanak çalışma dosyasına eklenir.",
    ),
    (
        "Erişim denetimi",
        "Ortak çalışma alanlarına erişim yalnızca görevi gereği bu alanları kullanan kişilere "
        "tanımlanır. Yetkilerin atanması ve kaldırılması takip kayıtlarında gösterilir. "
        "Kullanılmayan erişim bilgileri düzenli arşiv incelemesinde kaldırılır.",
    ),
    (
        "Saha kabulü",
        "Sahaya teslim edilen malzeme veya kayıtlar ilgili görevli ile birlikte kontrol edilir. "
        "Teslimde aksaklık belirlenirse bulgu, etkilediği iş adımıyla birlikte yazılır. "
        "Kabul kaydı açık bulgular kapatıldığında tamamlanır.",
    ),
    (
        "Çalışma dili",
        "Ortak işlem kayıtları ve kabul belgeleri Türkçe hazırlanır. Başka dilde iletilen teknik "
        "ekler gerektiğinde açıklayıcı bir özetle birlikte dosyalanır. Çeviride tereddüt olduğunda "
        "ilgili teknik ekip belgeyi birlikte inceler.",
    ),
    (
        "Arşiv düzeni",
        "İşin yürütülmesinde kullanılan belgeler konu ve sürüm bilgileriyle sınıflandırılır. "
        "Tekrarlanan kopyalar ayıklanırken işlemin kaynağını gösteren belgeler korunur. "
        "Arşiv dizinindeki kayıtlar her teslimden sonra güncellenir.",
    ),
    (
        "Yetkili temsilciler",
        "Temsilcilerin görev kapsamı ilgili iş akışına katılan kişilere yazılı olarak duyurulur. "
        "Temsilci değişiklikleri açık işlerin devri tamamlandığında kayıt altına alınır. "
        "Yeni temsilci önceki kararların yer aldığı çalışma dosyasını teslim alır.",
    ),
    (
        "İş sürekliliği",
        "İşin kesintiye uğrama riski değerlendirildiğinde alternatif çalışma adımları hazırlanır. "
        "Bu adımları devreye alacak görevli ile kullanılacak kayıtlar önceden belirlenir. "
        "Kesinti giderildiğinde alınan önlemlerin sonuçları raporlanır.",
    ),
    (
        "Fikrî haklar",
        "Çalışma sırasında kullanılan önceden mevcut araçlar ve yöntemler ilgili "
        "hak sahibinde kalır. "
        "Teslim edilen çıktının kullanım koşulları teknik ekiyle birlikte değerlendirilir. "
        "Yeni üretilen içerik hakkında tereddüt doğarsa kayıtlar karşılıklı incelenir.",
    ),
    (
        "Bilgi güvenliği",
        "Ortak dosyaların hangi kanaldan paylaşılacağı ve kimlere açılacağı önceden belirlenir. "
        "Bir güvenlik olayı tespit edildiğinde olayın etkilediği kayıtlar envanterden belirlenir. "
        "Olay incelemesi tamamlandığında düzeltici işlemler raporlanır.",
    ),
    (
        "Feragat",
        "Bir hakkın belli bir olayda kullanılmamış olması o haktan sürekli vazgeçildiği sonucunu "
        "doğurmaz. Sonraki işlemler uygulanabilir hükümler uyarınca ele alınır. "
        "Yazılı mutabakatlar ilgili iş kaydına bağlanır.",
    ),
    (
        "Teslim belgeleri",
        "Her teslimde dosya içeriği ile değişiklik açıklamalarını gösteren bir "
        "envanter paylaşılır. "
        "Eksik kalemler saptanırsa teslim belgesinde görünür biçimde işaretlenir. "
        "Envanter ancak eksikler giderildikten sonra kesinleştirilir.",
    ),
    (
        "Denetim hazırlığı",
        "Denetim için istenen kayıtların kapsamı ve erişim yöntemleri önceden belirlenir. "
        "İnceleme sırasında alınan notlar dayanak belgelerle birlikte tutulur. "
        "Denetim kapanışında açık bulgular için sorumlu kişiler ve izlenecek adımlar açıklanır.",
    ),
    (
        "Ekipman ve araçlar",
        "Çalışma alanında kullanılan ekipmanın uygunluğu iş başlamadan önce kontrol edilir. "
        "Kontrolde ortaya çıkan bir eksik, kullanımı etkilemesi hâlinde iş planında belirtilir. "
        "Ekipman değişikliği teslim ve kabul kayıtlarıyla izlenir.",
    ),
    (
        "Çevresel uygulamalar",
        "İş sırasında oluşan atıklar ilgili çalışma alanının uygulamalarına göre ayrılır. "
        "Atığın nasıl değerlendirildiğini gösteren kayıtlar görevli ekip tarafından saklanır. "
        "Kaynak kullanımını azaltmak üzere tespit edilen öneriler dönem sonunda gözden geçirilir.",
    ),
    (
        "Düzenleyici değişiklikler",
        "Yürürlükteki kurallarda değişiklik olduğunda ilgili çalışma adımları yeniden incelenir. "
        "Değişiklikten etkilenen belgeler bir listede toplanır ve güncelleme önceliği belirlenir. "
        "Geçiş sırasında uygulanan yöntemler izleme kayıtlarına eklenir.",
    ),
    (
        "Kapanış işlemleri",
        "İş kapanırken açık işlemler, teslim alınmış kayıtlar ve iade edilecek araçlar birlikte "
        "gözden geçirilir. Kapanış dosyasında her kaleme ait son durum görünür olmalıdır. "
        "Kapanış kaydı tarafların yetkili ekiplerine iletilir.",
    ),
)

CONTRACT_NOTES = (
    "Uygulamaya ilişkin görüş ve kayıtlar ilgili madde başlığı altında dosyalanır.",
    "Bu başlık altındaki işlemler çalışma dosyasındaki son onaylı açıklamaya göre izlenir.",
    "Gerektiğinde bu hükmün uygulamasına ait belgeler ortak incelemeye sunulur.",
)

REPORT_SECTIONS = (
    (
        "Yönetim değerlendirmesi",
        "Yönetim dönem boyunca üretim ile satış planlarını birlikte değerlendirdi. Aylık izleme "
        "toplantılarında planlanan işler, tamamlanan işler ve gelecek döneme aktarılan başlıklar "
        "ayrı ayrı incelendi. Teslimde yaşanan aksamalar için bölümlerden uygulama sırasında "
        "tutulan kayıtlar istendi. Gündeme taşınan bulgular, iş programı güncellenmeden önce "
        "ilgili birimlerle görüşüldü. Bir sonraki değerlendirmeye kadar izlenecek maddeler ortak "
        "çalışma dosyasına işlendi.",
    ),
    (
        "Yatırım programı",
        "Yatırım programında bakım ihtiyaçları, yeni araç alımları ve mevcut "
        "donanımın iyileştirilmesi "
        "farklı başlıklar altında izlendi. Birimler önerdikleri çalışmaların tahmini kullanım "
        "ömrünü ve beklenen etkisini yazılı olarak bildirdi. İlk değerlendirmede üretim akışını "
        "durdurmadan gerçekleştirilebilecek işler öne alındı. Alım ve kurulum aşamaları arasında "
        "sahaya ayrılan zaman yeniden planlandı. Yılın sonunda teslim alınan araçlar envanter "
        "kayıtlarıyla karşılaştırıldı.",
    ),
    (
        "Nakit ve işletme sermayesi",
        "Nakit planlaması hazırlanırken tahsilat takvimi, stokların dönüş hızı ve düzenli ödeme "
        "kalemleri birlikte ele alındı. Dönem içindeki değişiklikler tahmin dosyasının ilgili "
        "bölümlerine kaydedildi. Olağan dışı bir hareket görüldüğünde mali işler birimi kaynağını "
        "inceledi ve izleme toplantısında sonucunu paylaştı. Yeni dönem için kullanılan "
        "varsayımlar yıl sonunda gözden geçirildi. Bu çalışma, bütçe planında beklenmeyen "
        "ödemeler için ayrılan payın belirlenmesine de temel oluşturdu.",
    ),
    (
        "Sürdürülebilirlik",
        "Kaynak kullanımı değerlendirilirken bakım gerektiren donanım ile işletme biçiminden doğan "
        "kayıplar ayrı incelendi. Çalışma alanlarından gelen öneriler doğrulanabilir ölçümlerle "
        "desteklendiğinde izleme listesine alındı. Atık ayrıştırma uygulamalarına "
        "ilişkin gözlemler "
        "saha toplantılarında paylaşıldı. Su ve elektrik tüketimi kayıtlarının dönemler arasında "
        "karşılaştırılabilmesi için ölçüm aralıkları düzenlendi. Yılsonu değerlendirmesinde açık "
        "kalan çalışmaların sonraki bakım programına alınması kararlaştırıldı.",
    ),
    (
        "İnsan kaynakları",
        "Personel planı hazırlanırken görevlerin yedeklenmesi, eğitim talepleri ve "
        "iş programındaki "
        "dönemsel yoğunluklar ele alındı. Yeni başlayanlar için kullanılan bilgilendirme içeriği "
        "saha ekiplerinin görüşleriyle güncellendi. Birimler, toplantılarda paylaşılan örneklerden "
        "hareketle uygulamalı eğitim başlıklarını belirledi. Eğitim sonrasında ekiplerin geri "
        "bildirimleri derlendi ve bir sonraki programda yeniden ele alınacak konular işaretlendi. "
        "Görev tanımları üzerindeki düzeltmeler de aynı dönemde tamamlandı.",
    ),
    (
        "Bilgi sistemleri",
        "Dijital kayıt çalışmasında önce iş akışlarında hangi verilerin kullanıldığı belirlendi. "
        "Ardından farklı birimlerin aynı kayıt için kullandığı alanlar karşılaştırıldı. Bilgi "
        "kalitesini etkileyen boş alanlar ve tutarsız tarih kayıtları bir çalışma "
        "listesine alındı. "
        "Erişim yetkileri görev tanımlarındaki güncellemelerle birlikte yeniden incelendi. "
        "Yedekleme denemeleri sırasında geri yüklenen kayıtlar örneklem yöntemiyle kontrol edildi. "
        "Tekrarlanan aksaklıklar için işlem adımları yeniden yazıldı.",
    ),
    (
        "Piyasa koşulları",
        "Döneme ilişkin piyasa değerlendirmesinde girdi fiyatları ile teslim "
        "sürelerindeki değişiklikler "
        "ele alındı. Varsayımlar hazırlanırken birimlerin saha gözlemleri ve düzenli fiyat "
        "izleme tabloları kullanıldı. Tek seferlik dalgalanmaların yıllık plana etkisi ayrıca "
        "değerlendirildi. İş hacmindeki mevsimsel değişiklikler için alternatif "
        "çalışma programları "
        "hazırlandı. Raporlama döneminin sonunda daha önce kullanılan tahminlerle gerçekleşen "
        "sonuçlar karşılaştırıldı ve farkların nedenleri kayda geçirildi.",
    ),
    (
        "Stok yönetimi",
        "Sayım programı, iş akışının mümkün olduğunca kesintisiz sürmesini sağlayacak biçimde "
        "düzenlendi. Fiziksel sayım ile kayıtlar arasında fark saptanan kalemler yeniden "
        "kontrol edildi. Hareket görmeyen ürünler ayrı bir listede incelendi ve kullanım "
        "önerileri birimlerden toplandı. Depolama koşullarından kaynaklanan riskler saha "
        "kontrollerine eklendi. Sonraki dönemde başka sıklıkla sayılması gereken ürün grupları "
        "kapanış toplantısında belirlendi.",
    ),
    (
        "İç kontrol",
        "İç kontrol çalışmaları öncelikle kritik kayıtların kaynağını ve onay sırasını kapsadı. "
        "İncelenen işlemler için belirlenen bulgular ilgili birimlerin görüşüne açıldı. Aynı "
        "işlem türünde birden fazla bulgu görülmesi hâlinde ortak nedenler araştırıldı. "
        "İyileştirme adımları, tamamlanma durumu izlenebilecek biçimde kayda geçirildi. "
        "Bir sonraki incelemede hangi örneklerin yeniden değerlendirileceği programlanarak "
        "açık bulgulara ait evrak erişilebilir biçimde arşivlendi.",
    ),
    (
        "Hizmet kalitesi",
        "İletilen geri bildirimler konu, işlem aşaması ve çözüm sürecine göre sınıflandırıldı. "
        "İşin tekrar yapılmasına neden olan başlıklar, günlük toplantılarda somut örnekler "
        "üzerinden ele alındı. Düzeltici uygulamalar önce sınırlı bir çalışma grubunda denendi. "
        "Saha gözlemleri yeterli olduğunda uygulama adımları güncellendi ve birimlere dağıtıldı. "
        "Aynı sorunun yeniden ortaya çıkıp çıkmadığı sonraki değerlendirme kayıtlarında izlendi.",
    ),
    (
        "Bakım ve kapasite",
        "Bakım programında arıza geçmişi ile iş yükünün dönem içindeki dağılımı "
        "birlikte ele alındı. "
        "Planlı duruşlar sırasında yapılan işlemler ekiplerin kullanabileceği ortak bir kayıt "
        "dosyasına işlendi. Daha uzun inceleme gerektiren araçlar ayrı bir gruba alındı. Yedek "
        "araçların nerede kullanılabileceği gözden geçirildi. Yapılan düzenlemelerin kapasite "
        "üzerindeki etkisi, izleyen haftaların üretim ve hazırlık tablolarında takip edildi.",
    ),
    (
        "Risklerin izlenmesi",
        "Risk değerlendirmesinde her başlık için gözlenebilecek erken belirtiler kaydedildi. "
        "Bir alandaki aksamaların diğer işlere etkisini görmek için farklı çalışma senaryoları "
        "incelendi. Açık kalan önlemler, sorumlu birimler ve mevcut kayıtların durumu üzerinden "
        "izlendi. Yeni bildirilen riskler önceliklendirme toplantısında ele alındı. Önceki "
        "dönemden devralınan başlıklar için alınmış önlemlerin uygulanıp "
        "uygulanmadığı ayrıca kontrol edildi.",
    ),
    (
        "Raporlama yöntemi",
        "Dönem tabloları hazırlanırken birimlerden gelen verilerin kapsadığı tarih aralıkları "
        "karşılaştırıldı. Aynı hareketin iki kez sayılmasını önlemek için kayıt kaynakları "
        "ayrıştırıldı. Sınıflandırma değişikliklerinin önceki dönemle karşılaştırmaya olan etkisi "
        "çalışma notlarına eklendi. Yıl sonu değerlendirmesi tamamlandığında dayanak tablolar "
        "bölüm dosyalarıyla birlikte arşivlendi. İzleyen dönemde kullanılacak veri "
        "toplama şablonuna "
        "inceleme sırasında belirlenen yeni kontrol alanları işlendi.",
    ),
    (
        "İş sağlığı",
        "Saha kontrollerinde ekiplerden çalışma alanındaki değişiklikleri bildirmeleri istendi. "
        "Risk arz eden bir durum tespit edildiğinde işlem adımı ve kullanılan araçlar kayda "
        "alındı. Benzer alanlarda görev yapan çalışanlarla sonuçlar paylaşıldı. İyileştirme "
        "gerektiren başlıklar eğitim ve denetim planına eklendi. Dönem kapanışında tamamlanan "
        "çalışmalar örnek kayıtlar üzerinden gözden geçirildi ve açık kalan "
        "konulara izleme görevi verildi.",
    ),
    (
        "Gelecek dönem planı",
        "Gelecek dönem planı hazırlanırken mevcut işlerin tamamlanma durumu ve "
        "birimlerin ayırabileceği "
        "kaynaklar karşılaştırıldı. Önceliklerin hızla değişebileceği alanlar için alternatif "
        "başlangıç tarihleri değerlendirildi. Maliyet öngörüleri önceki gerçekleşmelerle "
        "birlikte incelendi. Programın izlenmesinde kullanılacak göstergeler sorumlu ekiplerle "
        "tartışıldı. Yönetim, ilerleme tablolarının daha kısa aralıklarla paylaşılması ve "
        "gerekçeli değişikliklerin aynı belgede gösterilmesi üzerinde uzlaştı.",
    ),
    (
        "Operasyon değerlendirmesi",
        "Operasyon kayıtları incelenirken iş hazırlığı ve uygulama aşamalarında geçen süreler "
        "ayrı ayrı ele alındı. Hacimdeki değişiklikler nedeniyle farklılaşan adımlar saha "
        "gözlemleriyle birlikte değerlendirildi. İyileştirme önerilerinin uygulanabilirliği "
        "ekip toplantılarında tartışıldı. Tamamlanan denemeler için sonuç tabloları oluşturuldu. "
        "Gelecek dönem programı hazırlanırken kalıcı hâle getirilecek uygulamalar ve yeni "
        "denemelere ayrılacak çalışma alanları bu tablolara dayanılarak belirlendi.",
    ),
)

REPORT_NOTES = (
    "Birimlerin görüşlerine çalışma tutanaklarında yer verildi.",
    "Konuyla ilgili sayım ve inceleme notları rapor dosyasına eklendi.",
    "Son kontrollerin sonuçları bir sonraki dönemin programına aktarıldı.",
)

NOISE = (
    (
        "Duyuru",
        "{name} ekibinin fotoğraf sergisi için ziyaretçi kaydı açıldı. "
        "Katılımcılar seçkide kullanılacak görselleri etkinlik dosyasından inceleyebilir.",
    ),
    (
        "Haber",
        "{name} adına düzenlenen okuma gününde gönüllüler çocuk kitaplarını seslendirdi. "
        "Katılımcıların hazırladığı çizimler etkinliğin kapanışında sergilendi.",
    ),
    (
        "Toplantı notu",
        "{name} gönüllüleri, kurum içi bahar buluşmasının oturma düzenini görüştü. "
        "Sahneye yakın sıralar prova için ayrıldı. İsim kartlarının basımı bir "
        "sonraki gündeme kaldı.",
    ),
    (
        "Duyuru",
        "{name} çalışanları için düzenlenen şiir dinletisinin programı paylaşıldı. "
        "Dinleyiciler etkinlik sonrasında programa ilişkin görüşlerini yazılı olarak iletebilir.",
    ),
    (
        "Haber",
        "{name} ekibinin hazırladığı kısa film gösterimi tamamlandı. "
        "Gösterimin ardından yönetmenle yapılan söyleşide çekim sürecinden görüntüler paylaşıldı.",
    ),
    (
        "Toplantı notu",
        "{name} hobi kulübü, masa oyunları buluşmasında uygulanacak kuralları gözden "
        "geçirdi. Ekipler kendi oyun setlerini yanında getirecek. Kapanışta sonuç "
        "çizelgesi doldurulacak.",
    ),
    (
        "Duyuru",
        "{name} için hazırlanan gönüllü çalışma panosuna yeni duyurular eklendi. "
        "Panonun kullanımına ilişkin düşünceler bir sonraki tanışma gününde toplanacak.",
    ),
    (
        "Haber",
        "{name} ekibinin doğa yürüyüşü, arşiv için hazırlanan bir fotoğraf seçkisiyle kapandı. "
        "Gönüllüler yürüyüşte gördükleri bitki türlerini etkinlik defterine kaydetti.",
    ),
    (
        "Toplantı notu",
        "{name} okuma grubu tarafından gönderilen başlıklar gündeme alındı. "
        "Bir sonraki buluşmanın açılışını yapacak gönüllü belirlendi ve okuma sırası güncellendi.",
    ),
)


A = TemplateSet(
    boilerplate=CONTRACT_BOILERPLATE,
    contract_notes=CONTRACT_NOTES,
    contract_title="# {name}",
    signature_heading="İmza ve taraflar",
    party_paragraphs=(
        "{num} numaralı metin {date} tarihinde {a} ({left}) ile {b} ({right}) arasında imzalandı.",
        "{num} numaralı sözleşmenin imza tarihi {date}. İmza sayfasında {right} olarak {b}, "
        "{left} olarak {a} yer alır.",
        "{date} tarihli imzalar, {num} numaralı sözleşmede {left} sıfatıyla {a} ve {right} "
        "sıfatıyla {b} adına atıldı.",
    ),
    signature_after=0,
    clause_slots=(1, 3, 5),
    registry_title="# Ticaret Sicil Kaydı",
    registry_record_position=0,
    registry_records=(
        "{name} için açılan sicil dosyasında şirket merkezi {city}, vergi numarası {key} "
        "olarak geçer.",
        "{name} adına tescil edilen vergi kimlik numarası {key}; merkez ili ise {city} olarak"
        " kayıtlıdır.",
        "{name} sicil özetinde merkez ili {city} olarak belirtilmiştir. Vergi no: {key}.",
    ),
    registry_name="Kayıtlı ticaret unvanı “{name}” olarak geçer. Faaliyet alanı {sector}.",
    ownership_passages=(
        "{owner}, {owned} unvanlı şirketin sermayesinde {pct} oranında paya sahiptir.",
        "{owned} şirketinin ortakları arasında {pct} payla {owner} yer almaktadır.",
        "{owned} şirketindeki payların {pct} oranındaki bölümü {owner} tarafından tutulmaktadır.",
        "{owned} için ortaklık dökümünde {owner} ({pct}) pay sahibi olarak gösterilmiştir.",
    ),
    ownership_list="{name} için sicile işlenen ortak ve pay dökümü: {entries}.",
    ownership_entry="{name} ({pct})",
    ownership_separator=", ",
    report_title="# {name} 2025 Faaliyet Raporu",
    finance_heading="Finansal göstergeler",
    finance_position=0,
    finance_notes=(
        "Yılın kapanışında ciro {sales} milyon TL, net kâr {profit} milyon TL olarak "
        "hesaplandı. Dönem sonundaki çalışan sayısı {employees} kişiydi.",
        "Yıllık gelir tablosunda {sales} milyon TL ciro ve {profit} milyon TL dönem kârı "
        "raporlandı. İnsan kaynakları kayıtlarına göre yıl {employees} çalışanla tamamlandı.",
        "Yıl boyunca satışların toplamı {sales} milyon TL'ye ulaştı. {profit} milyon TL net "
        "kâr kaydedildi. Dönem raporuna esas personel mevcudu {employees} kişiydi.",
    ),
    report_sections=REPORT_SECTIONS,
    report_notes=REPORT_NOTES,
    portfolio_heading="Pay portföyü",
    meetings_heading="Birim görüşmeleri",
    meeting_notes=(
        "{name} için sonraki dönem hazırlıkları bakım programı üzerinden görüşüldü.",
        "{name} ekibinden gelen çalışma notları dönem kapanışında incelendi.",
        "İzleme toplantılarında {name} için uygulama kayıtlarının güncellenmesi ele alındı.",
    ),
    noise_title="# {heading} · {index:03d}",
    noise=NOISE,
    free_text_question="{number} numaralı metne göre {lower_question}",
    clauses={
        "bedel": ClauseText(
            "Sözleşme bedeli",
            (
                "Toplam sözleşme bedeli {v} olarak belirlenmiştir.",
                "Taraflar bu iş için toplam {v} ödenmesi üzerinde anlaşmıştır.",
            ),
            "Toplam sözleşme bedeli ne kadardır?",
        ),
        "sure": ClauseText(
            "Yürürlük süresi",
            (
                "Sözleşmenin yürürlük süresi {v} olarak kararlaştırılmıştır.",
                "İmzanın ardından sözleşme {v} boyunca yürürlükte kalır.",
            ),
            "Sözleşmenin yürürlük süresi kaç gündür?",
        ),
        "bildirim": ClauseText(
            "Fesih bildirimi",
            (
                "Olağan fesih için {v} önceden yazılı bildirim yapılır.",
                "Fesih ihbarnamesi, olağan fesih tarihinden en az {v} önce ulaşmalıdır.",
            ),
            "Olağan fesih için kaç gün önceden bildirim gerekir?",
        ),
        "faiz": ClauseText(
            "Gecikme faizi",
            (
                "Geciken ödemelere yıllık {v} gecikme faizi uygulanır.",
                "Ödeme vadesi aşılırsa yıllık gecikme faizi {v} üzerinden hesaplanır.",
            ),
            "Yıllık gecikme faizi oranı nedir?",
        ),
        "depozito": ClauseText(
            "Güvence bedeli",
            (
                "Kiracı, {v} tutarında depozito yatırır.",
                "Kira konusu yer teslim edilmeden {v} depozito ödenecektir.",
            ),
            "Depozito ne kadardır?",
        ),
        "gizlilik": ClauseText(
            "Gizlilik süresi",
            (
                "Gizlilik yükümlülüğü sözleşme bittikten sonra {v} daha sürer.",
                "Sözleşmenin sona ermesinden itibaren {v} boyunca gizlilik korunur.",
            ),
            "Sözleşme bittikten sonra gizlilik yükümlülüğü kaç gün sürer?",
        ),
        "kullanici": ClauseText(
            "Kullanıcı sınırı",
            (
                "Lisans, eş zamanlı en fazla {v} için geçerlidir.",
                "Yazılımın eş zamanlı kullanımı {v} ile sınırlandırılmıştır.",
            ),
            "Lisans aynı anda en fazla kaç kullanıcıya izin verir?",
        ),
    },
    questions={
        "company tax": (
            "“{name}” için sicilde hangi vergi numarası kayıtlı?",
            "Vergi kimlik numarası aranan şirket “{name}”. Sicilde hangi numara yazıyor?",
        ),
        "company city": (
            "“{name}” şirketinin merkezi hangi şehirde?",
            "Sicile göre “{name}” şirketinin merkez ili neresi?",
        ),
        "owner city": ("“{name}” şirketinde en yüksek payı elinde tutan ortağın merkezi nerede?",),
        "owner tax": ("“{name}” şirketinin en büyük hissedarının vergi numarası nedir?",),
        "ancestor city": (
            "“{name}” şirketinin en büyük hissedarına bakın. O hissedarın en büyük ortağının "
            "merkezi hangi şehirde?",
        ),
        "ancestor tax": (
            "“{name}” şirketindeki en yüksek paylı ortağın en yüksek paylı ortağı sicilde "
            "hangi vergi numarasıyla kayıtlı?",
        ),
        "third ancestor": (
            "“{name}” şirketinden başlayıp her adımda en büyük hissedara geçin. Üçüncü adımda"
            " ulaşılan şirket hangisi?",
        ),
        "indirect cities": (
            "“{name}” şirketinin tek bir aracı şirket üzerinden dolaylı pay sahibi olduğu "
            "şirketlerin merkezleri hangi şehirlerdedir? Her şehri bir kez yazın.",
        ),
        "signature date": (
            "{number} numaralı sözleşme hangi tarihte imzalandı?",
            "İmza tarihi aranan sözleşme {number}. Sözleşme ne zaman imzalanmış?",
        ),
        "both parties": ("{number} numaralı sözleşmenin iki tarafı kimlerdir?",),
        "party ancestor": (
            "{number} sözleşmesindeki {label} şirketin en büyük hissedarının en büyük "
            "hissedarı hangi şirkettir?",
            "{number} sözleşmesinin {label} tarafını bulun. Bu tarafın en yüksek paylı "
            "ortağında en büyük pay kime aittir?",
        ),
        "party city": (
            "{number} sözleşmesinde {label} sıfatıyla imza atan şirketin merkez ili neresi?",
        ),
        "party tax": (
            "{number} numaralı sözleşmedeki {label} şirketin vergi kimlik numarası nedir?",
        ),
        "party owner": (
            "{number} sözleşmesinin {label} tarafında yer alan şirketin en büyük hissedarı kimdir?",
        ),
    },
)
