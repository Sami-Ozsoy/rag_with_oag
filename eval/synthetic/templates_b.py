"""Held-out Turkish phrasing for the same synthetic facts and question families."""

from typing import TYPE_CHECKING

if TYPE_CHECKING or __package__:
    from .templates import ClauseText, TemplateSet
else:
    from templates import ClauseText, TemplateSet


CONTRACT_BOILERPLATE = (
    (
        "Evrak sürümleri",
        "Eklerden birinin yeni nüshası dolaşıma girdiğinde kapak sayfasına revizyon işareti konur. "
        "Arşiv görevlisi önceki nüshayı ayrı bir klasöre kaldırır ve değişen sayfaları dizine "
        "işler.",
    ),
    (
        "Tebellüğ kaydı",
        "Elden verilen yazının bir örneğine teslim alan görevli adını ve teslim anını işler. "
        "Bu örnek, göndericinin saklayacağı nüsha olup asıl evrak muhatabın kayıt biriminde kalır.",
    ),
    (
        "Saha giriş kartları",
        "Göreve başlamadan önce danışmadan geçici giriş kartı alınır. Giriş noktası değişirse "
        "kartın tanımı güvenlik masasında yenilenir ve daha önce açılan yetki iptal edilir.",
    ),
    (
        "Dosya adlandırma usulü",
        "Paylaşıma açılan her dosya adında belge türü ile sürüm işareti bulunur. Adı aynı olan "
        "belgeler yüklenirken içeriklerinin ayrıştığı noktalar gönderim fişinde açıkça yazılır.",
    ),
    (
        "Temsil yetkisinin teyidi",
        "Bir temsilci değiştiğinde görevlendirme yazısını yeni temsilci ibraz eder. Yetki "
        "belgesinin kapsamı, o kişinin imzalayacağı işlem türleri için sicil defterinden "
        "doğrulanır.",
    ),
    (
        "Numune mühürleri",
        "Teslim edilen numune açılmadan önce kabın üzerindeki mühür işareti tutanağa geçirilir. "
        "İşaret okunaksızsa yeni numune istenir ve ilk kap inceleme bitene dek muhafaza edilir.",
    ),
    (
        "Kabulde inceleme çizelgesi",
        "Muayeneye ayrılan kalemler kabul formunda ayrı satırlara dağıtılır. İnceleyicinin "
        "gözlemi ile teslimi yapan ekibin açıklaması formun karşılıklı sütunlarına yazılır.",
    ),
    (
        "Bakım penceresi",
        "Bakım nedeniyle erişime kapatılacak bölüm, vardiya panosunda geçici bir işaretle "
        "gösterilir. "
        "Tekrar kullanıma açma yetkisi, kontrol fişini imzalayan saha sorumlusuna aittir.",
    ),
    (
        "Dijital onay izleri",
        "Elektronik onay veren görevli işlem açıklamasına dayanak belgenin dizin kodunu girer. "
        "Sonradan yapılan düzeltme ilk kaydın üzerine yazılmaz ve yeni bir işlem olarak görünür.",
    ),
    (
        "Emanet araç fişi",
        "Ortak kullanımdan alınan bir araç için zimmet masasında alındı fişi açılır. İade "
        "sırasında "
        ""
        "teslimde gözlenen aşınma ve eksik parçalar aracın önceki durum çizelgesiyle "
        "karşılaştırılır.",
    ),
    (
        "Teknik çizimlerin okunması",
        "Ölçü hakkında tereddüt çıkarsa çizimin kenarındaki açıklama kutusu esas alınır. Ölçünün "
        "işe yansıması hakkında çizerin verdiği cevap çizim numarasıyla birlikte muhafaza edilir.",
    ),
    (
        "Soru sicili",
        "İşin uygulanması sırasında açılan her teknik soruya bir takip kodu atanır. Cevabı "
        "başka bir dosyada bulunan sorunun satırına o dosyaya götüren bir gönderme bırakılır.",
    ),
    (
        "İş değişikliği önerileri",
        "Uygulamada değişiklik isteyen birim, talebin sebebini ve dokunduğu işleri başvurusuna "
        "ekler. "
        "Karar çıkıncaya dek önerinin uygulanabilirliği çizelgenin deneme sütununda incelenir.",
    ),
    (
        "Teslim alanının paylaşımı",
        "Teslim noktasını birden çok ekibin kullanması halinde boşaltma sırası saha krokisine "
        "işlenir. "
        "Geçici kapatma gerektiğinde ekiplerin bekleyeceği yer de krokide ayrıca gösterilir.",
    ),
    (
        "Belge çevirileri",
        "Açıklama amacıyla hazırlanan çeviriye dayandığı nüshanın sayfa işareti yazılır. İki "
        "metin arasında yorum farkı çıktığında konu, ilgili sayfalar iliştirilerek yazıyla "
        "iletilir.",
    ),
    (
        "Vardiya devri",
        "Giden vardiya, henüz doğrulanmamış işlemleri sonraki vardiyanın defterine işler. Bu "
        "satırlar kapatıldığında kontrolü tamamlayan görevlinin adı silinmeyecek biçimde kalır.",
    ),
    (
        "Tarama kalitesi",
        "Fiziksel evraktan alınan görüntünün kenarları ile kaşesinin okunaklılığı kontrol edilir. "
        "Zayıf görüntü yeniden taranır ve arşivde hangi kopyanın kullanılacağı dizinde gösterilir.",
    ),
    (
        "Yazışma eki kontrolü",
        "Muhataba giden yazının sonuna iliştirilen belgelerin adları sayfa sırasına göre "
        "listelenir. "
        "Listede bulunup zarftan çıkmayan bir belge varsa evrak masası gönderi fişine bunu işler.",
    ),
    (
        "Deneme ortamına aktarım",
        "Uygulamadan alınan örnek kayıt, deneme ortamına taşınmadan önce izleme koduyla "
        "işaretlenir. "
        "Deneme bitince sonraki işlemlerde kullanılacak sonuçlar, örneğin kendisinden ayrı "
        "saklanır.",
    ),
    (
        "Düzeltme tutanağı",
        "Kâğıt üzerindeki bir hata giderilirken eski yazı okunur kalacak biçimde üstü çizilir. "
        "Yeni ifade satırın yanına eklenir ve değişikliği yapan kişi kenar boşluğunu paraflar.",
    ),
    (
        "Altlık ve ambalaj",
        "Taşıma için ayrılan kaplar, kullanım yerinde kolayca ayrılabilecek biçimde işaretlenir. "
        "Geri alınacak taşıma altlıkları, teslim mahallinde bırakılan malzemeden ayrı sayılır.",
    ),
    (
        "Geçici erişim tanımı",
        "Dışarıdan gelen görevliye verilen erişim, görevlendirildiği çalışma alanıyla "
        "sınırlandırılır. "
        "Görev yeri değiştiğinde yeni izin belgesi önceki erişimin kapatılmasını da gösterir.",
    ),
    (
        "Kontrol cihazı arızası",
        "Ölçüm aygıtının güvenilirliği konusunda kuşku oluşursa aygıt çalışma tezgâhından çekilir. "
        "O andan önce alınmış sonuçlar, gerekirse yeniden ölçülmek üzere işaretli olarak saklanır.",
    ),
    (
        "Yüz yüze oturum kaydı",
        "Görüşmede karar verilen işler, oturum sonunda katılımcıların görebileceği bir listeye "
        "alınır. "
        "Karara bağlanmayan başlıkların yanına neden bekledikleri ve hangi belgenin istendiği "
        "yazılır.",
    ),
    (
        "Malzeme kökeni",
        "Gelen malzemenin izlenebilmesi için ambalaj üzerindeki seri işareti depo fişine "
        "aktarılır. "
        ""
        "Seri işareti okunmadığında malzeme, kökeni doğrulanmış kalemlerle aynı rafa "
        "yerleştirilmez.",
    ),
    (
        "İşaretleme dili",
        "Sahada kullanılan renklerin ve simgelerin karşılığı girişteki lejantta açıklanır. Lejanta "
        "yapılan ekler, aynı işareti taşıyan etiketlerin gözden geçirilmesiyle birlikte yürütülür.",
    ),
    (
        "El kitabı dağıtımı",
        "Çalışma yöntemini açıklayan kitapçıklar, basım işareti denetlenerek görev noktalarına "
        "verilir. "
        "Yenisi dağıtıldığında eski nüshanın toplandığı yer, teslim çizelgesine ayrıca işlenir.",
    ),
    (
        "Uygulama uyuşmazlıklarının kaydı",
        "Talimatın sahada farklı yorumlandığı durumlarda kullanılan iki işlem yolu da açıklanır. "
        "Karar veren görevli, hangi yöntemi benimsediğini gerekçesiyle aynı kayıt üzerine yazar.",
    ),
)

REPORT_SECTIONS = (
    (
        "Üretim hattının ritmi",
        "Darboğazın kaynağını görmek isteyen atölye ekibi vardiyaları ayrı ayrı izledi. Hattın boş "
        "kaldığı anların çoğunda sıradaki malzeme henüz hazırlık tezgâhına ulaşmamıştı. Bunun "
        "üzerine "
        "hazırlık sırasını gösteren pano makine başından yükleme alanına taşındı. Yeni düzenin "
        "etkisi "
        "önce benzer işlerin art arda geldiği vardiyalarda gözlendi. Küçük seriler için ayrı bir "
        "akış "
        "denemesi yapılmasına gerek görüldüğünden bu işlerin çizelgesi sonraki oturuma bırakıldı.",
    ),
    (
        "Teklif defteri",
        "Teklif aşamasında bekleyen dosyalar, fiyat hesabından önce karşılanması gereken şartlara "
        "göre ayrıştırıldı. Özellikle çizim bekleyen işlerde ilk gönderim ile son açıklama "
        "arasındaki "
        "fark, satış masasında tutulan defterden görülebiliyordu. Ekip bu bilgiyi, cevapsız kalmış "
        "teknik soruları yeniden sıralamak için kullandı. Temsilciler saha ziyaretlerinden "
        "getirdiği "
        "gözlemleri aynı defterin boş kalan kenarına işledi. Yeni teklif formunda bu gözlemler "
        "için "
        ""
        "ayrılan kutunun nasıl kullanılacağı örnek dosyalarla anlatıldı.",
    ),
    (
        "Atölye yerleşiminin yenilenmesi",
        "Atölye krokisi üzerinde önce taşıma arabalarının dönüş yapabildiği alanlar işaretlendi. "
        "Ziyaret yollarına taşan geçici tezgâhlar için duvar boyunca daha geniş bir şerit ayrıldı. "
        "Yerleşimi sınamak üzere boş kasalarla yapılan turda bazı rafların görüş açısını kapattığı "
        "anlaşıldı. Raf yüksekliği değiştirilince malzeme çağrısı veren levhaların yeri de "
        "ayarlandı. "
        "Son çizim, taşıma rotasını fiilen kullanan vardiyaların açıklamalarıyla tamamlandı.",
    ),
    (
        "Alacakların izlediği takvim",
        "Tahsilat öngörüsünün ne kadar erken yapılabileceğini görmek için mali işler faturadan "
        "banka "
        "hareketine kadar oluşan belgeleri yan yana getirdi. Aynı müşteriye ait farklı "
        "siparişlerin "
        ""
        "birlikte ödendiği örneklerde dekont açıklamasından tek başına sonuç çıkarmak zordu. Ekip, "
        "bu örnekleri ayırabilmek için mutabakat dosyasına bir bağlantı sütunu ekledi. Takvimdeki "
        "olağandışı satırların nereden geldiği sonraki toplantıda bu sütun üzerinden tartışıldı.",
    ),
    (
        "Ölçüm noktalarının haritası",
        "Tüketim ölçerlerin kapsadığı bölümler, tesisin kat planı üzerinde farklı renklerle "
        "gösterildi. "
        "Bu çalışma sırasında iki ayrı kullanım noktasının aynı sayaçtan beslendiği görüldü. Bakım "
        "ekibi bu hatlar için geçici ölçüm yaparak ortak kaydın dağılımı hakkında fikir edindi. "
        "Ölçü alma yöntemi değiştiğinde eski çizelgeye ulaşılabilmesi için yeni plana arşiv "
        "numarası "
        "eklendi. Sonraki okuma turlarında kablo güzergâhı da aynı planın bir kopyasında "
        "doğrulandı.",
    ),
    (
        "İşe alışma gözlemleri",
        "Yeni ekip üyeleri, ilk vardiyalarında hangi işaretleri anlamakta zorlandıklarını "
        "mentorlara "
        "anlattı. Bu görüşmeler sırasında giriş kitapçığındaki kimi terimlerin sahada başka "
        "anlamlarda kullanıldığı ortaya çıktı. Eğitim birimi her terim için çalışanın "
        "karşılaşacağı "
        ""
        "bir örnek ekledi ve metni sonraki gruba okuttu. Örnekleri anlaşılır bulan çalışanlar, "
        "kitapçığın yanında dağıtılan saha krokisinde de benzer açıklamalar olmasını istedi.",
    ),
    (
        "Sayısal kayıtların dolaşımı",
        "Bir veri satırının üretim ekranından arşive ulaşıncaya dek hangi araçlardan geçtiği adım "
        "adım izlendi. İlk kopyada bulunup sonraki listede kaybolan alanlar ayrı bir deneme "
        "dosyasına "
        "taşındı. Ekip, kaybın aktarım ayarından mı kaynaklandığını anlamak için elle tutulan "
        "defteri "
        "de inceledi. Tarih damgalarının kullanımını tutarlı kılan düzenleme küçük bir grupta "
        "sınandı. İncelemenin ardından hangi kopyanın denetime sunulacağı sistem kılavuzuna "
        "yazıldı.",
    ),
    (
        "Tedarik rotalarının sınanması",
        "Yüklenip yola çıkmayı bekleyen kalemlerle henüz ambalajlanmamış olanlar farklı işaretler "
        "aldı. Bu ayrım, gecikmenin depo içindeki hazırlıktan mı taşıma rotasından mı geldiğini "
        "görmeyi kolaylaştırdı. Bir teslim günü iptal olduğunda boşalan yerin nasıl kullanılacağı "
        "taşıyıcıyla yapılan örnek planlama üzerinden çalışıldı. Saha ekibi yeni akışın rampadaki "
        "karşılaşmaları azaltıp azaltmadığını giriş fişlerinin sırasına bakarak değerlendirdi.",
    ),
    (
        "Raftaki bekleme süresi",
        "Hareketsiz kalan stokların belirlenmesi için sayım görevlileri önce son çekiş fişine "
        "baktı. "
        "Ölçümün tek bir güne dayanması halinde sık kullanılan bazı yedeklerin yanlış gruba "
        "girdiği "
        ""
        "ortaya çıktı. Bunun üzerine ayırma ölçütleri, rafın kullanım amacını gösterecek biçimde "
        "ayrıntılandırıldı. Depo krokisine aktarılan sonuçlar, yeni yerleşimle açılan alanın nasıl "
        "kullanılacağını tartışan vardiya temsilcilerine sunuldu.",
    ),
    (
        "İmza yollarının sınanması",
        "Denetçiler, işlem kaydına kadar ulaşmayan onayların hangi masada kaldığını araştırdı. "
        "Bazı hallerde bir önceki görevlinin kararı defterde durduğu halde dosyanın kapağına "
        "işlenmemişti. Benzer örnekler üzerinde onayın ilk kaynağına götüren bağlantılar sınandı. "
        "Belgeyi açan görevlinin ek bir yazışma yapmadan kaynağı görebilmesi öncelik kazandı. "
        "Böylece yeni kapak düzeninin hangi işlem sınıflarına yayılabileceği görüldü.",
    ),
    (
        "Geri dönüş kayıtlarının okunması",
        "Sahadan gelen çağrıların metinleri, çözüme ulaştıran işlemlerle beraber okundu. Aynı "
        "kelimeyi "
        "kullanan kayıtların farklı aksaklıklara ait olabildiği görüldüğünden çağrının yapıldığı "
        "noktayı belirten alan öne çıkarıldı. Yineleyen başvurular bu alan üzerinden "
        "karşılaştırıldı. "
        "Ekip, önceki açıklama zaten yeterliyse dosyayı açanın o açıklamaya nasıl erişeceğini "
        "gösteren küçük bir deneme hazırladı. Uygulama masaları denemede kullanılan örnekleri "
        "seçti.",
    ),
    (
        "Yedek donanım rotası",
        "Bakım bekleyen makinelerin kayıtları, yerlerine getirilen yedeklerin hareketleriyle yan "
        "yana incelendi. Birkaç istasyona ulaşmak için aynı koridor kullanıldığından taşımanın "
        "yapıldığı saat, bakımın kendi süresi kadar önemliydi. Vardiya amirleri, dar "
        "geçişlerin boş "
        ""
        "kalacağı aralıkları işaretleyen bir plan üzerinde anlaştı. Planda görülen kısıtlar, "
        "teslim "
        ""
        "edilecek parçaların hangi sırayla hazırlanacağını belirleyen fişlere de yansıdı.",
    ),
    (
        "Erken uyarı işaretleri",
        "Aksamalar başlamadan önce elde hangi işaretlerin olduğu, geriye dönük örnekler üzerinden "
        "araştırıldı. Üretim ve satın alma masaları aynı olay için farklı tarihler kullandığında "
        "değerlendirmeyi ortak bir zaman çizgisine taşımak gerekti. Her masanın hangi bilginin ilk "
        "alıcısı olduğunu gösteren çizim bu çalışma sırasında hazırlandı. Bir sonraki denemede "
        "uyarının kime ulaşacağını saptamak için bu çizimden yararlanılması benimsendi.",
    ),
    (
        "Çalışma tablolarının kökeni",
        "Toplamları oluşturan satırlar örnek aylar seçilerek ilk belgeye dek takip edildi. "
        "Devreden "
        ""
        "hareketi ayrıca işaretleyen tabloların karşılaştırmada daha rahat okunduğu görüldü. Bu "
        "nedenle veri derleyen birimler, dönemi kapatan satırla henüz açık olanı farklı "
        "işaretlerle "
        ""
        "göstermeye başladı. Açıklama ihtiyacı duyulan örneklerde birim kodunun yanında dayanak "
        "dosyanın bulunduğu klasör de yazıldı. Arşiv dizini buna göre genişletildi.",
    ),
    (
        "Geçiş yollarındaki gözlemler",
        "Saha turunda zemindeki uyarı işaretlerinin araç sürücüsü tarafından nereden görüldüğü "
        "kontrol edildi. Kör noktaların yoğun olduğu alanların krokisine yaklaşma yönleri de "
        "eklendi. "
        "Ekip, yaya yolunu değiştirmeden görünürlüğü artırabilecek levha biçimlerini ayrı ayrı "
        "denedi. "
        "Görüş mesafesi hakkında ekiplerin aktardığı bilgiler, izleyen turda fotoğraflarla "
        "karşılaştırılmak üzere gözlem çizelgesine geçirildi.",
    ),
    (
        "Bir sonraki çevrime hazırlık",
        "Yönetim yeni dönemin ilk işlerini belirlerken yarım kalan denemeler için sorumlu "
        "ekiplerden "
        "kısa özetler topladı. Kaynağı hazır olan işler ile üst karara bağlı olanlar ayrı "
        "gündemlerde "
        "görüşüldü. Bir işin açılış koşulu değiştiğinde bunun sonraki işleri nasıl etkilediği plan "
        "panosunda oklarla gösterildi. Vardiyaya inecek son listede ekip üyesinin hangi belgeden "
        "başlayabileceği, işin adının hemen altında yazıldı.",
    ),
)

NOISE = (
    (
        "Mahalle sahnesi",
        "Kukla gösterisinde kullanılacak gölgeler akşam provasına yetişti. Program defterine "
        "{name} ekibinden gelen seyircilerin giriş kapısı da çizildi. Ara verildiğinde izleyiciler "
        "bahçedeki bez afişlerin nasıl boyandığını atölye gönüllülerine sordu.",
    ),
    (
        "Tohum takası",
        "{name} ekibinden bahçe meraklıları, pazarda hangi tohumları masaya koyacağını seçti. "
        "Torbalara yöre ve hasat mevsimi yazıldı. Parkın gölge tarafındaki masaya küçük bir sulama "
        "maketi kuruldu ve ziyaretçiler ekim denemelerini orada konuştu.",
    ),
    (
        "Açık hava çizimleri",
        "İskele başında çizim yapan grup, kâğıtlar rüzgârda kalkmasın diye mandal getirdi. "
        "{name} ekibinden katılanlar ufuk çizgisini farklı renklerle denedi. Toplanma saati "
        "yaklaşınca "
        "yarım kalan çalışmaların fotoğrafları ortak albüme yüklendi.",
    ),
    (
        "Yerel kuş sayımı",
        "Dürbünlerini hazırlayan gönüllüler, sazlığın kenarındaki taş yolda buluştu. Geziden "
        "önce {name} ekibine gözlem noktalarını gösteren bir broşür ulaştı. Suya yakın duran "
        "kuşlar "
        ""
        "deftere çizilirken uzaktan duyulan ötüşler ayrı bir sayfada tarif edildi.",
    ),
    (
        "Seramik fırını",
        "Çömlek atölyesinde küçük kaseler kurumaya bırakıldı. {name} ekibinden gelenler, "
        "hangi sırın ışıkta mat göründüğünü renk taşlarına bakarak tartıştı. Sonraki buluşmada "
        "fırından çıkacak parçalar için pencere yanında bir sergi köşesi ayrıldı.",
    ),
    (
        "Bisiklet rotası defteri",
        "Park çıkışındaki yokuşu geçince dinlenmeye uygun bir çeşme bulundu. {name} ekibinden "
        "bisikletçiler dönüş yolunu çizdikleri deftere bu noktayı da ekledi. Rota tamamlandığında "
        "herkes en çok hangi manzarayı hatırladığını küçük bir karta yazdı.",
    ),
)

B = TemplateSet(
    contract_title="# Akit dosyası | {name}",
    signature_heading="Taraf künyeleri ve akdin düzenlenişi",
    party_paragraphs=(
        "İki imza taşıyan {num} kodlu belgenin tarih hanesine {date} yazılmıştır. "
        "Belgeye {a} adına ({left}) ve {b} adına ({right}) imza konmuştur.",
        "{a} ile {b}, {date} günü akdi imzalamıştır. {num} referansını taşıyan nüshada "
        "birincisinin görevi “{left}”, ikincisinin görevi “{right}” diye yazılıdır.",
        "İmza çizelgesinde “{left}” hanesini {a}, “{right}” hanesini ise {b} doldurmuştur. "
        "Çizelgenin bağlı olduğu {num} kodlu akit için imzaların atıldığı gün {date} olarak "
        "kayda alınmıştır.",
    ),
    signature_after=3,
    clause_slots=(0, 2, 6),
    boilerplate=CONTRACT_BOILERPLATE,
    contract_notes=(
        "Uygulayıcılar, dayanak evrakı bulmak için kapak dizinindeki bağlantıya başvurur.",
        "Karar ve ona dayanak olan görüşler aynı dosya numarasıyla izlenir.",
        "Bir açıklama eklenecekse açıklamanın hangi nüshaya ait olduğu başında gösterilir.",
    ),
    registry_title="# Tescil fişinden alıntı",
    registry_record_position=1,
    registry_records=(
        "{name} — Merkez ili {city} / Vergi mükellefi kimliği {key}.",
        "Merkezi {city} ilinde bulunan {name} adına vergi sicilinde {key} kimliği açılmıştır.",
        "Vergi idaresi {key} sayılı kimliği {name} adına düzenlemiştir. Aynı şirketin merkez "
        "adresi {city} ili sınırları içindedir.",
    ),
    registry_name="Faaliyet sınıfı {sector} olan kuruluş «{name}» adıyla tescillidir.",
    ownership_passages=(
        "{owner}, {owned} sermayesine {pct} oranıyla iştirak eder.",
        "{owned} nezdindeki hisselerden {owner} adına kayıtlı olanların sermayeye oranı {pct}.",
        "Sermayenin {pct} oranındaki dilimi, {owned} ortaklık defterinde {owner} adına "
        "tescil edilmiştir.",
        "{owned} sermayesindeki {pct} payın sahibi olarak {owner} kayda geçmiştir.",
    ),
    ownership_list="{name} sermayesinin ortaklara göre dağılımı aşağıda verilmiştir.\n\n{entries}",
    ownership_entry="- {name} — sermaye payı {pct}",
    ownership_separator="\n",
    report_title="# 2025 Yıl Sonu İncelemesi | {name}",
    finance_heading="Dönemin sayısal özeti",
    finance_position=2,
    finance_notes=(
        "Hesap cetveli dönemi {sales} milyon TL satış geliri ve {profit} milyon TL net kazançla "
        "kapattı. Kapanış bordrosunda {employees} çalışan bulunuyordu.",
        "Personel mevcudu kapanış itibarıyla {employees} kişiye ulaştı. Dönem hesabına göre "
        "satışlardan {sales} milyon TL gelir sağlandı ve net kâr hanesine {profit} milyon TL "
        "yazıldı.",
        "Kapanış cetvelinin satış satırı {sales} milyon TL, net sonuç satırı {profit} milyon TL "
        "gösteriyor. İnsan kaynakları dönemin son bordrosunda {employees} kişiyi kayda aldı.",
    ),
    report_sections=REPORT_SECTIONS,
    report_notes=(
        "Birim, bu çalışmadan kalan örnekleri yeni dönem dosyasında erişime açtı.",
        "Sunumda kullanılan taslakların hangi tarihte hazırlandığı arşiv dizininde görülebilir.",
        "Ara sonuçları karşılaştıran ekip, bir sonraki oturum için ayrı bir çizelge hazırladı.",
    ),
    portfolio_heading="İştirak kayıtlarından seçmeler",
    meetings_heading="Saha temaslarından notlar",
    meeting_notes=(
        "Takvim taslağı hakkında bilgi almak üzere {name} ekibiyle kısa bir oturum yapıldı.",
        "{name} saha ekibi, vardiya çizelgesindeki boş satırlarla ilgili gözlemlerini iletti.",
        "Yeni formun okunabilirliği, {name} temsilcilerinin getirdiği örnek üzerinde tartışıldı.",
    ),
    noise_title="# Etkinlik ajandası {index:03d} | {heading}",
    noise=NOISE,
    free_text_question="Akit referansı {number}. {question}",
    clauses={
        "bedel": ClauseText(
            "İşin karşılığı",
            (
                "İşin tamamı karşılığında ödenecek meblağ {v} tutarındadır.",
                "Ödeme çizelgesinin genel toplamında akde ait {v} yer alır.",
            ),
            "İşin tamamı için tarafların anlaştığı tutar nedir?",
        ),
        "sure": ClauseText(
            "Akdin geçerli kaldığı dönem",
            (
                "Akdin geçerli olduğu dönem, imza gününden itibaren {v} sürer.",
                "Yürürlük takviminde imzayla başlayan geçerlilik dönemi {v} uzunluğundadır.",
            ),
            "Akdin imzayla başlayan geçerlilik dönemi kaç gün devam eder?",
        ),
        "bildirim": ClauseText(
            "Olağan sona erdirme ihbarı",
            (
                "Olağan yoldan akdi sona erdirmek isteyen taraf, yazılı ihbarı en az {v} kala "
                "muhatabına ulaştırır.",
                "Yazılı ihbarın ulaştığı gün ile olağan fesih günü arasında en az {v} bırakılır.",
            ),
            "Olağan fesih günüyle yazılı ihbarın ulaşması arasında en az kaç gün olmalı?",
        ),
        "faiz": ClauseText(
            "Temerrüt hesabı",
            (
                "Vaktinde karşılanmayan borç için temerrüt hesabında {v} yıllık faiz esas alınır.",
                "Akdin temerrüt cetveli, vadesinden sonra yapılan ödemeler için yıllık faiz "
                "oranını {v} gösterir.",
            ),
            "Vade geçince temerrüt hesabında hangi yıllık faiz oranı esas alınıyor?",
        ),
        "depozito": ClauseText(
            "Kiracının vereceği teminat",
            (
                "Kiracıdan teslim öncesinde alınacak güvence {v} olarak hesaplanmıştır.",
                "Akdin teminat hanesinde, kiracının yatıracağı depozito için {v} ayrılmıştır.",
            ),
            "Kiracı kira teminatı olarak hangi tutarı yatıracak?",
        ),
        "gizlilik": ClauseText(
            "Sona ermeden sonraki sır saklama",
            (
                "Akdin bitiminden sonra sır saklama borcunun kalkması için {v} geçmelidir.",
                "Akdin sona erdiği günü izleyen {v} boyunca tarafların sır saklama borcu devam "
                "eder.",
            ),
            "Akdin bitimini izleyen kaç gün boyunca taraflar sır saklamakla yükümlü?",
        ),
        "kullanici": ClauseText(
            "Eşzamanlı oturum kapasitesi",
            (
                "Lisans denetimi, aynı anda oturum açabilecek kişileri {v} ile sınırlar.",
                "Lisans kapsamındaki yazılıma aynı anda erişebilecek kişi sayısı en çok {v} olur.",
            ),
            "Lisans kapsamında aynı anda yazılıma erişebilecek kişi sayısının üst sınırı nedir?",
        ),
    },
    questions={
        "company tax": (
            "Vergi mükellefi olarak «{name}» adına düzenlenen kimlik numarasını verir misiniz?",
            "Tescil fişinde «{name}» adına açılmış vergi kimliğinin numarası kaçtır?",
        ),
        "company city": (
            "Tescil fişindeki merkez adresi «{name}» için hangi ili gösteriyor?",
            "Merkez adresinin bağlı olduğu ili yazar mısınız? Aradığım şirket «{name}».",
        ),
        "owner city": (
            "«{name}» sermayesindeki pay sıralamasında başı çeken şirketin resmî merkez adresi "
            "hangi ilde?",
        ),
        "owner tax": (
            "«{name}» sermayesindeki en yüksek payı taşıyan ortak vergi siciline hangi kimlikle "
            "girmiş?",
        ),
        "ancestor city": (
            "«{name}» üzerinden sermaye zincirinde iki seviye yukarı çıkın ve her seviyede payı "
            "en yüksek ortağı seçin. Vardığınız şirketin merkez adresi hangi ilde kayıtlı?",
        ),
        "ancestor tax": (
            "«{name}» için payı en yüksek ortağı seçip aynı seçimi onun ortakları arasında "
            "tekrarladığınızda, ulaştığınız şirketin vergi kimliği kaçtır?",
        ),
        "third ancestor": (
            "Sermaye zincirini «{name}» şirketinden yukarı doğru üç halka izleyin. Her halkada "
            "payı en yüksek ortağı seçtiğinizde son halkanın unvanı nedir?",
        ),
        "indirect cities": (
            "«{name}» şirketinin doğrudan ortak olduğu şirketleri bulun ve onların doğrudan "
            "ortak olduğu kuruluşlara bakın. Bu kuruluşların merkez illerini tekrarsız sıralayın.",
        ),
        "signature date": (
            "Akit dosyasındaki {number} referansı için imzaların atıldığı günü yazabilir misiniz?",
            "{number} kodlu akdin imza çizelgesinde hangi gün kayıtlı?",
        ),
        "both parties": (
            "İki imza sahibinin şirket adlarını {number} referanslı akit dosyasından çıkarır "
            "mısınız?",
        ),
        "party ancestor": (
            "{number} kodlu akitte “{label}” hanesindeki şirketten başlayarak ortaklık zincirinde "
            "iki kez en yüksek payı seçin. Zincirin sonunda hangi şirket var?",
            "{number} kodlu belgede “{label}” hanesindeki şirketin en yüksek paylı ortağını "
            "bulun. O ortağın sermaye dağılımında ilk sırayı hangi şirket alıyor?",
        ),
        "party city": (
            "{number} referanslı akdin “{label}” hanesinde yazılı şirket hangi ili merkez adresi "
            "olarak tescil ettirmiş?",
        ),
        "party tax": (
            "Akit referansı {number} ve taraf hanesi “{label}”. Bu hanedeki şirket vergi "
            "siciline hangi kimlik numarasıyla girmiş?",
        ),
        "party owner": (
            "{number} kodlu belgede “{label}” hanesini dolduran şirketin sermaye dağılımında "
            "ilk sırayı alan ortak hangi şirkettir?",
        ),
    },
)
