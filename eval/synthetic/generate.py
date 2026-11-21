"""Generate a Turkish contract evaluation corpus from a seeded world model."""

from __future__ import annotations

import argparse
import json
import random
import shlex
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date, timedelta
from itertools import cycle, pairwise
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast

import tiktoken
import yaml

if TYPE_CHECKING or __package__:
    from .templates import (
        CONTRACT_BOILERPLATE,
        CONTRACT_NOTES,
        NOISE,
        REPORT_NOTES,
        REPORT_SECTIONS,
    )
    from .text import matches, norm
else:
    from templates import CONTRACT_BOILERPLATE, CONTRACT_NOTES, NOISE, REPORT_NOTES, REPORT_SECTIONS
    from text import matches, norm

GOLDEN = Path(__file__).resolve().parents[1] / "golden"
SET_DEFAULT = Path("eval/synthetic/sets/default")
STEMS = (
    "Zirve Toros Kuzey Ege Atlas Ada Akasya Alize Anadolu Ardıç Asya Ayaz Aydın Bahar Belen "
    "Bereket Bilge Bora Bozdağ Bulut Çağlayan Çamlı Çınar Dalyan Defne Delta Deniz Deren "
    "Doğa Doğu Doruk Duru Elvan Engin Erguvan Ekin Fırat Gelincik Gediz Gökçe Güneş Güney "
    "Hazar Hisar Ihlamur Ilgın Irmak Işık İnci Kanyon Karaca Karadeniz Kardelen Kaya Kemer "
    "Kervan Kıyı Koru Kozalak Kuzgun Lale Liman Lodos Marmara Mercan Meşe Milas Nil Ova Pınar "
    "Poyraz Rüzgâr Sedir Simge Suna Taşkent Toprak Turna Ufuk Uludağ Umut Vadi Yakamoz Yankı "
    "Yayla Yeşil Yıldız Yonca Zeytin"
).split()
SECTORS = (
    "Gıda Kimya Lojistik Enerji Yazılım İnşaat Tekstil Danışmanlık Mekanik Tarım Ambalaj "
    "Elektronik Metal Makine Mobilya Cam Kâğıt Yatırım Yapı Sağlık"
).split()
CITIES = (
    "İstanbul Ankara İzmir Bursa Antalya Adana Mersin Konya Kayseri Kocaeli Gaziantep "
    "Samsun Trabzon Erzurum Eskişehir Denizli Manisa Balıkesir Aydın Muğla Tekirdağ Edirne "
    "Çanakkale Sakarya Malatya Diyarbakır Şanlıurfa Van Ordu Hatay"
).split()
MONTHS = ("Ocak Şubat Mart Nisan Mayıs Haziran Temmuz Ağustos Eylül Ekim Kasım Aralık").split()
CONTRACT_TYPES = (
    ("HS", "hizmet", "Hizmet Sözleşmesi"),
    ("KS", "kira", "Kira Sözleşmesi"),
    ("ES", "enerji", "Enerji Tedarik Sözleşmesi"),
    ("DS", "danismanlik", "Danışmanlık Sözleşmesi"),
    ("YS", "lisans", "Yazılım Lisans Sözleşmesi"),
)
ROLE_NAMES = {
    "musteri": "müşteri",
    "saglayici": "sağlayıcı",
    "kiraci": "kiracı",
    "kiraya_veren": "kiraya veren",
}
GROUPS = ("single_hop", "multi_hop", "deep_hop", "free_text")
ONTOLOGY_PROPERTIES = frozenset({"vergi_no", "sehir", "imza_tarihi", "numara", "tur"})
type TokenCount = Callable[[str], int]


@dataclass(frozen=True)
class Company:
    key: str
    base: str
    form: str
    city: str
    sector: str

    @property
    def name(self) -> str:
        return f"{self.base} {self.form}"

    @property
    def holding(self) -> bool:
        return self.sector == "Holding"

    @property
    def registry(self) -> str:
        return f"sicil_{self.key}.md"

    @property
    def report(self) -> str:
        return f"faaliyet_{self.key}.md"


@dataclass(frozen=True)
class Share:
    owner: Company
    owned: Company
    percent: int


@dataclass(frozen=True)
class Clause:
    field: str
    heading: str
    value: int
    unit: str
    statements: tuple[str, str]
    question: str

    def formats(self) -> list[str]:
        if self.unit == "%":
            value = f"{self.value // 100},{self.value % 100:02d}"
            return [
                f"%{value}",
                f"% {value}",
                f"yüzde {value}",
                f"{value}%",
                f"%{value.replace(',', '.')}",
            ]
        return list(
            dict.fromkeys(
                f"{number} {self.unit}"
                for number in (
                    f"{self.value:,}".replace(",", "."),
                    str(self.value),
                    f"{self.value:,}",
                )
            )
        )


@dataclass(frozen=True)
class Contract:
    number: str
    kind: str
    title: str
    signed: date
    parties: tuple[Company, Company]
    clauses: tuple[Clause, ...]

    @property
    def name(self) -> str:
        return f"{self.number} {self.title}"

    @property
    def document(self) -> str:
        return f"{self.number.lower().replace('-', '_')}.md"

    @property
    def roles(self) -> tuple[str, str]:
        return ("kiraci", "kiraya_veren") if self.kind == "kira" else ("musteri", "saglayici")


@dataclass(frozen=True)
class Fact:
    subject: str
    field: str
    value: str


@dataclass(frozen=True)
class Evidence:
    fact: Fact
    literals: tuple[str, ...]


@dataclass
class Document:
    kind: str
    filename: str
    blocks: list[str] = field(default_factory=list)
    claims: list[tuple[str, Evidence]] = field(default_factory=list)

    def add(self, text: str, *evidence: Evidence) -> None:
        self.blocks.append(text)
        self.claims.extend((text, item) for item in evidence)

    @property
    def text(self) -> str:
        return "\n\n".join(self.blocks) + "\n"


class QuestionRow(TypedDict):
    q: str
    docs: list[str]
    answer: list[list[str]]
    chain: list[list[dict[str, object]]]


class SourceEvidence(TypedDict):
    doc: str
    literals: list[str]


@dataclass(frozen=True)
class Question:
    subject: str
    family: str
    text: str
    paths: tuple[tuple[str, ...], ...]
    alias_text: str | None = None


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def make_companies(rng: random.Random, count: int) -> list[Company]:
    holding_count = max(1, round(count / 10))
    pair_count = round(count / 10)
    regular_count = count - holding_count - pair_count
    require(regular_count >= 3, "unsupported company count")
    stems = rng.sample(STEMS, len(STEMS))
    # Preserve some genuinely unambiguous one-word aliases, even in the default world.
    reserved = min(15, max(1, regular_count // 5))
    require(holding_count <= len(stems) - reserved, "holding name pool exhausted")
    base_pool = [(f"{stem} {sector}", sector) for stem in stems[reserved:] for sector in SECTORS]
    require(regular_count - reserved <= len(base_pool), "company name pool exhausted")
    names = [(f"{stem} Holding", "Holding") for stem in stems[reserved : reserved + holding_count]]
    for stem in stems[:reserved]:
        sector = rng.choice(SECTORS)
        names.append((f"{stem} {sector}", sector))
    names.extend(rng.sample(base_pool, regular_count - reserved))

    def company(base: str, sector: str, cities: list[str]) -> Company:
        return Company(
            next(tax_numbers),
            base,
            "A.Ş." if sector == "Holding" else rng.choice(("A.Ş.", "Ltd. Şti.")),
            rng.choice(cities),
            sector,
        )

    tax_numbers = iter(
        str(value) for value in rng.sample(range(1_000_000_000, 10_000_000_000), count)
    )
    companies = [company(base, sector, CITIES) for base, sector in names]
    for original in rng.sample(companies[holding_count:], pair_count):
        extra = rng.choice(("Pazarlama", "Dış Ticaret", "Dağıtım", "Yatırım Grubu"))
        companies.append(
            company(
                f"{original.base} {extra}",
                original.sector,
                [city for city in CITIES if city != original.city],
            )
        )
    require(len({norm(c.base) for c in companies}) == count, "company name collision")
    return companies


def make_shares(rng: random.Random, companies: list[Company]) -> list[Share]:
    holdings = [c for c in companies if c.holding]
    regular = [c for c in companies if not c.holding]
    rng.shuffle(regular)
    size = len(regular)
    first = max(1, size // 5)
    second = first + max(1, size // 4)
    third = max(second + 1, size - size // 5)
    layers = [holdings, regular[:first], regular[first:second], regular[second:third]]
    shares = [
        Share(rng.choice(parents), child, rng.randint(51, 85))
        for parents, children in pairwise(layers)
        for child in children
    ]

    # Join separate components for additional shareholders. This preserves an
    # undirected forest even though some targets have multiple incoming edges.
    roots = {c.key: c.key for c in companies}

    def root(company: Company) -> str:
        key = company.key
        while roots[key] != key:
            key = roots[key]
        return key

    for share in shares:
        roots[root(share.owned)] = root(share.owner)
    depths = {c.key: level for level, layer in enumerate(layers) for c in layer}
    for share in shares.copy():
        spare = 100 - share.percent
        for _ in range(rng.choices((0, 1, 2), weights=(6, 3, 1), k=1)[0]):
            candidates = [
                c
                for c in companies
                if depths.get(c.key, 0) < depths[share.owned.key] and root(c) != root(share.owned)
            ]
            if not candidates or spare < 5:
                break
            owner = rng.choice(candidates)
            percent = rng.randint(5, min(20, spare))
            shares.append(Share(owner, share.owned, percent))
            roots[root(share.owned)] = root(owner)
            spare -= percent
    return shares


def make_contracts(rng: random.Random, count: int, companies: list[Company]) -> list[Contract]:
    specs = (
        (
            "bedel",
            "Sözleşme bedeli",
            "TL",
            (
                "Toplam sözleşme bedeli {v} olarak belirlenmiştir.",
                "Taraflar bu iş için toplam {v} ödenmesi üzerinde anlaşmıştır.",
            ),
            "Toplam sözleşme bedeli ne kadardır?",
            range(50_000, 15_000_001, 50),
        ),
        (
            "sure",
            "Yürürlük süresi",
            "gün",
            (
                "Sözleşmenin yürürlük süresi {v} olarak kararlaştırılmıştır.",
                "İmzanın ardından sözleşme {v} boyunca yürürlükte kalır.",
            ),
            "Sözleşmenin yürürlük süresi kaç gündür?",
            range(90, 1_826),
        ),
        (
            "bildirim",
            "Fesih bildirimi",
            "gün",
            (
                "Olağan fesih için {v} önceden yazılı bildirim yapılır.",
                "Fesih ihbarnamesi, olağan fesih tarihinden en az {v} önce ulaşmalıdır.",
            ),
            "Olağan fesih için kaç gün önceden bildirim gerekir?",
            range(15, 366),
        ),
        (
            "faiz",
            "Gecikme faizi",
            "%",
            (
                "Geciken ödemelere yıllık {v} gecikme faizi uygulanır.",
                "Ödeme vadesi aşılırsa yıllık gecikme faizi {v} üzerinden hesaplanır.",
            ),
            "Yıllık gecikme faizi oranı nedir?",
            range(550, 5_401),
        ),
        (
            "depozito",
            "Güvence bedeli",
            "TL",
            (
                "Kiracı, {v} tutarında depozito yatırır.",
                "Kira konusu yer teslim edilmeden {v} depozito ödenecektir.",
            ),
            "Depozito ne kadardır?",
            range(20_000, 400_001, 50),
        ),
        (
            "gizlilik",
            "Gizlilik süresi",
            "gün",
            (
                "Gizlilik yükümlülüğü sözleşme bittikten sonra {v} daha sürer.",
                "Sözleşmenin sona ermesinden itibaren {v} boyunca gizlilik korunur.",
            ),
            "Sözleşme bittikten sonra gizlilik yükümlülüğü kaç gün sürer?",
            range(400, 3_651),
        ),
        (
            "kullanici",
            "Kullanıcı sınırı",
            "kullanıcı",
            (
                "Lisans, eş zamanlı en fazla {v} için geçerlidir.",
                "Yazılımın eş zamanlı kullanımı {v} ile sınırlandırılmıştır.",
            ),
            "Lisans aynı anda en fazla kaç kullanıcıya izin verir?",
            range(40, 5_001),
        ),
    )
    # Sample each clause from its own value pool without replacement where
    # possible. The referenced contract and clause always select a single value.
    values = {}
    for name, _, _, _, _, pool in specs:
        values[name] = (
            iter(rng.sample(pool, count))
            if count <= len(pool)
            else cycle(rng.sample(pool, len(pool)))
        )
    kinds = [CONTRACT_TYPES[i % len(CONTRACT_TYPES)] for i in range(count)]
    rng.shuffle(kinds)
    sequences: Counter[tuple[str, int]] = Counter()
    contracts = []
    for prefix, kind, title in kinds:
        signed = date(2021, 1, 1) + timedelta(days=rng.randrange(1_826))
        sequences[prefix, signed.year] += 1
        number = f"{prefix}-{signed.year}-{sequences[prefix, signed.year]:03d}"
        first, second = rng.sample(companies, 2)
        clauses = tuple(
            Clause(name, heading, next(values[name]), unit, statements, question)
            for name, heading, unit, statements, question, _ in specs
            if name not in {"depozito", "kullanici"}
            or (name == "depozito" and kind == "kira")
            or (name == "kullanici" and kind == "lisans")
        )
        contracts.append(Contract(number, kind, title, signed, (first, second), clauses))
    return contracts


def date_formats(value: date) -> list[str]:
    return [
        f"{value.day} {MONTHS[value.month - 1]} {value.year}",
        value.isoformat(),
        f"{value:%d.%m.%Y}",
    ]


def percent_formats(value: int) -> tuple[str, ...]:
    return (f"yüzde {value}", f"%{value}", f"% {value}", f"{value}%")


def name_variants(companies: list[Company]) -> dict[str, tuple[str, ...]]:
    names = {c.key: norm(c.name) for c in companies}
    variants = {}
    for c in companies:
        alternate = f"{c.base} " + ("AŞ" if c.form == "A.Ş." else "Ltd Şti")
        shortened = (c.base, c.base.split()[0])
        safe = []
        for name in shortened:
            folded = norm(name)
            if all(folded not in other for key, other in names.items() if key != c.key):
                safe.append(name)
        variants[c.key] = (c.name, alternate, *safe)
    return variants


def mention(
    rng: random.Random, names: dict[str, tuple[str, ...]], company: Company, *, later: bool = False
) -> str:
    options = names[company.key]
    if later and len(options) > 2:
        return rng.choice(options[2:])
    # The first mention uses either the full name, the suffix-free base, or an
    # alternate spelling of the legal form. One-word aliases come later.
    return rng.choices(options[:3], weights=(6, 2, 2)[: len(options[:3])], k=1)[0]


def ownership_passage(
    rng: random.Random,
    share: Share,
    names: dict[str, tuple[str, ...]],
    *,
    owner_name: str | None = None,
    owned_name: str | None = None,
) -> tuple[str, Evidence]:
    owner = owner_name or mention(rng, names, share.owner)
    owned = owned_name or mention(rng, names, share.owned)
    percent = rng.choice(percent_formats(share.percent))
    text = rng.choice(
        (
            "{owner}, {owned} unvanlı şirketin sermayesinde {pct} oranında paya sahiptir.",
            "{owned} şirketinin ortakları arasında {pct} payla {owner} yer almaktadır.",
            "{owned} şirketindeki payların {pct} oranındaki bölümü {owner} "
            "tarafından tutulmaktadır.",
            "{owned} için ortaklık dökümünde {owner} ({pct}) pay sahibi olarak gösterilmiştir.",
        )
    ).format(owner=owner, owned=owned, pct=percent)
    return text, Evidence(Fact(share.owned.key, "owner", share.owner.key), (owner, owned, percent))


def render_registry(
    rng: random.Random,
    company: Company,
    shares: list[Share],
    names: dict[str, tuple[str, ...]],
) -> Document:
    doc = Document("trade registry", company.registry)
    doc.add("# Ticaret Sicil Kaydı")
    first = mention(rng, names, company)
    identity = rng.choice(
        (
            "{name} için açılan sicil dosyasında şirket merkezi {city}, vergi "
            "numarası {key} olarak geçer.",
            "{name} adına tescil edilen vergi kimlik numarası {key}; merkez ili "
            "ise {city} olarak kayıtlıdır.",
            "{name} sicil özetinde merkez ili {city} olarak belirtilmiştir. Vergi no: {key}.",
        )
    ).format(name=first, city=company.city, key=company.key)
    doc.add(
        identity,
        Evidence(Fact(company.key, "vergi_no", company.key), (first, company.key)),
        Evidence(Fact(company.key, "sehir", company.city), (first, company.city)),
    )
    doc.add(
        f"Kayıtlı ticaret unvanı “{company.name}” olarak geçer. Faaliyet alanı {company.sector}.",
        Evidence(Fact(company.key, "name", company.name), (company.name,)),
    )
    if len(shares) > 1 and rng.random() < 0.5:
        entries = [
            (mention(rng, names, share.owner), rng.choice(percent_formats(share.percent)))
            for share in shares
        ]
        listed = ", ".join(f"{name} ({percent})" for name, percent in entries)
        doc.add(
            f"{first} için sicile işlenen ortak ve pay dökümü: {listed}.",
            *(
                Evidence(Fact(company.key, "owner", share.owner.key), (first, name, percent))
                for share, (name, percent) in zip(shares, entries, strict=True)
            ),
        )
    else:
        for share in shares:
            text, evidence = ownership_passage(rng, share, names, owned_name=first)
            doc.add(text, evidence)
    return doc


def render_report(
    rng: random.Random,
    owner: Company,
    shares: list[Share],
    names: dict[str, tuple[str, ...]],
    count_tokens: TokenCount,
) -> Document:
    doc = Document("annual report", owner.report)
    first = mention(rng, names, owner)
    doc.add(f"# {first} 2025 Faaliyet Raporu")
    sales = rng.randint(25, 2_700)
    profit = rng.randint(1, max(1, sales // 6))
    employees = rng.randint(35, 9_400)
    doc.add("## Finansal göstergeler")
    doc.add(
        rng.choice(
            (
                "Yılın kapanışında ciro {sales} milyon TL, net kâr {profit} milyon "
                "TL olarak hesaplandı. "
                "Dönem sonundaki çalışan sayısı {employees} kişiydi.",
                "Yıllık gelir tablosunda {sales} milyon TL ciro ve {profit} milyon TL dönem kârı "
                "raporlandı. İnsan kaynakları kayıtlarına göre yıl {employees} "
                "çalışanla tamamlandı.",
                "Yıl boyunca satışların toplamı {sales} milyon TL'ye ulaştı. "
                "{profit} milyon TL net kâr "
                "kaydedildi. Dönem raporuna esas personel mevcudu {employees} kişiydi.",
            )
        ).format(sales=sales, profit=profit, employees=employees)
    )
    sections = iter(rng.sample(REPORT_SECTIONS, len(REPORT_SECTIONS)))
    for _ in range(rng.randint(2, 5)):
        heading, body = next(sections)
        doc.add(f"## {heading}\n\n{body} {rng.choice(REPORT_NOTES)}")
    if shares:
        doc.add("## Pay portföyü")
        for share in shares:
            text, evidence = ownership_passage(rng, share, names, owner_name=first)
            doc.add(text, evidence)
        doc.add("## Birim görüşmeleri")
        for share in shares:
            short = mention(rng, names, share.owned, later=True)
            doc.add(
                rng.choice(
                    (
                        "{name} için sonraki dönem hazırlıkları bakım programı "
                        "üzerinden görüşüldü.",
                        "{name} ekibinden gelen çalışma notları dönem kapanışında incelendi.",
                        "İzleme toplantılarında {name} için uygulama kayıtlarının "
                        "güncellenmesi ele alındı.",
                    )
                ).format(name=short)
            )
    target = rng.randint(1_550, 1_900)
    for heading, body in sections:
        if count_tokens(doc.text) >= target:
            break
        doc.add(f"## {heading}\n\n{body} {rng.choice(REPORT_NOTES)}")
    require(1_500 <= count_tokens(doc.text) <= 3_000, f"annual report length: {doc.filename}")
    return doc


def render_contract(
    rng: random.Random,
    contract: Contract,
    names: dict[str, tuple[str, ...]],
    count_tokens: TokenCount,
) -> Document:
    doc = Document("contract", contract.document)
    name = contract.name
    doc.add(
        f"# {name}",
        Evidence(Fact(contract.number, "name", name), (name,)),
        Evidence(Fact(contract.number, "numara", contract.number), (contract.number,)),
        Evidence(Fact(contract.number, "tur", contract.kind), (contract.title,)),
    )
    first, second = (mention(rng, names, company) for company in contract.parties)
    left, right = (ROLE_NAMES[role] for role in contract.roles)
    signed = rng.choice(date_formats(contract.signed))
    text = rng.choice(
        (
            "{num} numaralı metin {date} tarihinde {a} ({left}) ile {b} ({right}) "
            "arasında imzalandı.",
            "{num} numaralı sözleşmenin imza tarihi {date}. İmza sayfasında {right} olarak {b}, "
            "{left} olarak {a} yer alır.",
            "{date} tarihli imzalar, {num} numaralı sözleşmede {left} sıfatıyla {a} ve {right} "
            "sıfatıyla {b} adına atıldı.",
        )
    ).format(num=contract.number, date=signed, a=first, b=second, left=left, right=right)
    parties = tuple(
        Evidence(Fact(contract.number, role, company.key), (contract.number, mention_text, label))
        for role, company, mention_text, label in zip(
            contract.roles, contract.parties, (first, second), (left, right), strict=True
        )
    )
    doc.add(
        f"## İmza ve taraflar\n\n{text}",
        *parties,
        Evidence(Fact(contract.number, "imza_tarihi", contract.signed.isoformat()), (signed,)),
    )
    clauses = list(contract.clauses)
    rng.shuffle(clauses)
    boilerplate = iter(rng.sample(CONTRACT_BOILERPLATE, len(CONTRACT_BOILERPLATE)))
    notes = iter(rng.choices(CONTRACT_NOTES, k=len(CONTRACT_BOILERPLATE)))

    def add_clause(clause: Clause) -> None:
        shown = rng.choice(clause.formats())
        text = rng.choice(clause.statements).format(v=shown)
        doc.add(
            f"## {clause.heading}\n\n{text}",
            Evidence(Fact(contract.number, clause.field, str(clause.value)), (shown,)),
        )

    for index, (heading, body) in enumerate(boilerplate):
        doc.add(f"## {heading}\n\n{body}" + (f" {next(notes)}" if rng.random() < 0.3 else ""))
        if index in (1, 3, 5) and len(clauses) > 1:
            add_clause(clauses.pop())
        if count_tokens(doc.text) >= 1_015:
            break
    require(count_tokens(doc.text) >= 1_000, f"late clause placement: {doc.filename}")
    for clause in clauses:
        add_clause(clause)
    require(1_000 < count_tokens(doc.text) <= 1_500, f"contract length: {doc.filename}")
    return doc


def render_documents(
    rng: random.Random,
    companies: list[Company],
    shares: list[Share],
    contracts: list[Contract],
    names: dict[str, tuple[str, ...]],
    count_tokens: TokenCount,
) -> list[Document]:
    report_only = set(rng.sample(shares, round(0.3 * len(shares))))
    incoming: dict[str, list[Share]] = defaultdict(list)
    outgoing: dict[str, list[Share]] = defaultdict(list)
    for share in shares:
        outgoing[share.owner.key].append(share)
        if share not in report_only:
            incoming[share.owned.key].append(share)
    docs = [render_registry(rng, c, incoming[c.key], names) for c in companies]
    docs.extend(
        render_report(rng, c, outgoing[c.key], names, count_tokens)
        for c in companies
        if c.holding or outgoing[c.key]
    )
    docs.extend(render_contract(rng, c, names, count_tokens) for c in contracts)
    for index in range(round(len(docs) / 9)):
        company = rng.choice(companies)
        heading, body = rng.choice(NOISE)
        doc = Document("noise", f"gundem_{index + 1:03d}.md")
        first = mention(rng, names, company)
        doc.add(f"# {heading} · {index + 1:03d}\n\n{body.format(name=first)}")
        docs.append(doc)
    return docs


@dataclass
class World:
    companies: list[Company]
    shares: list[Share]
    contracts: list[Contract]
    names: dict[str, tuple[str, ...]] = field(init=False)
    literals: dict[Fact, tuple[tuple[str, ...], ...]] = field(init=False, default_factory=dict)
    answers: dict[Fact, list[str]] = field(init=False, default_factory=dict)
    links: dict[tuple[str, str], list[tuple[str, Fact]]] = field(
        init=False, default_factory=lambda: defaultdict(list)
    )

    def __post_init__(self) -> None:
        self.names = name_variants(self.companies)
        for company in self.companies:
            key = company.key
            self.record(Fact(key, "name", company.name), (company.name,))
            self.record(Fact(key, "vergi_no", key), (key,), self.names[key])
            self.record(Fact(key, "sehir", company.city), (company.city,), self.names[key])
        incoming: dict[str, list[Share]] = defaultdict(list)
        for share in self.shares:
            owner, owned = share.owner.key, share.owned.key
            fact = Fact(owned, "owner", owner)
            self.record(fact, self.names[owner], self.names[owned], percent_formats(share.percent))
            self.links[owner, "child"].append((owned, fact))
            incoming[owned].append(share)
        for owned, shareholders in incoming.items():
            largest = max(share.percent for share in shareholders)
            leaders = [
                (s.owner.key, Fact(owned, "owner", s.owner.key))
                for s in shareholders
                if s.percent == largest
            ]
            require(len(leaders) == 1, f"no unique main shareholder for {owned}")
            self.links[owned, "main_owner"] = leaders
        for contract in self.contracts:
            number = contract.number
            self.record(Fact(number, "name", contract.name), (contract.name,))
            self.record(Fact(number, "numara", number), (number,))
            self.record(Fact(number, "tur", contract.kind), (contract.title,))
            self.record(
                Fact(number, "imza_tarihi", contract.signed.isoformat()),
                tuple(date_formats(contract.signed)),
            )
            for role, party in zip(contract.roles, contract.parties, strict=True):
                self.record(
                    Fact(number, role, party.key),
                    self.names[party.key],
                    (number,),
                    (ROLE_NAMES[role],),
                )
            for clause in contract.clauses:
                self.record(Fact(number, clause.field, str(clause.value)), tuple(clause.formats()))

    def record(self, fact: Fact, answer: tuple[str, ...], *literals: tuple[str, ...]) -> None:
        require(fact not in self.literals, f"repeated world fact: {fact}")
        self.literals[fact] = (answer, *literals)
        self.answers[fact] = list(answer)
        self.links[fact.subject, fact.field].append((fact.value, fact))

    def graph(self) -> dict[str, list[dict[str, object]]]:
        entities: list[dict[str, object]] = [
            {
                "type": "Sirket",
                "key": c.key,
                "name": c.name,
                "props": {"vergi_no": c.key, "sehir": c.city},
            }
            for c in self.companies
        ]
        entities.extend(
            {
                "type": "Sozlesme",
                "key": c.number,
                "name": c.name,
                "props": {"numara": c.number, "tur": c.kind, "imza_tarihi": c.signed.isoformat()},
            }
            for c in self.contracts
        )
        relations: list[dict[str, object]] = [
            {
                "type": "ORTAGIDIR",
                "source": s.owner.key,
                "target": s.owned.key,
                "props": {"oran": s.percent},
            }
            for s in self.shares
        ]
        relations.extend(
            {"type": "TARAFIDIR", "source": party.key, "target": c.number, "props": {"rol": role}}
            for c in self.contracts
            for role, party in zip(c.roles, c.parties, strict=True)
        )
        return {"entities": entities, "relations": relations}

    def follow(self, question: Question) -> list[tuple[str, list[Fact], list[str]]]:
        results = []
        for path in question.paths:
            states: list[tuple[str, list[Fact]]] = [(question.subject, [])]
            for step in path:
                next_states: list[tuple[str, list[Fact]]] = []
                for node, chain in states:
                    links = self.links.get((node, step), [])
                    require(
                        step == "child" or len(links) <= 1,
                        f"ambiguous step {step} in {question.text}",
                    )
                    next_states.extend((target, [*chain, fact]) for target, fact in links)
                states = next_states
            if not states:
                return []
            for target, chain in states:
                answers = (
                    self.names[target] if path[-1] == "main_owner" else self.answers[chain[-1]]
                )
                results.append((target, chain, list(answers)))
        return results


def check_evidence(world: World, documents: list[Document]) -> dict[Fact, list[SourceEvidence]]:
    sources: dict[Fact, list[SourceEvidence]] = defaultdict(list)
    require(len({d.filename for d in documents}) == len(documents), "duplicate document filename")
    require(len({d.text for d in documents}) == len(documents), "duplicate document content")
    for doc in documents:
        text = doc.text
        for statement, evidence in doc.claims:
            fact = evidence.fact
            require(fact in world.literals, f"unexpected fact in {doc.filename}: {fact}")
            require(
                statement in text and all(literal in statement for literal in evidence.literals),
                f"fact text missing in {doc.filename}: {fact}",
            )
            literals = set(evidence.literals)
            require(
                all(literals.intersection(group) for group in world.literals[fact]),
                f"fact value or endpoint missing in {doc.filename}: {fact}",
            )
            sources[fact].append({"doc": doc.filename, "literals": list(evidence.literals)})
    missing = world.literals.keys() - sources.keys()
    require(not missing, f"unstated facts: {sorted(map(str, missing))}")
    return sources


def chain_step(
    fact: Fact,
    shares: dict[tuple[str, str], int],
    clause_fields: set[str],
    sources: list[SourceEvidence],
) -> dict[str, object]:
    step: dict[str, object]
    if fact.field == "owner":
        step = {
            "kind": "relation",
            "type": "ORTAGIDIR",
            "source": fact.value,
            "target": fact.subject,
            "props": {"oran": shares[fact.value, fact.subject]},
        }
    elif fact.field in ROLE_NAMES:
        step = {
            "kind": "relation",
            "type": "TARAFIDIR",
            "source": fact.value,
            "target": fact.subject,
            "props": {"rol": fact.field},
        }
    elif fact.field in ONTOLOGY_PROPERTIES:
        step = {"kind": "property", "entity": fact.subject, "prop": fact.field, "value": fact.value}
    elif fact.field in clause_fields:
        step = {"kind": "text", "entity": fact.subject, "field": fact.field, "value": fact.value}
    else:
        raise ValueError(f"unsupported chain fact: {fact}")
    evidence = [
        {"doc": source["doc"], "literals": list(source["literals"])}
        for source in sorted(sources, key=lambda source: source["doc"])
    ]
    return {**step, "evidence": evidence}


def candidates(world: World, rng: random.Random) -> dict[str, list[Question]]:
    groups: dict[str, list[Question]] = {group: [] for group in GROUPS}
    for company in world.companies:
        key = company.key
        name = company.name
        short = rng.choice(world.names[key][2:]) if len(world.names[key]) > 2 else None

        def company_question(
            group: str,
            family: str,
            question: str,
            path: tuple[str, ...],
            *,
            key: str = key,
            name: str = name,
            short: str | None = short,
        ) -> None:
            groups[group].append(
                Question(
                    key,
                    family,
                    question.format(name=name),
                    (path,),
                    question.format(name=short) if short else None,
                )
            )

        company_question(
            "single_hop",
            "company tax",
            rng.choice(
                (
                    "“{name}” için sicilde hangi vergi numarası kayıtlı?",
                    "Vergi kimlik numarası aranan şirket “{name}”. Sicilde hangi numara yazıyor?",
                )
            ),
            ("vergi_no",),
        )
        company_question(
            "single_hop",
            "company city",
            rng.choice(
                (
                    "“{name}” şirketinin merkezi hangi şehirde?",
                    "Sicile göre “{name}” şirketinin merkez ili neresi?",
                )
            ),
            ("sehir",),
        )
        company_question(
            "multi_hop",
            "owner city",
            "“{name}” şirketinde en yüksek payı elinde tutan ortağın merkezi nerede?",
            ("main_owner", "sehir"),
        )
        company_question(
            "multi_hop",
            "owner tax",
            "“{name}” şirketinin en büyük hissedarının vergi numarası nedir?",
            ("main_owner", "vergi_no"),
        )
        company_question(
            "deep_hop",
            "ancestor city",
            "“{name}” şirketinin en büyük hissedarına bakın. O hissedarın en büyük "
            "ortağının merkezi hangi şehirde?",
            ("main_owner", "main_owner", "sehir"),
        )
        company_question(
            "deep_hop",
            "ancestor tax",
            "“{name}” şirketindeki en yüksek paylı ortağın en yüksek paylı ortağı "
            "sicilde hangi vergi numarasıyla kayıtlı?",
            ("main_owner", "main_owner", "vergi_no"),
        )
        company_question(
            "deep_hop",
            "third ancestor",
            "“{name}” şirketinden başlayıp her adımda en büyük hissedara geçin. "
            "Üçüncü adımda ulaşılan şirket hangisi?",
            ("main_owner", "main_owner", "main_owner"),
        )
        if company.holding:
            company_question(
                "deep_hop",
                "indirect cities",
                "“{name}” şirketinin tek bir aracı şirket üzerinden dolaylı pay "
                "sahibi olduğu şirketlerin merkezleri hangi şehirlerdedir? "
                "Her şehri bir kez yazın.",
                ("child", "child", "sehir"),
            )
    for contract in world.contracts:
        number = contract.number
        groups["single_hop"].append(
            Question(
                number,
                "signature date",
                rng.choice(
                    (
                        f"{number} numaralı sözleşme hangi tarihte imzalandı?",
                        f"İmza tarihi aranan sözleşme {number}. Sözleşme ne zaman imzalanmış?",
                    )
                ),
                (("imza_tarihi",),),
            )
        )
        left, right = contract.roles
        groups["single_hop"].append(
            Question(
                number,
                "both parties",
                f"{number} numaralı sözleşmenin iki tarafı kimlerdir?",
                ((left,), (right,)),
            )
        )
        for role in contract.roles:
            label = ROLE_NAMES[role]
            for field_name, family, text in (
                (
                    "sehir",
                    "party city",
                    f"{number} sözleşmesinde {label} sıfatıyla imza atan şirketin "
                    "merkez ili neresi?",
                ),
                (
                    "vergi_no",
                    "party tax",
                    f"{number} numaralı sözleşmedeki {label} şirketin vergi kimlik numarası nedir?",
                ),
                (
                    "main_owner",
                    "party owner",
                    f"{number} sözleşmesinin {label} tarafında yer alan şirketin en "
                    "büyük hissedarı kimdir?",
                ),
            ):
                groups["multi_hop"].append(Question(number, family, text, ((role, field_name),)))
            groups["deep_hop"].append(
                Question(
                    number,
                    "party ancestor",
                    rng.choice(
                        (
                            f"{number} sözleşmesindeki {label} şirketin en büyük hissedarının "
                            "en büyük hissedarı hangi şirkettir?",
                            f"{number} sözleşmesinin {label} tarafını bulun. Bu "
                            "tarafın en yüksek paylı "
                            "ortağında en büyük pay kime aittir?",
                        )
                    ),
                    ((role, "main_owner", "main_owner"),),
                )
            )
        for clause in contract.clauses:
            groups["free_text"].append(
                Question(
                    number,
                    clause.field,
                    f"{number} numaralı metne göre "
                    f"{clause.question[0].lower()}{clause.question[1:]}",
                    ((clause.field,),),
                )
            )
    return groups


def balanced_sample(rng: random.Random, questions: list[Question], count: int) -> list[Question]:
    families: dict[str, list[Question]] = defaultdict(list)
    for question in questions:
        families[question.family].append(question)
    order = list(families)
    rng.shuffle(order)
    for queue in families.values():
        rng.shuffle(queue)
    selected: list[Question] = []
    while len(selected) < count and order:
        remaining = []
        for family in order:
            selected.append(families[family].pop())
            if len(selected) == count:
                break
            if families[family]:
                remaining.append(family)
        order = remaining
    return selected


def make_questions(
    rng: random.Random,
    world: World,
    count: int,
    documents: list[Document],
    sources: dict[Fact, list[SourceEvidence]],
) -> dict[str, list[QuestionRow]]:
    document_text = {doc.filename: doc.text for doc in documents}
    shares = {(share.owner.key, share.owned.key): share.percent for share in world.shares}
    clause_fields = {clause.field for contract in world.contracts for clause in contract.clauses}
    groups: dict[str, list[QuestionRow]] = {}
    per_group, remainder = divmod(count, len(GROUPS))
    seen: set[str] = set()
    for index, (group, questions) in enumerate(candidates(world, rng).items()):
        quota = per_group + (index < remainder)
        viable = [q for q in questions if world.follow(q)]
        require(
            len(viable) >= quota,
            f"{group}: requested {quota} questions, only {len(viable)} unique questions available",
        )
        aliases = balanced_sample(rng, [q for q in viable if q.alias_text], round(quota * 4 / 9))
        alias_set = set(aliases)
        full = balanced_sample(rng, [q for q in viable if q not in alias_set], quota - len(aliases))
        selected = aliases + full
        rng.shuffle(selected)
        groups[group] = []
        for question in selected:
            text = question.alias_text if question in alias_set else question.text
            assert text is not None
            require(norm(text) not in seen, f"repeated question: {text}")
            seen.add(norm(text))
            paths = world.follow(question)
            require(bool(paths), f"incomplete question: {text}")
            answer: dict[str, list[str]] = {}
            chains: dict[str, list[dict[str, object]]] = {}
            docs: dict[str, None] = {}
            for target, path, alternatives in paths:
                answer[target] = alternatives
                steps = []
                for fact in path:
                    evidence = sources.get(fact, [])
                    require(bool(evidence), f"missing hop {fact}: {text}")
                    docs[evidence[0]["doc"]] = None
                    steps.append(chain_step(fact, shares, clause_fields, evidence))
                # Follow the answer's deduplication when several paths yield the same target.
                chains[target] = steps
            for alternatives in answer.values():
                require(
                    any(matches(document_text[doc], [alternatives]) for doc in docs),
                    f"unstated answer group {alternatives}: {text}",
                )
            for chain in chains.values():
                for step in chain:
                    evidence = cast(list[SourceEvidence], step["evidence"])
                    require(
                        any(source["doc"] in docs for source in evidence),
                        f"chain step without evidence in question docs: {text}: {step}",
                    )
            groups[group].append(
                {
                    "q": text,
                    "docs": list(docs),
                    "answer": list(answer.values()),
                    "chain": list(chains.values()),
                }
            )
    return groups


def readme(
    seed: int,
    world: World,
    documents: list[Document],
    questions: dict[str, list[QuestionRow]],
    tokens: int,
) -> str:
    kinds = Counter(doc.kind for doc in documents)
    by_kind = ", ".join(f"{n} {kind}" for kind, n in kinds.items())
    company_count = len(world.companies)
    contract_count = len(world.contracts)
    question_count = sum(map(len, questions.values()))
    # A stable example destination keeps the README identical for copies elsewhere.
    out = (
        SET_DEFAULT
        if (seed, company_count, contract_count, question_count) == (7, 150, 300, 160)
        else Path("synthetic-set")
    )
    command = (
        "python eval/synthetic/generate.py --out "
        f"{shlex.quote(out.as_posix())} --seed {seed} --companies {company_count} "
        f"--contracts {contract_count} --questions {question_count}"
    )
    return "\n".join(
        [
            "# Synthetic Turkish evaluation set",
            "",
            "Reproduce this set with the following command.",
            "",
            "```bash",
            command,
            "```",
            "",
            f"- Seed {seed}",
            f"- {len(documents)} documents ({by_kind})",
            f"- {len(world.companies) + len(world.contracts)} entities "
            f"({len(world.companies)} Sirket, {len(world.contracts)} Sozlesme)",
            f"- {len(world.shares) + 2 * len(world.contracts)} relations "
            f"({len(world.shares)} ORTAGIDIR, {2 * len(world.contracts)} TARAFIDIR)",
            f"- {sum(map(len, questions.values()))} questions "
            f"({', '.join(f'{len(items)} {group}' for group, items in questions.items())})",
            f"- {tokens:,} document tokens (cl100k_base, counted per file)",
            "",
            "Use `--hops 3` with the evaluation runner to allow three graph hops.",
            "",
        ]
    )


def generate(
    out: Path, seed: int, company_count: int, contract_count: int, question_count: int
) -> None:
    require(
        contract_count > 0 and question_count >= 4,
        "provide at least one contract and four questions",
    )
    rng = random.Random(seed)
    encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(encoding.encode_ordinary(text))

    companies = make_companies(rng, company_count)
    world = World(
        companies, make_shares(rng, companies), make_contracts(rng, contract_count, companies)
    )
    documents = render_documents(
        rng, companies, world.shares, world.contracts, world.names, count_tokens
    )
    sources = check_evidence(world, documents)
    questions = make_questions(rng, world, question_count, documents, sources)
    tokens = sum(count_tokens(doc.text) for doc in documents)
    if company_count == 150 and contract_count == 300:
        require(tokens >= 150_000, f"default corpus is too small: {tokens} tokens")
    ontology = (GOLDEN / "ontology.yaml").read_bytes()

    doc_dir = out / "docs"
    readme_path = out / "README.md"
    prior_docs = set(doc_dir.glob("*.md"))
    require(
        not prior_docs
        or (
            readme_path.is_file()
            and readme_path.read_text(encoding="utf-8").startswith(
                "# Synthetic Turkish evaluation set\n"
            )
        ),
        f"output already contains documents from another corpus: {out}",
    )
    doc_dir.mkdir(parents=True, exist_ok=True)
    for doc in documents:
        (doc_dir / doc.filename).write_text(doc.text, encoding="utf-8", newline="\n")
    (out / "ontology.yaml").write_bytes(ontology)
    (out / "questions.yaml").write_text(
        yaml.safe_dump(questions, sort_keys=False, allow_unicode=True, width=100),
        encoding="utf-8",
        newline="\n",
    )
    (out / "expected_graph.json").write_text(
        json.dumps(world.graph(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    readme_path.write_text(
        readme(seed, world, documents, questions, tokens), encoding="utf-8", newline="\n"
    )
    for stale in prior_docs - {doc_dir / doc.filename for doc in documents}:
        stale.unlink()
    print(
        f"Generated {len(documents)} documents, {question_count} questions, "
        f"{tokens:,} tokens in {out}"
    )


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--companies", type=int, default=150)
    parser.add_argument("--contracts", type=int, default=300)
    parser.add_argument("--questions", type=int, default=160)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        generate(args.out, args.seed, args.companies, args.contracts, args.questions)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
