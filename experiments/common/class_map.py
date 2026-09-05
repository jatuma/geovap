"""Mapovaci tabulka JVF kod -> nejblizsi verejna trida, podle SS5.2 03_semanticka_segmentace.md,
rozsirena o vsechny kody skutecne pritomne v Drazkove (viz e0_data_sanity.py vystup).

`public` je orientacni nazev tridy z Cityscapes/ADE20k/Mapillary Vistas taxonomie (podle toho,
co konkretni pouzity model umi) -- pouzito jen jako navrh pro E3 skorovani, ne jako zavazny
seznam trid pro produkci (ten je v SS2 03_semanticka_segmentace.md).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ClassInfo:
    name_cz: str
    public: str | None  # None = zadny dobry verejny ekvivalent (vlastni trida)
    note: str = ""


JVF_CLASSES: dict[str, ClassInfo] = {
    "0100000304": ClassInfo("hranice dopravní stavby nebo plochy", "road|sidewalk|curb", "nejcetnejsi trida v Drazkove"),
    "0100000162": ClassInfo("plot", None, "vlastni trida, 5 materialu; nejblizsi obecny Fence"),
    "0100000300": ClassInfo("hranice stavby", "building", ""),
    "0100000299": ClassInfo("hranice budovy", "building", ""),
    "0100000165": ClassInfo("stavebně upravený vjezd na pozemek", "driveway", "jen Vistas, licencne blokovane"),
    "0100000215": ClassInfo("udržovaná plocha zeleně", "vegetation", ""),
    "0100000001": ClassInfo("budova", "building", ""),
    "0100000051": ClassInfo("příkop, násep, zářez dopravní stavby", None, "geometricky jev, ne fotometricky"),
    "0100000305": ClassInfo("hranice přírodního a polopřírodního objektu", "terrain|vegetation", ""),
    "0100000308": ClassInfo("hranice udržované zeleně", "vegetation", ""),
    "0100000320": ClassInfo("přidružená plocha pozemní komunikace", "road", ""),
    "0100000202": ClassInfo("neidentifikovaný objekt", None, "unikovy poklop, nelze mapovat"),
    "0100000209": ClassInfo("zahrada", None, "vyuziti uzemi, ne vzhled - nesegmentovat z obrazu"),
    "0100000183": ClassInfo("patka, deska, monolit, pilíř", None, ""),
    "0100000017": ClassInfo("nájezd", "driveway", ""),
    "0100000306": ClassInfo("hranice vodního díla", "water", ""),
    "0100000187": ClassInfo("stavba pro zpevnění povrchu", None, ""),
    "0100000302": ClassInfo("hranice zdi", "wall", ""),
    "0100000314": ClassInfo("ostatní zastřešená stavba", "building", ""),
    "0100000193": ClassInfo("čelo propustku", None, "vzacna trida, jen ~56 instanci"),
    "0100000323": ClassInfo("vnitřní členění dopravní plochy", "road", ""),
    "0100000189": ClassInfo("dvůr, nádvoří", "ground", ""),
    "0100000084": ClassInfo("studna na veřejném prostranství", None, "vlastni trida, vzacna"),
    "0100000195": ClassInfo("průběh propustku", None, ""),
    "0100000007": ClassInfo("chodník", "sidewalk", ""),
    "0100000217": ClassInfo("terénní hrana", None, "geometricky jev"),
    "0100000207": ClassInfo("zemědělská plocha", None, "fuzovat s LPIS, nesegmentovat"),
    "0100000078": ClassInfo("stavebně upravené koryto", "water", ""),
    "0100000203": ClassInfo("vodní tok", "water", ""),
    "0100000055": ClassInfo("manipulační plocha", "ground", ""),
    "0100000199": ClassInfo("zábradlí", "guard_rail", "vzacna trida, jen ~8 instanci; pozor - verejny guard_rail je spis svodidlo"),
    "0100000168": ClassInfo("zeď", "wall", ""),
    "0100000080": ClassInfo("meliorační příkop, žlab", None, ""),
    "0100000301": ClassInfo("hranice schodiště", "stairs", ""),
    "0100000166": ClassInfo("schodiště", "stairs", ""),
    "0100000213": ClassInfo("hospodářsky nevyužívaná plocha", None, ""),
    "0100000179": ClassInfo("skleník", "building", ""),
    "0100000201": ClassInfo("nosič technického zařízení", "utility_pole", ""),
    "0100000011": ClassInfo("parkoviště, odstavná plocha", "parking", ""),
    "0100000083": ClassInfo("meliorační šachta", "manhole", "jen Vistas, licencne blokovane"),
    "0100000307": ClassInfo("hranice ostatní plochy", "ground", ""),
    "0100000310": ClassInfo("vnitřní členění budov a staveb", "building", ""),
    "0100000005": ClassInfo("provozní plocha pozemní komunikace", "road", ""),
    "0100000058": ClassInfo("plocha mostní konstrukce", "bridge", ""),
    "0100000330": ClassInfo("nádrž bez vzdouvacího objektu", "water", ""),
    "0100000159": ClassInfo("drobná kulturní stavba", "building", ""),
    "0100000154": ClassInfo("drobná sakrální stavba", "building", ""),
    "0100000152": ClassInfo("hřiště", "ground", ""),
    "0100000218": ClassInfo("podrobný bod ZPS", None, "geodeticky bod, nema vizualni pritomnost"),
    "0100000219": ClassInfo("výškový bod na terénu", None, "geodeticky bod"),
    "0100000220": ClassInfo("identický bod", None, "geodeticky bod"),
}


def lookup(jvfcode_base: str) -> ClassInfo | None:
    return JVF_CLASSES.get(jvfcode_base)
