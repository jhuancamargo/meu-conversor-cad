"""
Conversor OpenStreetMap -> DXF georreferenciado.

Substitui o CadMapper para contexto de implantacao: a arquiteta escolhe um
retangulo no mapa e recebe um DXF em metros, em UTM local, com o entorno
separado por layers (edificios, vias, caminhos, agua) e curvas de nivel.

Fontes:
- Geometria: Overpass API (OpenStreetMap), dados da comunidade.
- Terreno:   OpenTopoData (EU-DEM 25m na Europa, SRTM 30m no resto).

AVISO DE PRECISAO: as curvas vem de modelo de elevacao de 25-30m. Servem
para CONTEXTO e estudo de massa, NAO substituem levantamento topografico.
E modelo de superficie: vegetacao e telhados influenciam a cota.
"""
import math
import time

import ezdxf
import numpy as np
import requests
from pyproj import CRS, Transformer

DXF_VERSION = "R2010"
UA = {"User-Agent": "ConversorCAD-OSM/1.0 (uso profissional de arquitetura)"}

# Varios espelhos: o servico e comunitario e fica sobrecarregado.
# Se um devolver 429/504, tenta-se o seguinte.
OVERPASS_ESPELHOS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
ELEVACAO = "https://api.opentopodata.org/v1"

# Area maxima do recorte. Protege a API publica do Overpass, que e um
# servico comunitario gratuito -- pedidos enormes sao abuso e sao cortados.
MAX_AREA_KM2 = 25.0

# A API publica de elevacao aceita ~100 pontos por pedido.
ELEV_LOTE = 100

# Caminhos a pe/trilhos ficam separados das vias de carro.
TAGS_CAMINHO = {"footway", "path", "track", "steps", "cycleway",
                "pedestrian", "bridleway"}

LAYERS = {
    "OSM_BUILDINGS":           1,   # vermelho
    "OSM_ROADS":               3,   # verde
    "OSM_PATHS":               4,   # ciano
    "OSM_WATER":               5,   # azul
    "SITE_BOUNDARIES":         6,   # magenta
    "TERRAIN_CONTOURS_MINOR":  8,   # cinza
    "TERRAIN_CONTOURS_MAJOR":  2,   # amarelo
    "EXPORT_BOUNDARY":         7,   # branco
    "LABELS":                  9,
}


def utm_para_bbox(sul, oeste, norte, este):
    """Escolhe a zona UTM pelo centro do recorte. Devolve (CRS, epsg)."""
    lat = (sul + norte) / 2.0
    lon = (oeste + este) / 2.0
    zona = int((lon + 180) // 6) + 1
    epsg = (32600 if lat >= 0 else 32700) + zona
    return CRS.from_epsg(epsg), epsg


def area_km2(sul, oeste, norte, este):
    """Area aproximada do recorte, em km2."""
    alt = (norte - sul) * 111.32
    larg = (este - oeste) * 111.32 * math.cos(math.radians((sul + norte) / 2))
    return abs(alt * larg)


def _overpass(sul, oeste, norte, este, camadas):
    """Pede ao Overpass so as camadas escolhidas, numa unica consulta.

    Nota: dividir em varias consultas (uma por tema) parecia mais seguro,
    mas cada pedido ao servidor publico tem ~9s de latencia de fila --
    5 pedidos sequenciais levavam >1min. Uma consulta so custa o mesmo
    ~3-9s uma unica vez. Evitamos 'relation' (pesada e rara para este uso)
    para nao provocar 504."""
    bbox = f"{sul},{oeste},{norte},{este}"
    partes = []
    if "buildings" in camadas:
        partes.append(f'way["building"]({bbox});')
    if "roads" in camadas or "paths" in camadas:
        partes.append(f'way["highway"]({bbox});')
    if "water" in camadas:
        partes.append(f'way["waterway"]({bbox});')
        partes.append(f'way["natural"="water"]({bbox});')
    if "boundaries" in camadas:
        partes.append(f'way["landuse"]({bbox});')
        partes.append(f'way["boundary"]({bbox});')
    if not partes:
        return []

    consulta = f"[out:json][timeout:50];({''.join(partes)});out geom;"

    ultimo = None
    for espelho in OVERPASS_ESPELHOS:
        try:
            r = requests.post(espelho, data={"data": consulta},
                              timeout=70, headers=UA)
            if r.status_code in (429, 504):
                ultimo = r.status_code
                time.sleep(1.5)
                continue
            r.raise_for_status()
            return r.json().get("elements", [])
        except requests.RequestException as exc:
            ultimo = exc
            time.sleep(1.5)
            continue

    if ultimo in (429, 504):
        raise RuntimeError("Os servidores do OpenStreetMap estão ocupados "
                           "neste momento. Espere um minuto e tente de novo, "
                           "ou desenhe uma área menor.")
    raise RuntimeError("Não consegui falar com o OpenStreetMap. "
                       "Verifique a ligação à internet.")


def _classifica(tags, camadas):
    """Diz em que layer a geometria entra, e se fecha o contorno."""
    if "building" in tags and "buildings" in camadas:
        return "OSM_BUILDINGS", True
    if ("waterway" in tags or tags.get("natural") == "water") and "water" in camadas:
        return "OSM_WATER", tags.get("natural") == "water"
    if ("landuse" in tags or "boundary" in tags) and "boundaries" in camadas:
        return "SITE_BOUNDARIES", True
    via = tags.get("highway")
    if via:
        if via in TAGS_CAMINHO:
            return ("OSM_PATHS", False) if "paths" in camadas else (None, False)
        return ("OSM_ROADS", False) if "roads" in camadas else (None, False)
    return None, False


def _elevacao(lats, lons, dataset):
    """Busca a grade de elevacao em lotes. Devolve matriz (nan onde nao ha dado)."""
    pontos = [(la, lo) for la in lats for lo in lons]
    alturas = []
    for i in range(0, len(pontos), ELEV_LOTE):
        lote = pontos[i:i + ELEV_LOTE]
        locs = "|".join(f"{la},{lo}" for la, lo in lote)
        r = requests.post(f"{ELEVACAO}/{dataset}", data={"locations": locs},
                          timeout=60, headers=UA)
        if not r.ok:
            raise RuntimeError("Não consegui obter os dados de terreno. "
                               "Tente novamente em alguns segundos.")
        dados = r.json()
        if "results" not in dados:
            raise RuntimeError("O serviço de terreno recusou o pedido. "
                               "Tente uma área menor.")
        alturas += [p.get("elevation") for p in dados["results"]]
        time.sleep(1.1)   # o servico publico pede 1 pedido/segundo
    Z = np.array([np.nan if a is None else float(a) for a in alturas])
    return Z.reshape(len(lats), len(lons))


def osm_para_dxf(sul, oeste, norte, este, saida,
                 camadas=("buildings", "roads", "paths", "water", "boundaries"),
                 curvas=True, intervalo=1.0, intervalo_maior=5.0,
                 grade=12, epsg_manual=None):
    """Converte um recorte do mapa em DXF. Medidas em metros."""
    area = area_km2(sul, oeste, norte, este)
    if area > MAX_AREA_KM2:
        return {"error": f"A área selecionada tem {area:.1f} km². "
                         f"O máximo é {MAX_AREA_KM2:.0f} km² — desenhe um "
                         f"retângulo menor."}
    if area <= 0:
        return {"error": "Desenhe um retângulo no mapa antes de exportar."}

    if epsg_manual:
        crs, epsg = CRS.from_epsg(int(epsg_manual)), int(epsg_manual)
    else:
        crs, epsg = utm_para_bbox(sul, oeste, norte, este)
    para_metros = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    doc = ezdxf.new(dxfversion=DXF_VERSION)
    doc.header["$INSUNITS"] = 6                 # 6 = metros
    msp = doc.modelspace()
    for nome, cor in LAYERS.items():
        doc.layers.add(nome, color=cor)

    contagem = {k: 0 for k in LAYERS}
    aviso = None

    # --- geometria do OpenStreetMap ---
    try:
        elementos = _overpass(sul, oeste, norte, este, camadas)
    except RuntimeError as exc:
        return {"error": str(exc)}
    except requests.RequestException:
        return {"error": "Não consegui falar com o OpenStreetMap. "
                         "Verifique a ligação à internet."}

    for el in elementos:
        geo = el.get("geometry")
        if not geo:
            continue
        layer, fechar = _classifica(el.get("tags", {}), camadas)
        if not layer:
            continue
        pts = [para_metros.transform(p["lon"], p["lat"]) for p in geo]
        if len(pts) < 2:
            continue
        # OSM fecha o anel repetindo o 1o ponto; o DXF nao precisa
        if fechar and len(pts) > 2 and pts[0] == pts[-1]:
            pts = pts[:-1]
        if len(pts) < 2:
            continue
        msp.add_lwpolyline(pts, close=fechar, dxfattribs={"layer": layer})
        contagem[layer] += 1

    # --- curvas de nivel ---
    if curvas:
        try:
            import contourpy
            dataset = "eudem25m" if (35 < sul < 71 and -25 < oeste < 45) else "srtm30m"
            lats = np.linspace(sul, norte, grade)
            lons = np.linspace(oeste, este, grade)
            Z = _elevacao(lats, lons, dataset)

            vazios = int(np.isnan(Z).sum())
            if vazios == Z.size:
                aviso = "Não há dados de terreno para esta área."
            else:
                if vazios:
                    # tipicamente mar: cota 0
                    Z = np.nan_to_num(Z, nan=0.0)
                    aviso = (f"{vazios} ponto(s) sem dado de terreno "
                             f"(provavelmente mar) — tratados como cota 0.")
                X = np.zeros_like(Z)
                Y = np.zeros_like(Z)
                for i, la in enumerate(lats):
                    for j, lo in enumerate(lons):
                        X[i, j], Y[i, j] = para_metros.transform(lo, la)

                gen = contourpy.contour_generator(X, Y, Z)
                base = math.floor(float(np.nanmin(Z)))
                topo = float(np.nanmax(Z))
                for nivel in np.arange(base, topo + intervalo, intervalo):
                    maior = abs(nivel % intervalo_maior) < 1e-6
                    lay = ("TERRAIN_CONTOURS_MAJOR" if maior
                           else "TERRAIN_CONTOURS_MINOR")
                    for linha in gen.lines(float(nivel)):
                        if len(linha) >= 2:
                            msp.add_lwpolyline(linha, dxfattribs={"layer": lay})
                            contagem[lay] += 1
                            if maior:
                                meio = linha[len(linha) // 2]
                                t = msp.add_text(f"{nivel:.0f}", dxfattribs={
                                    "layer": "LABELS", "height": 2.0})
                                t.set_placement((float(meio[0]), float(meio[1])))
                                contagem["LABELS"] += 1
        except RuntimeError as exc:
            aviso = str(exc)
        except Exception:
            aviso = "Não consegui gerar as curvas de nível; o resto foi exportado."

    # --- retangulo do recorte ---
    cantos = [para_metros.transform(oeste, sul), para_metros.transform(este, sul),
              para_metros.transform(este, norte), para_metros.transform(oeste, norte)]
    msp.add_lwpolyline(cantos, close=True, dxfattribs={"layer": "EXPORT_BOUNDARY"})
    contagem["EXPORT_BOUNDARY"] = 1

    total = sum(contagem.values())
    if total <= 1:
        return {"error": "Não há dados do OpenStreetMap nesta área. "
                         "Tente outro local ou um retângulo maior."}

    # layers vazias so poluem o AutoCAD
    for nome, n in contagem.items():
        if n == 0 and nome in doc.layers:
            doc.layers.remove(nome)

    doc.saveas(saida)

    xs = [c[0] for c in cantos]
    ys = [c[1] for c in cantos]
    return {
        "entidades": total,
        "edificios": contagem["OSM_BUILDINGS"],
        "vias": contagem["OSM_ROADS"],
        "caminhos": contagem["OSM_PATHS"],
        "agua": contagem["OSM_WATER"],
        "limites": contagem["SITE_BOUNDARIES"],
        "curvas": (contagem["TERRAIN_CONTOURS_MINOR"]
                   + contagem["TERRAIN_CONTOURS_MAJOR"]),
        "epsg": epsg,
        "zona": crs.name,
        "medida": f"{max(xs)-min(xs):.0f} x {max(ys)-min(ys):.0f} m",
        "area_km2": round(area, 2),
        "aviso": aviso,
        "error": None,
    }


if __name__ == "__main__":
    import sys, json
    # Altea (Alicante) por omissao -- terreno Signature 01
    s, w, n, e = 38.5975, -0.0560, 38.6035, -0.0480
    if len(sys.argv) >= 5:
        s, w, n, e = (float(x) for x in sys.argv[1:5])
    saida = sys.argv[5] if len(sys.argv) > 5 else "osm.dxf"
    t = time.time()
    print(json.dumps(osm_para_dxf(s, w, n, e, saida), ensure_ascii=False, indent=2))
    print(f"{time.time()-t:.1f}s -> {saida}")
