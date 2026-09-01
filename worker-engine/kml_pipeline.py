"""
Conversor KML/KMZ -> DXF (georreferenciado em UTM/metros).

- Faz parse do KML (XML puro, sem libs externas).
- Suporta Point, LineString, Polygon (com furos/inner rings).
- Projeta lat/lon (WGS84) para UTM em metros (Transverse Mercator, precisao mm).
- Escreve DXF (ezdxf). O AutoCAD abre DXF e permite 'Salvar como .dwg'.

Coordenadas: mantem UTM real (georreferenciado). Cada Placemark vira uma
layer com o nome do local.
"""
import math, zipfile, re
import xml.etree.ElementTree as ET
import ezdxf

# --- WGS84 -> UTM (sem pyproj) -------------------------------------------
_A = 6378137.0            # semi-eixo maior WGS84
_F = 1 / 298.257223563    # achatamento
_K0 = 0.9996
_E2 = _F * (2 - _F)
_EP2 = _E2 / (1 - _E2)


def _utm_zone(lon, lat):
    zone = int((lon + 180) / 6) + 1
    return zone, (lat >= 0)


def latlon_to_utm(lat, lon, zone=None, north=None):
    if zone is None:
        zone, north = _utm_zone(lon, lat)
    lat_r = math.radians(lat)
    lon_r = math.radians(lon)
    lon0 = math.radians((zone - 1) * 6 - 180 + 3)

    N = _A / math.sqrt(1 - _E2 * math.sin(lat_r) ** 2)
    T = math.tan(lat_r) ** 2
    C = _EP2 * math.cos(lat_r) ** 2
    Aa = math.cos(lat_r) * (lon_r - lon0)
    M = _A * ((1 - _E2/4 - 3*_E2**2/64 - 5*_E2**3/256) * lat_r
              - (3*_E2/8 + 3*_E2**2/32 + 45*_E2**3/1024) * math.sin(2*lat_r)
              + (15*_E2**2/256 + 45*_E2**3/1024) * math.sin(4*lat_r)
              - (35*_E2**3/3072) * math.sin(6*lat_r))

    easting = (_K0 * N * (Aa + (1 - T + C) * Aa**3/6
               + (5 - 18*T + T**2 + 72*C - 58*_EP2) * Aa**5/120) + 500000.0)
    northing = (_K0 * (M + N * math.tan(lat_r) * (Aa**2/2
                + (5 - T + 9*C + 4*C**2) * Aa**4/24
                + (61 - 58*T + T**2 + 600*C - 330*_EP2) * Aa**6/720)))
    if not north:
        northing += 10000000.0
    return easting, northing, zone, north


# --- Parse KML ------------------------------------------------------------
def _localname(tag):
    return tag.split('}')[-1]


def _read_kml_bytes(path):
    if path.lower().endswith('.kmz'):
        with zipfile.ZipFile(path) as z:
            name = next((n for n in z.namelist() if n.lower().endswith('.kml')),
                        None)
            if not name:
                raise ValueError("KMZ sem arquivo .kml dentro")
            return z.read(name)
    with open(path, 'rb') as f:
        return f.read()


def _parse_coords(text):
    """'lon,lat,alt lon,lat,alt ...' -> [(lat, lon), ...]"""
    pts = []
    for tok in text.replace('\n', ' ').split():
        parts = tok.split(',')
        if len(parts) >= 2:
            try:
                lon = float(parts[0]); lat = float(parts[1])
                pts.append((lat, lon))
            except ValueError:
                continue
    return pts


def _find(el, name):
    return [c for c in el.iter() if _localname(c.tag) == name]


def kml_to_dxf(kml_path, output_path, local=False):
    """local=True desloca tudo para perto da origem (0,0), mantendo as
    medidas reais em metros. local=False mantem as coordenadas UTM reais
    (georreferenciado)."""
    data = _read_kml_bytes(kml_path)
    # remove namespace default para simplificar (mantemos localname na busca)
    root = ET.fromstring(data)

    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()

    fixed_zone = [None, None]   # trava a zona UTM na 1a coordenada
    counts = {"pontos": 0, "linhas": 0, "poligonos": 0}
    n_place = 0

    def project(latlon):
        lat, lon = latlon
        if fixed_zone[0] is None:
            _, _, z, nth = latlon_to_utm(lat, lon)
            fixed_zone[0], fixed_zone[1] = z, nth
        e, n, _, _ = latlon_to_utm(lat, lon, fixed_zone[0], fixed_zone[1])
        return (e, n)

    placemarks = _find(root, 'Placemark')
    for pm in placemarks:
        n_place += 1
        names = _find(pm, 'name')
        raw = (names[0].text if names and names[0].text else f"Local_{n_place}")
        layer = re.sub(r'[^A-Za-z0-9_\-]', '_', raw)[:60] or f"Local_{n_place}"
        if layer not in doc.layers:
            doc.layers.add(layer)
        attr = {"layer": layer}

        # Polygons (outer + inner rings)
        for poly in _find(pm, 'Polygon'):
            for tag, is_outer in (('outerBoundaryIs', True),
                                  ('innerBoundaryIs', False)):
                for b in _find(poly, tag):
                    for ring in _find(b, 'LinearRing'):
                        for c in _find(ring, 'coordinates'):
                            pts = [project(p) for p in _parse_coords(c.text or '')]
                            if len(pts) >= 2:
                                msp.add_lwpolyline(pts, close=True,
                                                   dxfattribs=attr)
                                counts["poligonos"] += 1

        # LineStrings
        for ls in _find(pm, 'LineString'):
            for c in _find(ls, 'coordinates'):
                pts = [project(p) for p in _parse_coords(c.text or '')]
                if len(pts) >= 2:
                    msp.add_lwpolyline(pts, dxfattribs=attr)
                    counts["linhas"] += 1

        # Points
        for pt in _find(pm, 'Point'):
            for c in _find(pt, 'coordinates'):
                pts = [project(p) for p in _parse_coords(c.text or '')]
                for xy in pts:
                    msp.add_point(xy, dxfattribs=attr)
                    if raw:
                        t = msp.add_text(raw, dxfattribs={
                            "layer": layer, "height": 2.0})
                        t.set_placement((xy[0] + 1, xy[1] + 1))
                    counts["pontos"] += 1

    # opcional: deslocar tudo para perto da origem (0,0), preservando medidas
    if local:
        allx, ally = [], []
        for e in msp:
            if e.dxftype() == 'LWPOLYLINE':
                for p in e.get_points():
                    allx.append(p[0]); ally.append(p[1])
            elif e.dxftype() == 'POINT':
                allx.append(e.dxf.location.x); ally.append(e.dxf.location.y)
        if allx:
            dx, dy = -min(allx), -min(ally)
            from ezdxf.math import Matrix44
            m = Matrix44.translate(dx, dy, 0)
            for e in msp:
                if e.dxftype() in ('LWPOLYLINE', 'POINT', 'TEXT', 'MTEXT'):
                    try:
                        e.transform(m)
                    except Exception:
                        pass

    doc.saveas(output_path)
    zone_txt = (f"{fixed_zone[0]}{'N' if fixed_zone[1] else 'S'}"
                if fixed_zone[0] else "-")
    total = counts["pontos"] + counts["linhas"] + counts["poligonos"]
    return {
        "placemarks": n_place,
        "zona_utm": zone_txt if not local else f"{zone_txt} (local 0,0)",
        "entidades": total,
        "coordenadas": "local" if local else "utm",
        **counts,
        "error": None if total else "Nenhuma geometria encontrada no KML",
    }


if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("uso: python3 kml_pipeline.py arquivo.kml [saida.dxf]")
        sys.exit(1)
    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else src.rsplit('.', 1)[0] + '.dxf'
    print(json.dumps(kml_to_dxf(src, out), ensure_ascii=False, indent=2))
    print("salvo em:", out)
