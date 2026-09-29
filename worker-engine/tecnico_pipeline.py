"""
Desenho tecnico (preto e branco) -> DXF com geometria de CAD de verdade.

O modo classico antigo reduzia tudo a um esqueleto de 1 pixel e escrevia
polilinhas. Em desenho tecnico isso estraga tres coisas:
  1. areas preenchidas (uma barra preta de 5mm) viram uma linha so, com
     "V" nas pontas;
  2. circulos saem como polilinhas cheias de vertices, nao como CIRCLE;
  3. nada sai na medida real.

Este modulo:
  - separa AREAS PREENCHIDAS dos TRACOS finos (medindo a espessura tipica
    do traco) e desenha as areas pelo contorno, com hachura solida;
  - reconhece em cada traco LINHAS retas, ARCOS e CIRCULOS (ajuste por
    minimos quadrados) em vez de polilinhas;
  - se houver OCR, le as cotas ("50mm", "O6mm") e poe o desenho em
    milimetros, ajustando o diametro dos circulos ao valor cotado.
"""
import math
import re

import cv2
import ezdxf
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
import sknw

DXF_VERSION = "R2010"

# Resolucao de trabalho: imagens pequenas sao ampliadas (o esqueleto fica
# mais estavel), gigantes sao reduzidas (memoria/tempo).
LADO_MIN = 1600
LADO_MAX = 4000


# --------------------------------------------------------------------------
# Imagem
# --------------------------------------------------------------------------
def _carrega_cinza(caminho):
    img = cv2.imread(caminho, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        cinza = img
    else:
        if img.shape[2] == 4:  # transparencia -> fundo branco
            a = img[:, :, 3:4].astype(np.float32) / 255.0
            img = (img[:, :, :3] * a + 255 * (1 - a)).astype(np.uint8)
        cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lado = max(cinza.shape)
    fator = 1.0
    if lado < LADO_MIN:
        fator = LADO_MIN / lado
    elif lado > LADO_MAX:
        fator = LADO_MAX / lado
    if fator != 1.0:
        cinza = cv2.resize(cinza, None, fx=fator, fy=fator,
                           interpolation=cv2.INTER_LANCZOS4 if fator > 1
                           else cv2.INTER_AREA)
    return cinza, fator


def _disco(r):
    r = max(1, int(round(r)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


# --------------------------------------------------------------------------
# Ajustes geometricos
# --------------------------------------------------------------------------
def _ajusta_reta(p):
    """PCA. Devolve (ponto, direcao, desvio_max)."""
    c = p.mean(axis=0)
    u, s, vt = np.linalg.svd(p - c, full_matrices=False)
    d = vt[0]
    n = np.array([-d[1], d[0]])
    desvio = np.abs((p - c) @ n)
    return c, d, float(desvio.max())


def _ajusta_circulo(p):
    """Kasa (algebrico) refinado por minimos quadrados geometricos.
    Devolve (cx, cy, r, desvio_max)."""
    x, y = p[:, 0], p[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x * x + y * y)
    try:
        D, E, F = np.linalg.lstsq(A, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    cx, cy = -D / 2, -E / 2
    r2 = cx * cx + cy * cy - F
    if r2 <= 0:
        return None

    def res(q):
        return np.hypot(x - q[0], y - q[1]) - q[2]

    q = least_squares(res, [cx, cy, math.sqrt(r2)], method="lm").x
    return q[0], q[1], abs(q[2]), float(np.abs(res(q)).max())


def _varredura(p, cx, cy):
    """Angulo percorrido (graus, com sinal) ao longo dos pontos."""
    ang = np.unwrap(np.arctan2(p[:, 1] - cy, p[:, 0] - cx))
    return math.degrees(ang[-1] - ang[0]), ang


class _Tol:
    def __init__(self, traco, nivel):
        # nivel 0-10: 10 = segue cada detalhe (tolerancia menor)
        f = 1.6 - 0.1 * max(0, min(10, nivel))       # 1.6 .. 0.6
        self.reta = max(1.2, 0.30 * traco) * f
        self.arco = max(1.2, 0.30 * traco) * f
        self.traco = traco


def _encaixa(p, tol):
    """Tenta reta, depois arco. Devolve primitiva ou None."""
    if len(p) < 2:
        return None
    c, d, dev = _ajusta_reta(p)
    if dev <= tol.reta:
        t = (p - c) @ d
        return ("LINE", c + d * t.min(), c + d * t.max()) if t[0] <= t[-1] \
            else ("LINE", c + d * t.max(), c + d * t.min())
    if len(p) < 8:
        return None
    circ = _ajusta_circulo(p)
    if circ is None:
        return None
    cx, cy, r, dev = circ
    if dev > tol.arco or r < 2 * tol.traco:
        return None
    sweep, _ = _varredura(p, cx, cy)
    if abs(sweep) < 12:
        return None
    return ("ARC", cx, cy, r, p[0], p[-1], sweep, p)


def _segmenta(p, tol):
    """Divide uma cadeia de pontos na menor sequencia de retas e arcos.
    Guloso: a partir de cada vertice de Douglas-Peucker, estende a
    primitiva o mais longe possivel enquanto ela ainda encaixa."""
    if len(p) < 2:
        return []
    ap = cv2.approxPolyDP(p.astype(np.float32).reshape(-1, 1, 2),
                          tol.reta * 0.8, False)
    # indices dos vertices DP na cadeia original
    idx = [0]
    j = 0
    for v in ap.reshape(-1, 2)[1:]:
        dist = np.hypot(p[j:, 0] - v[0], p[j:, 1] - v[1])
        j = j + int(dist.argmin())
        if j > idx[-1]:
            idx.append(j)
    if idx[-1] != len(p) - 1:
        idx.append(len(p) - 1)

    out = []
    a = 0
    while a < len(idx) - 1:
        melhor = None
        for b in range(a + 1, len(idx)):
            prim = _encaixa(p[idx[a]:idx[b] + 1], tol)
            if prim is None:
                if melhor is not None:
                    break
                continue
            melhor = (b, prim)
        if melhor is None:           # nao encaixou nada: polilinha curta
            b = a + 1
            out.append(("POLY", p[idx[a]:idx[b] + 1]))
        else:
            b, prim = melhor
            out.append(prim)
        a = b
    return out


# --------------------------------------------------------------------------
# OCR das cotas (opcional: sem easyocr o resto funciona, so sem escala)
# --------------------------------------------------------------------------
_LEITOR = None
_ALFABETO = "0123456789.,m"   # uma cota e so isto; menos letras = menos erro


def _leitor():
    global _LEITOR
    if _LEITOR is None:
        import easyocr
        try:
            import torch
            gpu = torch.cuda.is_available()
        except Exception:
            gpu = False
        _LEITOR = easyocr.Reader(["en"], gpu=gpu, verbose=False)
    return _LEITOR


def ler_cotas(cinza):
    """Etapa 1: detetar ONDE ha texto (inclui texto vertical).
    Etapa 2: reler cada caixa sozinha -- o reconhecimento da imagem toda
    confundia o simbolo de diametro e lia "O50mm" como "8"."""
    try:
        leitor = _leitor()
    except Exception:
        return []
    achados = leitor.readtext(cinza, rotation_info=[90, 270],
                              allowlist=_ALFABETO)
    textos = []
    for caixa, txt0, conf0 in achados:
        xs = [q[0] for q in caixa]
        ys = [q[1] for q in caixa]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        vertical = (y1 - y0) > 1.4 * (x1 - x0)
        # a 1a leitura so vale se ja parecer uma cota ("8" nao e cota)
        melhor = (conf0, txt0) if _RE_COTA.search(txt0) else (-1.0, txt0)
        # Relemos SEMPRE o que tem numero: o OCR chegou a ler "13mm" com 99%
        # de confianca quando era o tracinho da cota colado a "3mm".
        if not re.search(r"\d", txt0):   # sem numero nenhum: nao e cota
            textos.append({"texto": txt0, "conf": float(conf0),
                           "caixa": (x0, y0, x1, y1), "vertical": vertical})
            continue
        for pad in (0.1, 0.3):
            m = int(pad * min(x1 - x0, y1 - y0))
            corte = cinza[max(0, int(y0) - m):int(y1) + m,
                          max(0, int(x0) - m):int(x1) + m]
            if corte.size == 0:
                continue
            giros = ([cv2.rotate(corte, cv2.ROTATE_90_CLOCKWISE),
                      cv2.rotate(corte, cv2.ROTATE_90_COUNTERCLOCKWISE)]
                     if vertical else [corte])
            for g in giros:
                for esc in (1.0, 0.5):
                    gg = cv2.resize(g, None, fx=esc, fy=esc,
                                    interpolation=cv2.INTER_AREA)
                    for _, t, c in leitor.recognize(gg, allowlist=_ALFABETO):
                        if _RE_COTA.search(t) and c > melhor[0]:
                            melhor = (c, t)
        textos.append({"texto": melhor[1], "conf": float(melhor[0]),
                       "caixa": (x0, y0, x1, y1), "vertical": vertical})
    return textos


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------
def _geometria(bin_, traco, tol, altura, caixas_fora):
    """Tracos finos -> retas/arcos/circulos; areas cheias -> contornos.
    caixas_fora: retangulos (texto) a tirar da geometria."""
    mascara = np.zeros_like(bin_)
    for (x0, y0, x1, y1) in caixas_fora:
        m = int(0.6 * traco)
        cv2.rectangle(mascara, (int(x0) - m, int(y0) - m),
                      (int(x1) + m, int(y1) + m), 255, -1)

    # areas preenchidas: sobrevivem a uma abertura com disco > traco
    cheio = cv2.morphologyEx(bin_, cv2.MORPH_OPEN, _disco(1.6 * traco))
    cheio[mascara > 0] = 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(cheio)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < (4 * traco) ** 2:
            cheio[lab == i] = 0

    finos = bin_.copy()
    finos[cv2.dilate(cheio, _disco(0.25 * traco)) > 0] = 0
    finos[mascara > 0] = 0

    esq = skeletonize(finos > 0).astype(np.uint16)
    grafo = sknw.build_sknw(esq, multi=True, ring=True)
    prims = []
    # Poda: so se cortam os "pelinhos" que o esqueleto cria junto a um
    # cruzamento (pedaco curto preso a uma juncao e solto na outra ponta).
    # Um traco curto SOLTO dos dois lados e tracejado ou ponto: fica.
    pelinho = max(6, int(1.2 * traco))
    grau = dict(grafo.degree())
    for s_, e_, k_ in grafo.edges(keys=True):
        pts = grafo[s_][e_][k_]["pts"]
        n = len(pts)
        soltos = (grau[s_] == 1) + (grau[e_] == 1)
        if soltos == 2 and n < max(3, int(traco)):
            continue            # sujidade: menor que a propria espessura
        if soltos == 1 and n < pelinho:
            continue            # pelinho do esqueleto
        if n < 2:
            continue
        p = np.column_stack([pts[:, 1], altura - pts[:, 0]]).astype(float)
        fechado = s_ == e_ or np.hypot(*(p[0] - p[-1])) < 1.5 * traco
        if fechado and len(p) >= 12:
            circ = _ajusta_circulo(p)
            if circ and circ[3] <= tol.arco and circ[2] > 1.5 * traco:
                prims.append(("CIRCLE", circ[0], circ[1], circ[2], p))
                continue
        prims += _segmenta(p, tol)

    cheios = []
    cont, _ = cv2.findContours(cheio, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_NONE)
    for c in cont:
        cheios.append(_contorno_cheio(c, altura, traco))

    prims = _unifica_circulos(prims, traco, bin_, altura)
    prims = _endireita(prims)
    prims = _solda(prims, cheios, traco)
    return prims, cheios


def _contorno_cheio(c, altura, traco):
    """Area preenchida -> poligono. Se for (quase) um retangulo, sai um
    retangulo exato: a abertura morfologica arredonda os cantos."""
    ret = cv2.minAreaRect(c)
    (w, h) = ret[1]
    if w * h > 0 and cv2.contourArea(c) / (w * h) > 0.9:
        ang = ret[2] % 90
        if ang < 2 or ang > 88:            # alinhado aos eixos
            x, y, ww, hh = cv2.boundingRect(c)
            pts = np.array([[x, y], [x + ww, y], [x + ww, y + hh], [x, y + hh]],
                           float)
        else:
            pts = cv2.boxPoints(ret).astype(float)
    else:
        pts = cv2.approxPolyDP(c, max(1.5, 0.4 * traco), True).reshape(-1, 2)
        pts = pts.astype(float)
    pts[:, 1] = altura - pts[:, 1]
    return pts


def converte(caminho, saida, nivel=7, usar_ocr=True):
    """Converte desenho tecnico em DXF. Devolve dict com estatisticas."""
    carregado = _carrega_cinza(caminho)
    if carregado is None:
        return {"error": "Não consegui abrir a imagem."}
    cinza, fator = carregado
    altura = cinza.shape[0]

    _, bin_ = cv2.threshold(cv2.GaussianBlur(cinza, (3, 3), 0), 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # espessura tipica do traco (2x a distancia ao fundo, no esqueleto)
    dist = cv2.distanceTransform(bin_, cv2.DIST_L2, 5)
    esq = skeletonize(bin_ > 0)
    if not esq.any():
        return {"error": "Não encontrei traços nesta imagem."}
    traco = max(2.0, 2.0 * float(np.median(dist[esq])))
    tol = _Tol(traco, nivel)

    # 1a passagem: geometria completa (o texto ainda e desenho)
    prims, cheios = _geometria(bin_, traco, tol, altura, [])

    # cotas -> escala. So vale a cota que a geometria CONFIRMA: uma leitura
    # errada do OCR nao pode inventar texto nem mudar a escala.
    textos = ler_cotas(cinza) if usar_ocr else []
    escala, confirmadas = _escala_pelas_cotas(textos, prims, altura, traco)
    for i, t in enumerate(textos):
        t["confirmada"] = i in confirmadas
    # vira TEXT: cota confirmada pela geometria, ou (havendo escala) leitura
    # muito segura. So as confirmadas decidem escala e diametros.
    cotas = [t for t in textos if t["confirmada"] or
             (escala and t["conf"] >= 0.8 and _valor_cota(t["texto"])[0])]

    # 2a passagem: tira as cotas confirmadas da geometria (viram TEXT)
    if cotas:
        prims, cheios = _geometria(bin_, traco, tol, altura,
                                   [t["caixa"] for t in cotas])

    k = escala if escala else 1.0
    if escala:
        prims = _ajusta_as_cotas(prims, cotas, escala, altura)
        prims = _solda(prims, cheios, traco)

    # ---- DXF ----
    doc = ezdxf.new(dxfversion=DXF_VERSION)
    doc.header["$INSUNITS"] = 4 if escala else 0   # 4 = mm
    msp = doc.modelspace()
    doc.layers.add("GEOMETRIA", color=7)
    doc.layers.add("PREENCHIDO", color=8)
    doc.layers.add("COTAS", color=3)
    cont_ = {"LINE": 0, "ARC": 0, "CIRCLE": 0, "POLY": 0}
    g = {"layer": "GEOMETRIA"}

    for pr in prims:
        t = pr[0]
        if t == "LINE":
            msp.add_line(tuple(pr[1] * k), tuple(pr[2] * k), dxfattribs=g)
        elif t == "CIRCLE":
            msp.add_circle((pr[1] * k, pr[2] * k), pr[3] * k, dxfattribs=g)
        elif t == "ARC":
            cx, cy, r, p0, p1, sweep = pr[1:7]
            a0 = math.degrees(math.atan2(p0[1] - cy, p0[0] - cx))
            a1 = math.degrees(math.atan2(p1[1] - cy, p1[0] - cx))
            if sweep < 0:          # DXF desenha arcos no sentido anti-horario
                a0, a1 = a1, a0
            msp.add_arc((cx * k, cy * k), r * k, a0, a1, dxfattribs=g)
        else:
            msp.add_lwpolyline([tuple(q * k) for q in pr[1]], dxfattribs=g)
        cont_[t] += 1

    for ap in cheios:
        pts = [tuple(q * k) for q in ap]
        msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "PREENCHIDO"})
        h = msp.add_hatch(color=8, dxfattribs={"layer": "PREENCHIDO"})
        h.paths.add_polyline_path(pts, is_closed=True)

    for t in cotas:
        v, diam = _valor_cota(t["texto"])
        conteudo = ("%%c" if diam else "") + f"{v:g}mm"
        x0, y0, x1, y1 = t["caixa"]
        if t["vertical"]:
            alt_txt = (x1 - x0) * 0.72 * k
            ins = (x1 * k - (x1 - x0) * 0.14 * k, (altura - y1) * k)
            rot = 90
        else:
            alt_txt = (y1 - y0) * 0.72 * k
            ins = (x0 * k, (altura - y1) * k + (y1 - y0) * 0.14 * k)
            rot = 0
        tx = msp.add_text(conteudo, dxfattribs={
            "layer": "COTAS", "height": max(alt_txt, 0.1), "rotation": rot})
        tx.set_placement(ins)

    doc.saveas(saida)
    total = sum(cont_.values()) + len(cheios)
    return {
        "retas": cont_["LINE"], "arcos": cont_["ARC"],
        "circulos": cont_["CIRCLE"], "polilinhas": cont_["POLY"],
        "preenchidos": len(cheios), "cotas_lidas": len(textos),
        "cotas_confirmadas": [("Ø" if _valor_cota(t["texto"])[1] else "")
                              + f"{_valor_cota(t['texto'])[0]:g}mm"
                              for t in cotas],
        "entidades": total, "unidade": "mm" if escala else "px",
        "mm_por_px": round(escala * fator, 5) if escala else None,
        "traco_px": round(traco, 1),
        "error": None if total else "Não encontrei geometria nesta imagem.",
    }


# --------------------------------------------------------------------------
# Limpeza
# --------------------------------------------------------------------------
def _cobertura(intervalos):
    """Graus cobertos pela uniao de intervalos angulares (a0, a1) em rad."""
    marcas = np.zeros(720, bool)
    for a0, a1 in intervalos:
        lo, hi = sorted((a0, a1))
        i0, i1 = int(np.floor(np.degrees(lo) * 2)), int(np.ceil(np.degrees(hi) * 2))
        for i in range(i0, i1 + 1):
            marcas[i % 720] = True
    return marcas.sum() / 2.0


def _evidencia_circulo(cx, cy, r, tinta, altura, traco):
    """Fracao da circunferencia (grau a grau) que tem tinta NA IMAGEM.
    Um circulo desenhado tem tinta a toda a volta (por baixo de uma area
    preenchida tambem conta); um arco cuja continuacao imaginaria atravessa
    papel em branco nao tem."""
    t = max(2, int(round(traco)))
    h, w = tinta.shape
    ok = 0
    for g in range(360):
        a = math.radians(g)
        x = int(round(cx + r * math.cos(a)))
        y = int(round(altura - (cy + r * math.sin(a))))
        if 0 <= x < w and 0 <= y < h and \
                tinta[max(0, y - t):y + t + 1, max(0, x - t):x + t + 1].any():
            ok += 1
    return ok / 360.0


def _unifica_circulos(prims, traco, tinta=None, altura=None):
    """Pedacos do mesmo circulo (centro e raio a menos de ~4%) sao
    reajustados JUNTOS, com todos os pontos, e passam a partilhar
    exatamente o mesmo centro e raio. Se os pedacos cobrem 270 graus ou
    mais, sai um CIRCLE inteiro (o que falta costuma estar tapado por uma
    area preenchida ou por um cruzamento de linhas)."""
    curvas = [i for i, p in enumerate(prims) if p[0] in ("ARC", "CIRCLE")]
    grupos, visto = [], set()
    for i in curvas:
        if i in visto:
            continue
        g = [i]
        visto.add(i)
        for j in curvas:
            if j in visto:
                continue
            a, b = prims[i], prims[j]
            lim = max(traco, 0.04 * max(a[3], b[3]))
            if math.hypot(a[1] - b[1], a[2] - b[2]) < lim and abs(a[3] - b[3]) < lim:
                g.append(j)
                visto.add(j)
        grupos.append(g)

    fora = set()
    novos = []
    # o grupo com MAIS pontos primeiro: e o circulo verdadeiro, e absorve
    # os pedacos soltos (ordenar por raio punha a frente o pedaco torto)
    def _suporte(g):
        return sum(len(_amostra(prims[k]) if prims[k][0] == "ARC" else prims[k][4])
                   for k in g)
    grupos.sort(key=lambda g: -_suporte(g))
    for g in grupos:
        g = [k for k in g if k not in fora]   # ja absorvidos por outro
        if not g:
            continue
        if len(g) == 1 and prims[g[0]][0] == "CIRCLE":
            continue
        pts = np.vstack([prims[k][7] if prims[k][0] == "ARC" else prims[k][4]
                         for k in g])
        circ = _ajusta_circulo(pts)
        if circ is None:
            continue
        cx, cy, r, _ = circ
        # arcos soltos cujos PONTOS estao em cima deste circulo (sujidade no
        # desenho entorta o ajuste do pedaco) entram no grupo. Retas NUNCA:
        # num circulo grande qualquer reta curta "fica perto" dele, e numa
        # planta ha milhares -- era assim que nasciam circulos fantasma.
        if r > 6 * traco:
            lim = max(1.5 * traco, 0.02 * r)
            for k, pk in enumerate(prims):
                if k in fora or k in g or pk[0] != "ARC":
                    continue
                if abs(pk[3] - r) > 0.15 * r:
                    continue
                a = pk[7]
                if np.abs(np.hypot(a[:, 0] - cx, a[:, 1] - cy) - r).max() < lim:
                    g.append(k)
            extra = [prims[k][7] for k in g if prims[k][0] == "ARC"]
            if extra:
                todos = np.vstack([pts] + extra)
                circ = _ajusta_circulo(todos)
                if circ is None:
                    continue
                cx, cy, r, _ = circ
                pts = todos
        fora.update(g)
        # cobertura medida nos PONTOS reais do grupo, em fatias de 2 graus
        ang = np.degrees(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)) % 360
        inteiro = len(np.unique((ang // 2).astype(int))) * 2 >= 270
        if inteiro and tinta is not None:
            inteiro = _evidencia_circulo(cx, cy, r, tinta, altura, traco) >= 0.9
        if inteiro:
            novos.append(("CIRCLE", cx, cy, r, pts))
            # pedacos (retas incluidas) totalmente EM CIMA do traco deste
            # circulo sao duplicados: o circulo ja os desenha. So se removem
            # DEPOIS de o circulo estar confirmado -- nao contam para ele.
            lim = max(1.5 * traco, 0.02 * r)
            for k, pk in enumerate(prims):
                if k in fora or pk[0] == "CIRCLE":
                    continue
                a = _amostra(pk)
                if a is not None and len(a) >= 2 and \
                        np.abs(np.hypot(a[:, 0] - cx, a[:, 1] - cy) - r).max() < lim:
                    fora.add(k)
        else:
            for k in g:
                pk = prims[k]
                if pk[0] == "ARC":
                    # o sentido tem de ser recalculado com o NOVO centro:
                    # com o sentido errado o DXF desenha o complemento do arco
                    sw, _ = _varredura(pk[7], cx, cy)
                    novos.append(("ARC", cx, cy, r, pk[4], pk[5], sw, pk[7]))
                else:
                    novos.append(pk)
    prims = [p for i, p in enumerate(prims) if i not in fora] + novos

    # aneis: centros quase coincidentes passam a ser o mesmo centro
    curvas = [i for i, p in enumerate(prims) if p[0] in ("ARC", "CIRCLE")]
    for i in curvas:
        for j in curvas:
            if j <= i:
                continue
            a, b = prims[i], prims[j]
            if math.hypot(a[1] - b[1], a[2] - b[2]) < traco:
                cx, cy = (a[1] + b[1]) / 2, (a[2] + b[2]) / 2
                prims[i] = (a[0], cx, cy) + tuple(a[3:])
                prims[j] = (b[0], cx, cy) + tuple(b[3:])
    return prims


def _amostra(pr):
    """Pontos ao longo de uma primitiva aberta."""
    if pr[0] == "LINE":
        t = np.linspace(0, 1, 12)[:, None]
        return pr[1] * (1 - t) + pr[2] * t
    if pr[0] == "ARC":
        return pr[7]
    if pr[0] == "POLY":
        return np.asarray(pr[1], float)
    return None


def _pontas(pr):
    """Pontas de uma primitiva aberta (LINE/ARC)."""
    if pr[0] == "LINE":
        return [pr[1], pr[2]]
    if pr[0] == "ARC":
        return [pr[4], pr[5]]
    return []


def _solda(prims, cheios, traco):
    """Pontas soltas a menos de 2 tracos de outra ponta (ou da borda de uma
    area preenchida) passam a encostar. Cada primitiva mantem a sua forma:
    a reta desliza ao longo de si propria, o arco ao longo do circulo."""
    raio = 2.0 * traco
    pontas = []                         # (indice_prim, 0|1, ponto)
    for i, pr in enumerate(prims):
        for k, q in enumerate(_pontas(pr)):
            pontas.append((i, k, np.array(q, float)))
    bordas = []
    for poli in cheios:
        for j in range(len(poli)):
            bordas.append((poli[j], poli[(j + 1) % len(poli)]))

    alvo = {}
    if not pontas:
        return prims
    arvore = cKDTree(np.array([q for _, _, q in pontas]))
    for n, (i, k, q) in enumerate(pontas):
        viz = [pontas[m][2] for m in arvore.query_ball_point(q, raio)
               if pontas[m][0] != i]
        if viz:
            alvo[(i, k)] = np.mean([q] + viz, axis=0)
            continue
        melhor = None
        for a, b in bordas:
            ab = b - a
            t = np.clip(np.dot(q - a, ab) / max(np.dot(ab, ab), 1e-9), 0, 1)
            c = a + t * ab
            d = float(np.hypot(*(q - c)))
            if d < raio and (melhor is None or d < melhor[0]):
                melhor = (d, c)
        if melhor:
            alvo[(i, k)] = melhor[1]

    out = list(prims)
    for (i, k), t in alvo.items():
        pr = out[i]
        if pr[0] == "LINE":
            a, b = pr[1].copy(), pr[2].copy()
            d = (b - a) / max(np.hypot(*(b - a)), 1e-9)
            base = a if k == 0 else b
            novo = base + d * float(np.dot(t - base, d))
            if k == 0:
                a = novo
            else:
                b = novo
            out[i] = ("LINE", a, b)
        elif pr[0] == "ARC":
            cx, cy, r = pr[1], pr[2], pr[3]
            ang = math.atan2(t[1] - cy, t[0] - cx)
            q = np.array([cx + r * math.cos(ang), cy + r * math.sin(ang)])
            p0, p1 = (q, pr[5]) if k == 0 else (pr[4], q)
            out[i] = ("ARC", cx, cy, r, p0, p1, pr[6], pr[7])
    return out


def _endireita(prims, graus=1.5):
    """Retas quase horizontais/verticais ficam exatamente H/V."""
    out = []
    for p in prims:
        if p[0] == "LINE":
            a, b = p[1].copy(), p[2].copy()
            ang = abs(math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))) % 180
            if ang < graus or ang > 180 - graus:
                y = (a[1] + b[1]) / 2
                a[1] = b[1] = y
            elif abs(ang - 90) < graus:
                x = (a[0] + b[0]) / 2
                a[0] = b[0] = x
            p = ("LINE", a, b)
        out.append(p)
    return out


# --------------------------------------------------------------------------
# Cotas -> escala
# --------------------------------------------------------------------------
_RE_COTA = re.compile(r"(\d+(?:[.,]\d+)?)\s*mm", re.I)
_RE_DIAM = re.compile(r"^[\s]*[Øø⌀Φφ0Oo]", re.I)


def _valor_cota(texto):
    """'50mm' -> (50, False); 'O50mm'/'050mm' -> (50, True).
    O OCR le o simbolo de diametro como '0': um '0' colado a outro
    digito ('06mm') nao e numero real (seria 0.6), e o simbolo."""
    t = texto.replace(" ", "")
    diam = False
    if re.match(r"^[ØøΦφ⌀Oo]", t):
        diam, t = True, t[1:]
    elif re.match(r"^0\d", t):
        diam, t = True, t[1:]
    m = _RE_COTA.search(t)
    if not m:
        return None, False
    v = float(m.group(1).replace(",", "."))
    return (v, diam) if v > 0 else (None, False)


def _linhas_de_cota(prims, traco):
    """Linha de cota = reta com um tracinho a ATRAVESSAR cada ponta
    (o |---| do desenho tecnico): ha traco perpendicular dos dois lados da
    ponta. Um canto de retangulo so tem perpendicular de um lado, por isso
    nao passa por cota."""
    retas = [(p[1], p[2]) for p in prims if p[0] == "LINE"]
    if not retas:
        return []
    # tracinhos candidatos: retas curtas, indexadas pelo ponto medio
    maxtick = 8 * traco
    ticks = [(c, e) for c, e in retas
             if 0.5 * traco <= np.hypot(*(e - c)) <= maxtick]
    if not ticks:
        return []
    arvore = cKDTree(np.array([(c + e) / 2 for c, e in ticks]))
    out = []
    for a, b in retas:
        L = float(np.hypot(*(b - a)))
        if L < 2 * traco:
            continue
        d = (b - a) / L
        n = np.array([-d[1], d[0]])
        pontas = 0
        for ponta in (a, b):
            lados = []
            for m in arvore.query_ball_point(ponta, 1.5 * traco + maxtick / 2):
                c, e = ticks[m]
                l2 = float(np.hypot(*(e - c)))
                if abs(float(np.dot((e - c) / l2, d))) > 0.5:
                    continue             # nao e atravessado
                if _dist_ponto_segmento(ponta, c, e) > 1.5 * traco:
                    continue
                lados += [float(np.dot(c - ponta, n)), float(np.dot(e - ponta, n))]
            if lados and max(lados) > 0.8 * traco and min(lados) < -0.8 * traco:
                pontas += 1
        if pontas == 2:
            out.append((a, b, L))
    return out


def _dist_ponto_segmento(q, a, b):
    ab = b - a
    t = np.clip(np.dot(q - a, ab) / max(np.dot(ab, ab), 1e-9), 0, 1)
    return float(np.hypot(*(q - (a + t * ab))))


def _escala_pelas_cotas(textos, prims, altura, traco):
    """Para cada cota lida, a linha de cota mais proxima da a medida em
    pixels -> uma escala mm/px por cota. Desenhos de catalogo nao estao
    exatamente a escala (as cotas variam ~10% entre si), por isso a escala
    final e a MEDIANA, e so contam as cotas a menos de 15% dela.
    Precisa de pelo menos 2 cotas a concordar."""
    linhas = _linhas_de_cota(prims, traco)
    if not linhas:
        return None, []
    por_cota = {}
    for i, t in enumerate(textos):
        v, _ = _valor_cota(t["texto"])
        if not v or t.get("conf", 1) < 0.4:
            continue
        x0, y0, x1, y1 = t["caixa"]
        centro = np.array([(x0 + x1) / 2, altura - (y0 + y1) / 2])
        alcance = 3 * max(x1 - x0, y1 - y0)
        dists = [(_dist_ponto_segmento(centro, a, b), L) for a, b, L in linhas]
        d, L = min(dists)
        if d < alcance:
            por_cota[i] = v / L
            t["linha_px"] = L
    if len(por_cota) < 2:
        return None, []
    med = float(np.median(list(por_cota.values())))
    boas = {i: e for i, e in por_cota.items() if abs(e / med - 1) < 0.15}
    if len(boas) < 2:
        return None, []
    escala = float(np.median(list(boas.values())))
    return escala, sorted(boas)


def _com_raio(p, r):
    """Mesmo circulo/arco com outro raio; as pontas do arco acompanham."""
    if p[0] == "CIRCLE":
        return (p[0], p[1], p[2], r) + tuple(p[4:])
    cx, cy = p[1], p[2]
    novas = []
    for q in (p[4], p[5]):
        a = math.atan2(q[1] - cy, q[0] - cx)
        novas.append(np.array([cx + r * math.cos(a), cy + r * math.sin(a)]))
    return ("ARC", cx, cy, r, novas[0], novas[1], p[6], p[7])


def _ajusta_as_cotas(prims, textos, escala, altura):
    """Cada cota de diametro ajusta UM circulo inteiro ao valor cotado.
    Qual circulo: se a cota tem a sua linha de cota, e o circulo cujo
    diametro em pixels bate (5%) com o comprimento DESSA linha -- nao
    depende da escala geral, que num desenho de catalogo fora de escala
    varia ~10% de cota para cota. Sem linha de cota, compara pela escala
    geral (8%). Entre candidatos, o mais perto do texto. Arcos soltos nao
    sao ajustados (um arco de raio parecido era empurrado e abria buracos)."""
    out = list(prims)
    for t in textos:
        if not t.get("confirmada"):
            continue
        v, diam = _valor_cota(t["texto"])
        if not (v and diam):
            continue
        x0, y0, x1, y1 = t["caixa"]
        cxt, cyt = (x0 + x1) / 2, altura - (y0 + y1) / 2
        L = t.get("linha_px")
        cand = []
        for i, p in enumerate(out):
            if p[0] != "CIRCLE":
                continue
            if L:
                bate = abs(2 * p[3] - L) / L < 0.05
            else:
                bate = abs(v - 2 * p[3] * escala) / v < 0.08
            if bate:
                cand.append((abs(math.hypot(p[1] - cxt, p[2] - cyt) - p[3]), i))
        if cand:
            i = min(cand)[1]
            out[i] = _com_raio(out[i], v / 2 / escala)
    return out
