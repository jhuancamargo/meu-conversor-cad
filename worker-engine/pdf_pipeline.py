"""
Conversor PDF -> DXF.

Plantas exportadas de CAD (AutoCAD, Revit, ArchiCAD, SketchUp) guardam a
geometria como VETOR dentro do PDF. Este modulo le esses vetores direto
(via PyMuPDF) e reescreve como entidades DXF -- sem rasterizar, entao as
linhas saem exatas, na medida, e editaveis no AutoCAD.

Quando o PDF e um scan (uma foto da planta dentro do PDF), nao ha vetor
nenhum para ler; nesse caso o app cai no pipeline de imagem.

Sobre DWG: o formato e fechado (Autodesk) e nao existe conversor livre.
O DXF gerado aqui abre nativamente no AutoCAD, que salva como .dwg.
"""
import math

import ezdxf
import fitz  # PyMuPDF

DXF_VERSION = "R2010"

# 1 ponto PostScript = 1/72 polegada = 25.4/72 mm.
# Convertendo para milimetros o DXF sai na escala real do papel:
# uma folha A3 (1191x842 pt) vira 420x297 mm.
PT_TO_MM = 25.4 / 72.0

# Curvas Bezier viram polilinhas com este numero de segmentos.
BEZIER_STEPS = 12


def _bezier(p0, p1, p2, p3, steps=BEZIER_STEPS):
    """Bezier cubica -> lista de pontos."""
    pts = []
    for i in range(1, steps + 1):
        t = i / steps
        s = 1 - t
        x = (s**3 * p0[0] + 3 * s*s*t * p1[0] + 3 * s*t*t * p2[0] + t**3 * p3[0])
        y = (s**3 * p0[1] + 3 * s*s*t * p1[1] + 3 * s*t*t * p2[1] + t**3 * p3[1])
        pts.append((x, y))
    return pts


def _layer_por_cor(cor):
    """Agrupa a geometria em layers por cor de traco -- e assim que o
    desenho original separava as disciplinas (parede, cota, mobiliario).
    Preservar isso deixa o DXF utilizavel no AutoCAD."""
    if not cor:
        return "PDF_GERAL"
    r, g, b = (int(round(c * 255)) for c in cor[:3])
    if r == g == b:
        return "PDF_PRETO" if r < 128 else "PDF_CINZA"
    return f"PDF_COR_{r:02X}{g:02X}{b:02X}"


def pdf_to_dxf(pdf_path: str, output_path: str, pagina: int = 0,
               incluir_texto: bool = True) -> dict:
    """Converte uma pagina de PDF vetorial em DXF.

    pagina        -> indice da pagina (0 = primeira)
    incluir_texto -> traz as anotacoes/cotas como TEXT editavel
    """
    doc_pdf = fitz.open(pdf_path)
    if not len(doc_pdf):
        return {"error": "PDF vazio"}
    if pagina >= len(doc_pdf):
        return {"error": f"O PDF tem {len(doc_pdf)} página(s); a {pagina+1}ª não existe"}

    page = doc_pdf[pagina]
    altura = page.rect.height  # para inverter Y (PDF cresce para baixo)

    def para_dxf(p):
        """ponto do PDF -> ponto do DXF, em milimetros, com Y para cima"""
        return (p[0] * PT_TO_MM, (altura - p[1]) * PT_TO_MM)

    doc = ezdxf.new(dxfversion=DXF_VERSION)
    doc.header["$INSUNITS"] = 4  # 4 = milimetros
    msp = doc.modelspace()

    layers = set()
    def garante_layer(nome):
        if nome not in layers:
            if nome not in doc.layers:
                doc.layers.add(nome)
            layers.add(nome)
        return nome

    n_linhas = n_curvas = n_retangulos = 0

    for desenho in page.get_drawings():
        layer = garante_layer(_layer_por_cor(desenho.get("color")))
        attr = {"layer": layer}

        for item in desenho["items"]:
            tipo = item[0]

            if tipo == "l":                      # linha
                a, b = para_dxf(item[1]), para_dxf(item[2])
                if a != b:
                    msp.add_line(a, b, dxfattribs=attr)
                    n_linhas += 1

            elif tipo == "c":                    # bezier cubica
                p0, p1, p2, p3 = (item[1], item[2], item[3], item[4])
                pts = [para_dxf(p0)] + [para_dxf(p) for p in _bezier(p0, p1, p2, p3)]
                msp.add_lwpolyline(pts, dxfattribs=attr)
                n_curvas += 1

            elif tipo == "re":                   # retangulo
                r = item[1]
                cantos = [(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1)]
                msp.add_lwpolyline([para_dxf(p) for p in cantos],
                                   close=True, dxfattribs=attr)
                n_retangulos += 1

            elif tipo == "qu":                   # quadrilatero
                q = item[1]
                cantos = [q.ul, q.ur, q.lr, q.ll]
                msp.add_lwpolyline([para_dxf((p.x, p.y)) for p in cantos],
                                   close=True, dxfattribs=attr)
                n_retangulos += 1

    # Texto: cotas e legendas viram TEXT, editavel no AutoCAD
    n_texto = 0
    if incluir_texto:
        layer_txt = garante_layer("PDF_TEXTO")
        for bloco in page.get_text("dict")["blocks"]:
            for linha in bloco.get("lines", []):
                for trecho in linha.get("spans", []):
                    conteudo = trecho["text"].strip()
                    if not conteudo:
                        continue
                    x, y = para_dxf((trecho["bbox"][0], trecho["bbox"][3]))
                    altura_txt = max(0.5, trecho["size"] * PT_TO_MM)
                    t = msp.add_text(conteudo, dxfattribs={
                        "layer": layer_txt, "height": altura_txt})
                    t.set_placement((x, y))
                    n_texto += 1

    total = n_linhas + n_curvas + n_retangulos
    if not total and not n_texto:
        doc_pdf.close()
        return {
            "error": "Este PDF não tem desenho vetorial — parece um scan ou "
                     "uma foto. Exporte a planta do CAD em PDF, ou use a aba "
                     "Imagem → DXF."
        }

    doc.saveas(output_path)
    largura_mm = round(page.rect.width * PT_TO_MM)
    altura_mm = round(altura * PT_TO_MM)
    n_paginas = len(doc_pdf)
    doc_pdf.close()

    return {
        "entidades": total,
        "linhas": n_linhas,
        "curvas": n_curvas,
        "retangulos": n_retangulos,
        "textos": n_texto,
        "layers": len(layers),
        "paginas": n_paginas,
        "formato": f"{largura_mm}x{altura_mm} mm",
        "error": None,
    }


def contar_paginas(pdf_path: str) -> int:
    with fitz.open(pdf_path) as d:
        return len(d)


if __name__ == "__main__":
    import sys, json, time
    if len(sys.argv) < 2:
        print("uso: python3 pdf_pipeline.py arquivo.pdf [saida.dxf] [pagina]")
        sys.exit(1)
    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else src.rsplit('.', 1)[0] + '.dxf'
    pg = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    t = time.time()
    info = pdf_to_dxf(src, out, pagina=pg)
    print(json.dumps(info, ensure_ascii=False, indent=2))
    print(f"{time.time()-t:.2f}s -> {out}")
