import ezdxf, cv2, numpy as np
import os, subprocess, shutil

# nº de linhas de varredura. Quanto maior, mais fiel (e maior o arquivo).
TARGET_ROWS = 400


def _collect_segments(msp):
    segs = []
    for ent in msp:
        t = ent.dxftype()
        if t == 'LWPOLYLINE':
            pts = [(p[0], p[1]) for p in ent.get_points()]
            closed = ent.closed
            for i in range(len(pts) - 1):
                segs.append((*pts[i], *pts[i+1]))
            if closed and len(pts) > 2:
                segs.append((*pts[-1], *pts[0]))
        elif t == 'POLYLINE':
            pts = [(v.dxf.location.x, v.dxf.location.y) for v in ent.vertices]
            for i in range(len(pts) - 1):
                segs.append((*pts[i], *pts[i+1]))
        elif t == 'LINE':
            s, e = ent.dxf.start, ent.dxf.end
            segs.append((s.x, s.y, e.x, e.y))
        elif t in ('CIRCLE', 'ARC'):
            c = ent.dxf.center; r = ent.dxf.radius
            if t == 'ARC':
                a0 = np.radians(ent.dxf.start_angle)
                a1 = np.radians(ent.dxf.end_angle)
                if a1 < a0: a1 += 2*np.pi
            else:
                a0, a1 = 0, 2*np.pi
            steps = max(16, int(r))
            ang = np.linspace(a0, a1, steps)
            xs = c.x + r*np.cos(ang); ys = c.y + r*np.sin(ang)
            for i in range(len(ang)-1):
                segs.append((xs[i], ys[i], xs[i+1], ys[i+1]))
    return segs


def dxf_to_pat(dxf_path: str, output_path: str, name: str = "CUSTOM") -> dict:
    """
    Converte DXF em PAT por varredura horizontal em ALTA resolucao.
    Mesmo metodo do architextures.org: cada faixa Y vira uma linha PAT
    (angle=0) cujo dash pattern desenha exatamente onde ha geometria.
    Mantem as coordenadas REAIS do desenho (escala 1 no AutoCAD).
    """
    doc = ezdxf.readfile(dxf_path)
    segs = _collect_segments(doc.modelspace())
    if not segs:
        return {"families": 0, "segments": 0, "error": "Nenhuma geometria no DXF"}

    xs = [s[0] for s in segs] + [s[2] for s in segs]
    ys = [s[1] for s in segs] + [s[3] for s in segs]
    x0, y0 = min(xs), min(ys)
    W = max(xs) - x0
    H = max(ys) - y0
    if W < 1e-6 or H < 1e-6:
        return {"families": 0, "segments": len(segs), "error": "Geometria degenerada"}

    # grade proporcional: TARGET_ROWS na altura, colunas mantendo aspect ratio
    rows = TARGET_ROWS
    cols = max(1, int(round(TARGET_ROWS * W / H)))

    # rasterizar em coordenadas reais -> pixels (linha 0 = topo)
    img = np.zeros((rows, cols), dtype=np.uint8)
    for sx0, sy0, sx1, sy1 in segs:
        c0 = int((sx0 - x0) / W * (cols - 1))
        c1 = int((sx1 - x0) / W * (cols - 1))
        r0 = int((1.0 - (sy0 - y0) / H) * (rows - 1))
        r1 = int((1.0 - (sy1 - y0) / H) * (rows - 1))
        cv2.line(img, (c0, r0), (c1, r1), 255, 1, cv2.LINE_8)

    result = _bitmap_to_pat(img, W, H, output_path, name)
    result["segments"] = len(segs)
    return result


def _bitmap_to_pat(img, W, H, output_path, name="CUSTOM"):
    """
    Nucleo compartilhado: recebe um bitmap binario (255 = desenhar) com tile
    de dimensoes reais WxH e escreve o PAT por varredura horizontal.
    Cada faixa Y vira 1+ linhas PAT (angle=0) com dash pattern.
    """
    rows, cols = img.shape
    step_x = W / cols

    safe = ''.join(ch if ch.isalnum() or ch == '_' else '_'
                   for ch in name.upper())[:31] or "CUSTOM"

    # O AutoCAD rejeita linhas com muitos dashes ("Too many dash specifications").
    # O limite varia por versao (algumas cortam bem abaixo de 80). Cada "run"
    # gera ate 2 dashes (vao + desenho); com 16 runs = ~33 dashes/linha, bem
    # seguro. Faixas densas viram varias linhas PAT no mesmo Y (como o architextures).
    MAX_RUNS_PER_LINE = 16

    pat_lines = []
    for row in range(rows):
        line = img[row]
        if not line.any():
            continue
        # Y real desta faixa (row 0 = topo = H)
        y_real = round((rows - 1 - row) / (rows - 1) * H, 4)

        # extrair "runs" (segmentos desenhados) como pares (inicio, fim) em X real
        runs = []
        in_run = False
        run_start = 0.0
        for col in range(cols + 1):
            on = col < cols and line[col] > 127
            cx = col * step_x
            if on and not in_run:
                run_start = cx
                in_run = True
            elif not on and in_run:
                if cx - run_start > 1e-6:
                    runs.append((run_start, cx))
                in_run = False
        if not runs:
            continue

        # emitir em lotes para respeitar o limite de dashes do AutoCAD
        for i in range(0, len(runs), MAX_RUNS_PER_LINE):
            batch = runs[i:i + MAX_RUNS_PER_LINE]
            # O AutoCAD exige os valores em PARES (traco, espaco, traco, espaco...)
            # comecando por um traco positivo. Por isso a linha comeca no inicio
            # do 1o segmento (x-origin = a0) em vez de por um espaco negativo.
            x_org = batch[0][0]
            dashes = []
            prev_end = None
            for a, b in batch:
                if prev_end is not None:
                    gap = a - prev_end            # espaco entre segmentos
                    dashes.append(-round(gap, 4))
                dashes.append(round(b - a, 4))     # traco (segmento desenhado)
                prev_end = b
            # espaco final: completa o periodo (largura do tile), fechando o par.
            # inclui o deslocamento inicial pois o padrao repete a cada W.
            closing = W - (prev_end - x_org)
            if closing <= 1e-4:
                # caso raro (segmento toca as duas bordas): encurta o ultimo
                # traco para sobrar um espaco minimo e manter o par valido.
                dashes[-1] = round(dashes[-1] - 0.01, 4)
                closing = 0.01
            dashes.append(-round(closing, 4))
            dash_str = ','.join(f"{d:g}" for d in dashes)
            pat_lines.append(f"0,{x_org:g},{y_real:g},0,{H:g},{dash_str}")

    if not pat_lines:
        return {"families": 0, "error": "Nada rasterizado (imagem vazia?)"}

    with open(output_path, 'w', encoding='ascii', errors='replace') as f:
        f.write(f"*{safe}, Gerado por Meu Conversor CAD\n")
        f.write("\n".join(pat_lines) + "\n")

    return {
        "families": len(pat_lines),
        "tile": f"{W:.1f}x{H:.1f}",
        "resolution": f"{cols}x{rows}",
    }


def image_to_pat(image_path: str, output_path: str, name: str = "CUSTOM",
                 modo: str = "auto") -> dict:
    """
    Converte IMAGEM direto em PAT (sem passar por DXF).
    modo="linhas" -> deteccao de borda (Canny). Para fotos/renders.
    modo="tracos" -> threshold Otsu (pixels escuros = desenho). Para
                     desenhos de padrao em preto e branco (piso, textura).
    modo="auto"   -> decide sozinho: se a imagem for majoritariamente
                     clara com tracos escuros usa 'tracos', senao 'linhas'.
    Tile = dimensoes em pixels da imagem (ajuste a escala no AutoCAD).
    """
    from PIL import Image as PILImage
    pil = PILImage.open(image_path).convert("RGBA")
    bg = PILImage.new("RGBA", pil.size, (255, 255, 255, 255))
    bg.paste(pil, mask=pil.split()[3])
    gray = cv2.cvtColor(np.array(bg.convert("RGB")), cv2.COLOR_RGB2GRAY)

    # reduzir para a resolucao de varredura mantendo aspect ratio
    H0, W0 = gray.shape
    rows = TARGET_ROWS
    cols = max(1, int(round(TARGET_ROWS * W0 / H0)))
    gray = cv2.resize(gray, (cols, rows), interpolation=cv2.INTER_AREA)

    if modo == "auto":
        # se a media for clara (fundo branco com tracos), usa threshold
        modo = "tracos" if gray.mean() > 160 else "linhas"

    if modo == "tracos":
        _, binary = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:  # linhas
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        binary = cv2.Canny(blur, 40, 120)
        binary = cv2.dilate(binary, np.ones((2, 2), np.uint8), iterations=1)

    # normaliza o tile: altura = 100 unidades (escala previsivel no AutoCAD).
    # pixels nao tem unidade real, entao a arquiteta escala 1 azulejo = 100 * scale.
    TILE_BASE = 100.0
    H = TILE_BASE
    W = TILE_BASE * W0 / H0
    result = _bitmap_to_pat(binary, W, H, output_path, name)
    result["modo"] = modo
    result["origem"] = "imagem"
    result["tile"] = f"{W:.1f}x{H:.1f}"
    return result


def _rasterize_svg(svg_path: str, png_path: str, width: int = 1200) -> bool:
    """Rasteriza um SVG para PNG (fundo transparente) usando inkscape ou
    ImageMagick. Retorna True se gerou o arquivo."""
    ink = shutil.which("inkscape")
    if ink:
        # inkscape 1.x
        cmds = [
            [ink, svg_path, "--export-type=png",
             f"--export-filename={png_path}", "-w", str(width)],
            # inkscape 0.9x (fallback)
            [ink, "-z", "-e", png_path, "-w", str(width), svg_path],
        ]
        for c in cmds:
            try:
                subprocess.run(c, check=True, capture_output=True, timeout=60)
                if os.path.exists(png_path) and os.path.getsize(png_path) > 0:
                    return True
            except Exception:
                continue
    conv = shutil.which("convert")
    if conv:
        try:
            subprocess.run([conv, "-density", "200", "-background", "none",
                            svg_path, png_path], check=True,
                           capture_output=True, timeout=60)
            return os.path.exists(png_path) and os.path.getsize(png_path) > 0
        except Exception:
            pass
    return False


def svg_to_pat(svg_path: str, output_path: str, name: str = "CUSTOM",
               modo: str = "auto") -> dict:
    """Converte SVG em PAT: rasteriza o SVG em alta resolucao e reusa a
    varredura de imagem. SVG e vetorial, entao o resultado fica bem nitido."""
    png_tmp = svg_path + ".raster.png"
    try:
        if not _rasterize_svg(svg_path, png_tmp):
            return {"families": 0,
                    "error": "Nao consegui rasterizar o SVG (inkscape/ImageMagick)"}
        result = image_to_pat(png_tmp, output_path, name=name, modo=modo)
        result["origem"] = "svg"
        return result
    finally:
        if os.path.exists(png_tmp):
            os.remove(png_tmp)
