import cv2
import numpy as np
import os
import uuid
import logging
import ezdxf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from starlette.background import BackgroundTask
from skimage.morphology import skeletonize
import sknw
from scipy.interpolate import splprep, splev

log = logging.getLogger("conversor")

app = FastAPI()
DXF_VERSION = "R2010"

# Limite de upload. Acima disso a conversao consome memoria demais e
# derruba o servidor (o modo classico faz upscale 4x da imagem).
# PDFs de planta costumam ser maiores, mas so sao lidos como vetor.
MAX_UPLOAD_MB = 60


def _save_upload(file: UploadFile, input_path: str) -> None:
    """Grava o upload em disco respeitando MAX_UPLOAD_MB.
    Le em blocos para nao carregar o arquivo inteiro na memoria."""
    limit = MAX_UPLOAD_MB * 1024 * 1024
    total = 0
    with open(input_path, "wb") as buf:
        while chunk := file.file.read(1024 * 1024):
            total += len(chunk)
            if total > limit:
                buf.close()
                os.remove(input_path)
                raise HTTPException(
                    413, f"Arquivo muito grande (limite {MAX_UPLOAD_MB} MB)")
            buf.write(chunk)


def _cleanup(*paths):
    """BackgroundTask que remove os temporarios DEPOIS de enviar a resposta.
    O FileResponse precisa do arquivo em disco durante o envio, por isso a
    limpeza nao pode ficar num finally."""
    def _rm():
        for p in paths:
            try:
                os.remove(p)
            except OSError:
                pass
    return BackgroundTask(_rm)

def extract_vetorial_dxf(image_path: str, output_path: str) -> bool:
    """Modo vetorial: Canny + findContours. Ideal para imagens digitais limpas (renders, capturas de tela)."""
    from PIL import Image as PILImage, ImageEnhance
    pil = PILImage.open(image_path).convert("RGBA")
    bg = PILImage.new("RGBA", pil.size, (255, 255, 255, 255))
    bg.paste(pil, mask=pil.split()[3])
    img = cv2.cvtColor(np.array(bg.convert("RGB")), cv2.COLOR_RGB2GRAY)
    img = np.array(ImageEnhance.Contrast(PILImage.fromarray(img)).enhance(2.0))
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    edges = cv2.Canny(blur, 30, 90)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    h = img.shape[0]
    doc = ezdxf.new(dxfversion=DXF_VERSION)
    msp = doc.modelspace()
    n = 0
    for cnt in contours:
        if cv2.arcLength(cnt, False) < 30:
            continue
        approx = cv2.approxPolyDP(cnt, 1.5, False)
        pts = [(float(p[0][0]), float(h - p[0][1])) for p in approx]
        if len(pts) >= 2:
            msp.add_lwpolyline(pts, dxfattribs={"layer": "VETORIAL"})
            n += 1
    doc.saveas(output_path)
    return n > 0


def extract_optimized_quality_dxf(image_path: str, output_path: str,
                                  nivel: int = 7) -> bool:
    """nivel 0-10 controla o detalhe das linhas.
    0 = mais simplificado (linhas retas, poucos segmentos);
    10 = maximo detalhe (segue todas as curvas)."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return False

    nivel = max(0, min(10, int(nivel)))
    # epsilon do Douglas-Peucker: alto = simplifica, baixo = detalha.
    # mapeamento exponencial entre 5.0 (nivel 0) e 0.02 (nivel 10).
    epsilon = 0.02 * (250.0 ** ((10 - nivel) / 10.0))
    # filtro de poeira: em niveis baixos remove mais fragmentos pequenos.
    dust = 5 + (10 - nivel)  # 5 (nivel 10) ate 15 (nivel 0)

    # 1. Upscale para precisão matemática (4x).
    # Teto de seguranca: o esqueleto/grafo/spline sao libs nativas e estouram
    # (SIGSEGV, mata o processo) em imagens grandes. Limitamos a area final
    # para que o upscale nunca ultrapasse MAX_PIXELS.
    MAX_PIXELS = 12_000_000  # ~3460x3460 depois do upscale
    scale = 4.0
    if img.shape[0] * img.shape[1] * scale * scale > MAX_PIXELS:
        scale = max(1.0, (MAX_PIXELS / (img.shape[0] * img.shape[1])) ** 0.5)
    gray = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)),
                      interpolation=cv2.INTER_LANCZOS4)

    # 2. Binarização (kernel 3x3: suaviza o serrilhado do upscale)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. Esqueletização (1 pixel)
    skeleton = skeletonize(binary > 127)

    # 4. Construção do Grafo Topológico
    graph = sknw.build_sknw(skeleton.astype(np.uint16))

    doc = ezdxf.new(dxfversion=DXF_VERSION)
    msp = doc.modelspace()
    img_height = skeleton.shape[0]
    n = 0

    for (start_node, end_node, edge_data) in graph.edges(data=True):
        pts = edge_data['pts']
        
        # Filtro de poeira (evita linhas microscópicas soltas)
        if len(pts) < dust: continue

        x_coords = pts[:, 1] / scale
        y_coords = (img_height - pts[:, 0]) / scale

        try:
            # 5. SPLINE COM FIDELIDADE TOTAL (s=0.0)
            tck, u = splprep([x_coords, y_coords], s=0.0, k=min(3, len(x_coords)-1))
            
            # AMUSTRAGEM ADAPTATIVA: Usamos metade dos pontos originais
            # para gerar a curva inicial, em vez de dobrar.
            num_pts = max(10, int(len(x_coords) / 2)) 
            u_fine = np.linspace(0, 1, num=num_pts)
            x_fine, y_fine = splev(u_fine, tck)

            fit_pts = [(float(x), float(y)) for x, y in zip(x_fine, y_fine)]

            # 6. O PULO DO GATO: Simplificação Douglas-Peucker (Redução de Pontos)
            # Analisa o vetor e remove vértices redundantes.
            curve_pts = np.array(fit_pts, dtype=np.float32).reshape(-1, 1, 2)
            
            # epsilon vem do nivel escolhido (ver topo da funcao).
            approx = cv2.approxPolyDP(curve_pts, epsilon, closed=False)
            
            final_pts = [(float(p[0][0]), float(p[0][1])) for p in approx]
            
            msp.add_lwpolyline(final_pts, dxfattribs={"layer": "NITIDEZ_OTIMIZADA"})
            n += 1

        except Exception as exc:
            # Fallback direto em caso de erro matemático (spline degenerada).
            # Registra a causa: silenciar aqui ja escondeu bugs no passado.
            log.debug("spline falhou, usando pontos brutos: %s", exc)
            fit_pts = [(float(x_coords[i]), float(y_coords[i])) for i in range(len(x_coords))]
            msp.add_lwpolyline(fit_pts, dxfattribs={"layer": "NITIDEZ_OTIMIZADA"})
            n += 1

    doc.saveas(output_path)
    return n > 0

# --- ROTAS DA API --- (Mantidas as mesmas do seu código original)
@app.get("/", response_class=HTMLResponse)
async def read_index():
    _index = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    with open(_index, "r", encoding="utf-8") as f:
        return f.read()

@app.post("/convert")
async def convert_image(file: UploadFile = File(...), modo: str = "ia",
                        nivel: int = 7):
    """Converte imagem em DXF.
    modo="ia"       -> deep learning (HED). Robusto a foto/sombra/ruido.
    modo="classico" -> binarizacao Otsu. Para scans com fundo branco perfeito.
    modo="vetorial" -> Canny + contornos. Para imagens digitais limpas (renders, capturas).
    nivel 0-10      -> (so no modo classico) detalhe das linhas: 0 simples, 10 detalhado.
    """
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".png",".jpg",".jpeg",".tiff",".bmp"):
        raise HTTPException(400, "Formato não suportado")

    uid = uuid.uuid4().hex
    input_path = f"input_{uid}{ext}"
    output_path = f"output_{uid}.dxf"

    _save_upload(file, input_path)

    try:
        if modo == "classico":
            success = extract_optimized_quality_dxf(input_path, output_path, nivel=nivel)
        elif modo == "vetorial":
            success = extract_vetorial_dxf(input_path, output_path)
        else:
            from ai_pipeline import convert as convert_ia
            info = convert_ia(input_path, output_path)
            success = info.get("polylines", 0) > 0

        if not success:
            # Nao e falha do servidor: a imagem nao tinha tracos detectaveis.
            raise HTTPException(
                422, "Nenhum contorno detectado nesta imagem. Tente outro modo "
                     "ou uma imagem com traços mais nítidos e bom contraste.")

        stem = os.path.splitext(file.filename)[0]
        sufixos = {"ia": "IA_HED", "classico": "LEVE_v2", "vetorial": "VETORIAL"}
        sufixo = sufixos.get(modo, "IA_HED")
        return FileResponse(path=output_path,
                            filename=f"{stem}_{sufixo}.dxf",
                            media_type="application/dxf",
                            background=_cleanup(input_path, output_path))
    except Exception:
        _cleanup(input_path, output_path).func()
        raise


@app.post("/convert-pat")
async def convert_to_pat(file: UploadFile = File(...), modo: str = "auto"):
    """Converte DXF OU imagem em padrão de hachura AutoCAD (.PAT).
    - .dxf                 -> varredura da geometria vetorial
    - .png/.jpg/.jpeg/...  -> varredura da imagem (modo=auto|tracos|linhas)
    """
    ext = os.path.splitext(file.filename)[1].lower()
    dxf_exts = (".dxf",)
    svg_exts = (".svg",)
    img_exts = (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp")
    if ext not in dxf_exts + svg_exts + img_exts:
        raise HTTPException(400, "Envie um .DXF, .SVG ou uma imagem (PNG/JPG)")

    uid = uuid.uuid4().hex
    input_path = f"input_{uid}{ext}"
    output_path = f"output_{uid}.pat"

    _save_upload(file, input_path)

    try:
        stem = os.path.splitext(file.filename)[0]
        if ext in dxf_exts:
            from pat_pipeline import dxf_to_pat
            result = dxf_to_pat(input_path, output_path, name=stem)
        elif ext in svg_exts:
            from pat_pipeline import svg_to_pat
            result = svg_to_pat(input_path, output_path, name=stem, modo=modo)
        else:
            from pat_pipeline import image_to_pat
            result = image_to_pat(input_path, output_path, name=stem, modo=modo)

        if result.get("error"):
            raise HTTPException(500, result["error"])

        headers = {"X-Families": str(result["families"])}
        if "segments" in result: headers["X-Segments"] = str(result["segments"])
        if "modo" in result:     headers["X-Modo"] = str(result["modo"])
        if "tile" in result:     headers["X-Tile"] = str(result["tile"])
        return FileResponse(
            path=output_path,
            filename=f"{stem}.pat",
            media_type="application/octet-stream",
            headers=headers,
            background=_cleanup(input_path, output_path),
        )
    except Exception:
        _cleanup(input_path, output_path).func()
        raise


@app.post("/convert-pdf")
async def convert_pdf_to_dxf(file: UploadFile = File(...), pagina: int = 0,
                             texto: bool = True):
    """Converte PDF vetorial (planta exportada de CAD) em DXF.
    pagina -> indice da pagina, 0 = primeira
    texto  -> traz as cotas/legendas como TEXT editavel
    """
    ext = os.path.splitext(file.filename)[1].lower()
    if ext != ".pdf":
        raise HTTPException(400, "Envie um arquivo .PDF")

    uid = uuid.uuid4().hex
    input_path = f"input_{uid}{ext}"
    output_path = f"output_{uid}.dxf"

    _save_upload(file, input_path)

    try:
        from pdf_pipeline import pdf_to_dxf
        stem = os.path.splitext(file.filename)[0]
        result = pdf_to_dxf(input_path, output_path,
                            pagina=pagina, incluir_texto=texto)

        if result.get("error"):
            # PDF escaneado (sem vetor) nao e erro do servidor: e o
            # arquivo errado para esta aba.
            raise HTTPException(422, result["error"])

        return FileResponse(
            path=output_path,
            filename=f"{stem}.dxf",
            media_type="application/dxf",
            headers={
                "X-Entidades": str(result["entidades"]),
                "X-Textos": str(result["textos"]),
                "X-Layers": str(result["layers"]),
                "X-Paginas": str(result["paginas"]),
                "X-Formato": str(result["formato"]),
            },
            background=_cleanup(input_path, output_path),
        )
    except Exception:
        _cleanup(input_path, output_path).func()
        raise


@app.post("/pdf-info")
async def pdf_info(file: UploadFile = File(...)):
    """Quantas paginas tem o PDF -- o frontend usa para oferecer a escolha."""
    if os.path.splitext(file.filename)[1].lower() != ".pdf":
        raise HTTPException(400, "Envie um arquivo .PDF")

    uid = uuid.uuid4().hex
    input_path = f"input_{uid}.pdf"
    _save_upload(file, input_path)
    try:
        from pdf_pipeline import contar_paginas
        return {"paginas": contar_paginas(input_path)}
    finally:
        try:
            os.remove(input_path)
        except OSError:
            pass


@app.post("/convert-kml")
async def convert_kml_to_dxf(file: UploadFile = File(...), local: bool = False):
    """Converte KML/KMZ (Google Earth) em DXF georreferenciado (UTM/metros).
    local=false -> coordenadas UTM reais (georreferenciado)
    local=true  -> deslocado para a origem (0,0), mantendo as medidas.
    """
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".kml", ".kmz"):
        raise HTTPException(400, "Envie um arquivo .KML ou .KMZ")

    uid = uuid.uuid4().hex
    input_path = f"input_{uid}{ext}"
    output_path = f"output_{uid}.dxf"

    _save_upload(file, input_path)

    try:
        from kml_pipeline import kml_to_dxf
        stem = os.path.splitext(file.filename)[0]
        result = kml_to_dxf(input_path, output_path, local=local)

        if result.get("error"):
            raise HTTPException(500, result["error"])

        return FileResponse(
            path=output_path,
            filename=f"{stem}.dxf",
            media_type="application/dxf",
            headers={
                "X-Placemarks": str(result["placemarks"]),
                "X-Entidades": str(result["entidades"]),
                "X-Zona": str(result["zona_utm"]),
                "X-Coordenadas": str(result["coordenadas"]),
            },
            background=_cleanup(input_path, output_path),
        )
    except Exception:
        _cleanup(input_path, output_path).func()
        raise