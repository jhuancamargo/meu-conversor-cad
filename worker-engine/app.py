import cv2
import numpy as np
import os
import uuid
import ezdxf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
import shutil
from skimage.morphology import skeletonize
import sknw
from scipy.interpolate import splprep, splev

app = FastAPI()
DXF_VERSION = "R2010"

def extract_optimized_quality_dxf(image_path: str, output_path: str) -> bool:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return False

    # 1. Upscale para precisão matemática (4x)
    scale = 4.0
    gray = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)),
                      interpolation=cv2.INTER_LANCZOS4)

    # 2. Binarização
    blur = cv2.GaussianBlur(gray, (1, 1), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. Esqueletização (1 pixel)
    skeleton = skeletonize(binary > 127)

    # 4. Construção do Grafo Topológico
    graph = sknw.build_sknw(skeleton.astype(np.uint16))

    doc = ezdxf.new(dxfversion=DXF_VERSION)
    msp = doc.modelspace()
    img_height = skeleton.shape[0]

    for (start_node, end_node, edge_data) in graph.edges(data=True):
        pts = edge_data['pts']
        
        # Filtro de poeira (evita linhas microscópicas soltas)
        if len(pts) < 5: continue

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
            
            # epsilon é a "Tolerância". Valor equilibrado para móveis.
            # Se ficar "quadrado", diminua para 0.02. Se ficar pesado, aumente para 0.1
            epsilon = 0.05 
            approx = cv2.approxPolyDP(curve_pts, epsilon, closed=False)
            
            final_pts = [(float(p[0][0]), float(p[0][1])) for p in approx]
            
            msp.add_lwpolyline(final_pts, dxfattribs={"layer": "NITIDEZ_OTIMIZADA"})

        except Exception:
            # Fallback direto em caso de erro matemático
            fit_pts = [(float(x_coords[i]), float(y_coords[i])) for i in range(len(x_coords))]
            msp.add_lwpolyline(fit_pts, dxfattribs={"layer": "NITIDEZ_OTIMIZADA"})

    doc.saveas(output_path)
    return True

# --- ROTAS DA API --- (Mantidas as mesmas do seu código original)
@app.get("/", response_class=HTMLResponse)
async def read_index():
    # Certifique-se de que o index.html existe na pasta
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/convert")
async def convert_image(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".png",".jpg",".jpeg",".tiff",".bmp"):
        raise HTTPException(400, "Formato não suportado")

    uid = uuid.uuid4().hex
    input_path = f"input_{uid}{ext}"
    output_path = f"output_{uid}.dxf"

    with open(input_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    try:
        success = extract_optimized_quality_dxf(input_path, output_path)
        if not success:
            raise HTTPException(500, "Falha no processamento")
            
        stem = os.path.splitext(file.filename)[0]
        return FileResponse(path=output_path,
                            filename=f"{stem}_LEVE_v2.dxf",
                            media_type="application/dxf")
    finally:
        if os.path.exists(input_path): os.remove(input_path)