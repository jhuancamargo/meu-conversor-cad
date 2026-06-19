"""
Pipeline de vetorizacao Imagem -> CAD (DXF) com IA.

Diferenca para o pipeline antigo (worker.py / app.py):
- Antes: binarizacao por Otsu (so funciona em fundo branco perfeito).
- Agora: deteccao de bordas por DEEP LEARNING (HED - Holistically-Nested
  Edge Detection), modelo neural pre-treinado. Funciona em fotos, croquis
  e plantas escaneadas com sombra/ruido, que e o caso de uso real.

O modelo HED e baixado automaticamente do Hugging Face (lllyasviel/Annotators)
na primeira execucao e roda em CPU ou GPU.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2
import ezdxf
from skimage.morphology import skeletonize
import sknw
from scipy.interpolate import splprep, splev
from PIL import Image

DXF_VERSION = "R2010"
_HED = None  # cache do modelo (carrega 1x)


def _get_hed():
    """Carrega o detector HED uma unica vez (singleton)."""
    global _HED
    if _HED is None:
        from controlnet_aux import HEDdetector
        _HED = HEDdetector.from_pretrained("lllyasviel/Annotators")
    return _HED


def hed_edges(image_path: str) -> np.ndarray:
    """Roda o HED e devolve um mapa de bordas (uint8, bordas claras em fundo escuro),
    no MESMO tamanho da imagem original."""
    pil = Image.open(image_path).convert("RGB")
    w, h = pil.size
    res = max(w, h)  # mantem a resolucao original (sem perder detalhe)
    hed = _get_hed()
    edge = hed(pil, detect_resolution=res, image_resolution=res)
    edge = edge.convert("L").resize((w, h), Image.BILINEAR)
    return np.array(edge)


def edges_to_dxf(edge_map: np.ndarray, output_path: str,
                 thresh: int = 30, scale: float = 2.0,
                 min_pts: int = 8, epsilon: float = 0.05) -> dict:
    """Converte mapa de bordas HED em DXF.
    Pipeline: threshold direto no mapa HED -> fecha lacunas -> findContours -> Douglas-Peucker -> DXF."""
    h, w = edge_map.shape[:2]

    # 1. Threshold direto no mapa HED
    normalized = cv2.normalize(edge_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, binary = cv2.threshold(normalized, 25, 255, cv2.THRESH_BINARY)

    # 2. Esqueleto: centraliza bordas grossas do HED em 1 pixel (elimina tremor)
    skeleton = skeletonize(binary > 127).astype(np.uint8) * 255

    # 3. Dilata levemente para findContours conectar pontos proximos
    kernel = np.ones((2, 2), np.uint8)
    skeleton = cv2.dilate(skeleton, kernel, iterations=1)

    contours, _ = cv2.findContours(skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # epsilon proporcional ao tamanho da imagem
    eps = max(1.5, min(w, h) * 0.004)

    doc = ezdxf.new(dxfversion=DXF_VERSION)
    msp = doc.modelspace()
    n = 0
    for cnt in contours:
        if cv2.arcLength(cnt, False) < 20:
            continue
        approx = cv2.approxPolyDP(cnt, eps, False)
        pts = [(float(p[0][0]), float(h - p[0][1])) for p in approx]
        if len(pts) >= 2:
            msp.add_lwpolyline(pts, dxfattribs={"layer": "IA_HED"})
            n += 1
    doc.saveas(output_path)
    return {"polylines": n, "edges_total": len(contours)}


def convert(image_path: str, output_path: str) -> dict:
    """Pipeline completo com IA: imagem -> bordas HED -> DXF."""
    edges = hed_edges(image_path)
    return edges_to_dxf(edges, output_path)


if __name__ == "__main__":
    import sys, time
    inp = sys.argv[1] if len(sys.argv) > 1 else "vectorizer/teste.png"
    out = sys.argv[2] if len(sys.argv) > 2 else "saida_ia.dxf"
    t = time.time()
    info = convert(inp, out)
    print(f"OK {info} em {time.time()-t:.2f}s -> {out}")
