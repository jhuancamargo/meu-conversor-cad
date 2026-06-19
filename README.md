# Meu Conversor CAD

Aplicação web que converte imagens (fotos, croquis, plantas) em arquivos **DXF editáveis** para AutoCAD e SketchUp, utilizando Inteligência Artificial.

Desenvolvido como projeto extensionista do módulo **Bootcamp Desenvolvimento Inteligência Artificial** — ADS · UNISAGRADO EaD · 2026.

Parceria com **The JenArch – Escritório de Arquitetura** (Bauru/SP).

---

## Modos de conversão

| Modo | Algoritmo | Indicado para |
|------|-----------|---------------|
| **IA** | HED (deep learning) | Fotos com sombra, croquis, iluminação irregular |
| **Clássico** | Otsu (binarização) | Scans com fundo branco uniforme |
| **Vetorial** | Canny + contornos | Renders, screenshots, imagens digitais limpas |

---

## Como o HED funciona

O **HED (Holistically-Nested Edge Detection)** é uma rede neural convolucional pré-treinada que detecta bordas em múltiplas escalas simultaneamente. Para cada pixel, retorna uma probabilidade de 0 a 1 de ser borda — funcionando mesmo em fotos com sombra e ruído onde métodos clássicos falham.

Pipeline: `imagem → HED (mapa de bordas) → threshold → esqueleto → findContours → Douglas-Peucker → DXF`

Modelo: [`lllyasviel/Annotators`](https://huggingface.co/lllyasviel/Annotators) via Hugging Face (download automático na primeira execução).

---

## Resultados

Foto com sombra/ruído:
- **Otsu:** 5.340 linhas / 1,6 MB — inutilizável
- **HED:** 187 linhas / 0,17 MB — **9× menor**, editável no AutoCAD

Planta arquitetônica real (The JenArch):
- Otsu: 2.347 linhas → HED: 828 linhas

Projeto Suite Villa Barcelona (PDF A3):
- Vetorial: 1.436 polylines em 0,5 segundo

---

## Evidências

A pasta [`evidencias/`](evidencias/) contém as comparações visuais e os arquivos DXF gerados a partir de imagens reais do escritório:

- `01_comparacao_otsu_vs_hed.png` — comparação principal
- `03_real_planta.png` — planta arquitetônica real convertida
- `04_poltrona_comparacao.png` — elemento do projeto Suite Villa Barcelona
- `metricas.json` — métricas numéricas dos testes

---

## Como rodar

```bash
cd meu-conversor-cad-main
pip install -r requirements.txt
PYTHONPATH=worker-engine python3 -m uvicorn worker-engine.app:app --port 8001
```

Acesse: http://localhost:8001

---

## Tecnologias

- **Python 3.11** · FastAPI · uvicorn
- **PyTorch** · controlnet_aux (HED) · Hugging Face
- **OpenCV** · scikit-image · sknw · SciPy · ezdxf · Pillow
- **HTML** + Tailwind CSS (frontend drag & drop)

---

## Estrutura

```
worker-engine/
  app.py          # API FastAPI (3 modos: ia / classico / vetorial)
  ai_pipeline.py  # Pipeline HED (deep learning)
  index.html      # Frontend drag & drop
evidencias/       # Comparações visuais e DXFs reais
requirements.txt
```

---

**Jhuan Yury Souto Camargo · Grupo 7 · ADS · UNISAGRADO 2026**
