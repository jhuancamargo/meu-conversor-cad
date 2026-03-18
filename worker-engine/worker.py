import cv2
import numpy as np
import ezdxf
import os

def process_image_to_dxf(image_path, output_dxf_path):
    print(f"Lendo a imagem: {image_path}")
    
    # 1. Carrega a imagem
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Imagem não encontrada. Verifique o caminho.")

    # 2. Pré-processamento (Limpeza e Contraste)
    # Converte para tons de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Suaviza a imagem para remover "sujeira" do JPG/PNG (ruído)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Binarização: Transforma tudo estritamente em Preto e Branco
    # Usando Canny Edge Detection para achar as bordas das linhas
    edges = cv2.Canny(blurred, 50, 150)

    # 3. Detecção de Contornos Matemáticos
    # RETR_TREE pega a hierarquia (linhas dentro de linhas)
    # CHAIN_APPROX_SIMPLE remove pontos redundantes nas retas
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    print(f"Encontrados {len(contours)} contornos. Gerando arquivo CAD...")

    # 4. Criação do arquivo DXF (Formato aceito pelo AutoCAD, SketchUp, etc)
    # R2010 é uma versão estável e amplamente compatível
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()

    # Pega a altura da imagem para inverter o eixo Y 
    # (Imagens crescem para baixo, CAD cresce para cima)
    height = img.shape[0]

    # 5. Desenhando os vetores no CAD
    for contour in contours:
        # APROXIMAÇÃO MATEMÁTICA (O "Pulo do Gato"):
        # epsilon é a margem de erro. Quanto menor, mais fiel aos pixels (mas gera arquivos pesados/tremidos).
        # Quanto maior, mais retas perfeitas (mas pode arredondar cantos que não deveria).
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Extrai os pontos (X, Y) do contorno aproximado
        points = []
        for p in approx:
            x, y = p[0]
            # Inverte o Y para a planta não ficar de cabeça para baixo no AutoCAD
            points.append((x, height - y))

        # Desenha uma Polyline (LWPolyline é mais leve e padrão em CAD)
        if len(points) > 1:
            msp.add_lwpolyline(points, close=True)

    # 6. Salva o arquivo final
    doc.saveas(output_dxf_path)
    print(f"Sucesso! Arquivo DXF salvo em: {output_dxf_path}")

if __name__ == "__main__":
    # Teste local: Coloque uma imagem chamada 'planta_teste.jpg' na mesma pasta
    input_file = "planta_teste.jpg"
    output_file = "resultado.dxf"
    
    if os.path.exists(input_file):
        process_image_to_dxf(input_file, output_file)
    else:
        print(f"Coloque uma imagem com o nome '{input_file}' na pasta para testar.")