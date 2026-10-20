import cv2
import pickle
import numpy as np

# Arquivo dedicado para os polígonos (mantém o CarParkPos antigo intacto)
FILE_NAME = 'CarParkPosPoly'

try:
    with open(FILE_NAME, 'rb') as f:
        posList = pickle.load(f)
except:
    posList = []

# Guarda os pontos temporários da vaga que está sendo clicada (até 4 pontos)
current_points = []


def mouseClick(events, x, y, flags, params):
    global current_points

    # Botão esquerdo: adiciona um ponto do polígono
    if events == cv2.EVENT_LBUTTONDOWN:
        current_points.append((x, y))
        if len(current_points) == 4:
            posList.append(current_points)
            current_points = []
            with open(FILE_NAME, 'wb') as f:
                pickle.dump(posList, f)

    # Botão direito: cancela pontos pendentes ou remove a vaga sob o cursor
    elif events == cv2.EVENT_RBUTTONDOWN:
        if len(current_points) > 0:
            current_points = []
        else:
            for i, pts in enumerate(posList):
                poly_arr = np.array(pts, dtype=np.int32)
                # Verifica se o clique com o botão direito está dentro do polígono
                if cv2.pointPolygonTest(poly_arr, (x, y), False) >= 0:
                    posList.pop(i)
                    break
            with open(FILE_NAME, 'wb') as f:
                pickle.dump(posList, f)


while True:
    img = cv2.imread('./assets/carParkImg.png')

    if img is None:
        print("Erro: não foi possível encontrar './assets/carParkImg.png'")
        break

    # Desenha as vagas já concluídas
    for pts in posList:
        poly_arr = np.array(pts, dtype=np.int32)
        cv2.polylines(img, [poly_arr], isClosed=True, color=(255, 0, 255), thickness=2)

    # Desenha os pontos/linhas da vaga em andamento
    for pt in current_points:
        cv2.circle(img, pt, 5, (0, 255, 255), -1)

    if len(current_points) > 1:
        cv2.polylines(img, [np.array(current_points, dtype=np.int32)], isClosed=False, color=(0, 255, 255), thickness=2)

    # Informações na tela
    info_text = f"Vagas marcadas: {len(posList)} | Pontos da vaga atual: {len(current_points)}/4"
    cv2.putText(img, info_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(img, "Esq: marca ponto | Dir: cancela/apaga | Q: sair", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow('Demarcador de Vagas (Poligonos)', img)
    cv2.setMouseCallback('Demarcador de Vagas (Poligonos)', mouseClick)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()