import cv2
import pickle
import cvzone
import numpy as np
import json
import time
from datetime import datetime

cap = cv2.VideoCapture('assets/carPark.mp4')

# Controle de tempo para salvar o JSON periodicamente (sem sobrecarregar o disco)
last_save_time = 0
SAVE_INTERVAL = 0.5  # salva a cada meio segundo

# Carrega o arquivo de polígonos (com fallback para converter o arquivo antigo de retângulos)
try:
    with open('CarParkPosPoly', 'rb') as f:
        posList = pickle.load(f)
except:
    posList = []

if len(posList) == 0:
    try:
        with open('CarParkPos', 'rb') as f:
            oldPosList = pickle.load(f)
        width, height = 107, 47
        posList = [
            [(x, y), (x + width, y), (x + width, y + height), (x, y + height)]
            for (x, y) in oldPosList
        ]
    except:
        posList = []


def checkParkingSpace(imgPro):
    spaceCounter = 0

    for pts in posList:
        pts_arr = np.array(pts, dtype=np.int32)

        # Bounding box ao redor do polígono para recortar a ROI
        x, y, w, h = cv2.boundingRect(pts_arr)
        imgCrop = imgPro[y:y+h, x:x+w]

        # Máscara local para isolar apenas os pixels de dentro do polígono
        pts_local = pts_arr - [x, y]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts_local], 255)

        # Aplica a máscara na ROI
        imgMasked = cv2.bitwise_and(imgCrop, imgCrop, mask=mask)

        # Contagem de pixels e porcentagem de preenchimento
        area_total = cv2.countNonZero(mask)
        if area_total == 0:
            area_total = 1

        count = cv2.countNonZero(imgMasked)
        ratio = (count / area_total) * 100

        # Se menos de 20% da vaga estiver alterada/branca, consideramos livre
        if ratio < 20.0:
            color = (0, 255, 0)
            thickness = 3
            spaceCounter += 1
        else:
            color = (0, 0, 255)
            thickness = 2

        # Desenha o polígono da vaga e a porcentagem de ocupação
        cv2.polylines(img, [pts_arr], isClosed=True, color=color, thickness=thickness)
        cvzone.putTextRect(img, f"{int(ratio)}%", (x, y + h - 5), scale=0.8, thickness=1, offset=2, colorR=color)

    cvzone.putTextRect(img, f"Vagas disponiveis: {spaceCounter}/{len(posList)}", (10, 50), scale=2, thickness=3, offset=10, colorR=(0, 0, 0))
    return spaceCounter, len(posList)



while True:

    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    success, img = cap.read()
    if not success or img is None:
        continue

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY_INV, 25, 16)
    imgMedian = cv2.medianBlur(imgThreshold, 5)
    kernel = np.ones((3, 3), np.uint8)
    imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)


    free_spaces, total_spaces = checkParkingSpace(imgDilate)

    # Salva periodicamente no arquivo JSON para alimentar o site/API
    current_time = time.time()
    if current_time - last_save_time > SAVE_INTERVAL:
        occupied_spaces = total_spaces - free_spaces

        parking_data = {
            "updated_at": datetime.now().strftime("%H:%M:%S"),
            "total_geral": total_spaces + 20,
            "livres_geral": free_spaces + 12,
            "ocupadas_geral": occupied_spaces + 8,
            "andares": [
                {
                    "andar": 1,
                    "nome": "1º Andar (Câmera ao Vivo)",
                    "total": total_spaces,
                    "livres": free_spaces,
                    "ocupadas": occupied_spaces
                },
                {
                    "andar": 2,
                    "nome": "2º Andar",
                    "total": 20,
                    "livres": 12,
                    "ocupadas": 8
                }
            ]
        }

        try:
            with open('parking_data.json', 'w', encoding='utf-8') as f:
                json.dump(parking_data, f, indent=4, ensure_ascii=False)
        except Exception:
            pass

        last_save_time = current_time
          

    cv2.imshow("Image", img)
    # cv2.imshow("ImageBlur", imgBlur)
    # cv2.imshow("ImageThreshold", imgMedian)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()