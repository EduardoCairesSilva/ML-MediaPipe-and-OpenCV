import cv2
import mediapipe as mp

mao = cv2.VideoCapture(0)

hand = mp.solutions.hands
hand1 = hand.Hands(max_num_hands=1)
mpDesenho = mp.solutions.drawing_utils

while True:
    verificar, img = mao.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resultados = hand1.process(imgRGB)
    maoPontos = resultados.multi_hand_landmarks
    a, l, _ = img.shape
    points = []
    if maoPontos:
        for pontos in maoPontos:
            print(pontos)
            mpDesenho.draw_landmarks(img, pontos, hand.HAND_CONNECTIONS)
            for id, cord in enumerate(pontos.landmark):
                cx, cy = int(cord.x*l), int(cord.y*a)
                cv2.putText(img, str(id), (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                points.append((cx, cy))

        dedos = [8, 12, 16, 20]
        count = 0
        if pontos:
            if points[4][0] < points[2][0]:
                count += 1
            for x in dedos:
                if points[x][1] < points[x-2][1]:
                    count += 1
        cv2.rectangle(img, (80, 10), (200, 100), (0, 0, 0), -1)
        cv2.putText(img, str(count), (100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 5)

    cv2.imshow("ReconhecimentoMao", img)
    if cv2.waitKey(1) == 27:
        break
