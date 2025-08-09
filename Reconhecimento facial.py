import mediapipe as mp
import cv2

webcam = cv2.VideoCapture(0)
reconhecimento_facial = mp.solutions.face_detection
reconhecimento_rosto = reconhecimento_facial.FaceDetection()
contorno = mp.solutions.drawing_utils

while True:
    verificador, frame = webcam.read()
    if not verificador:
        break

    lista_rostos = reconhecimento_rosto.process(frame)

    if lista_rostos.detections:
        for rosto in lista_rostos.detections:
            contorno.draw_detection(frame, rosto)
    cv2.imshow("Webcam ativa", frame)

    if cv2.waitKey(5) == 27:
        break

webcam.release()

# sdsdadsadadasdasasdasd