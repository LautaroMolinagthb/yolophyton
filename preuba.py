import cv2
import numpy as np

# Cargar los archivos de configuración y pesos de YOLOv3
net = cv2.dnn.readNet('model/yolov3.weights', 'model/yolov3.cfg')
print(net)
classes = []
with open('model/coco.names', 'r') as f:
    classes = f.read().splitlines()

# Iniciar la captura de video desde la cámara web
cap = cv2.VideoCapture(0)  # El argumento '0' indica la cámara predeterminada

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocesamiento de la imagen para YOLOv3
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Realizar la detección y obtener las salidas
    outs = net.forward(net.getUnconnectedOutLayersNames())

    conf_threshold = 0.5  # Umbral de confianza
    nms_threshold = 0.4   # Umbral para la supresión de no máximos

    class_ids = []
    confidences = []
    boxes = []

    # Procesar las salidas para obtener las cajas delimitadoras, confianzas y clases
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                # Coordenadas de la caja delimitadora
                center_x, center_y, width, height = (detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])).astype(int)
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                boxes.append([x, y, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Aplicar supresión de no máximos para eliminar detecciones redundantes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes[i]
            x, y, w, h = box
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Mostrar el video con las detecciones
    cv2.imshow('YOLOv3 Object Detection', frame)
    key = cv2.waitKey(1)
    if key == 27:  # Presiona 'ESC' para salir
        break

cap.release()
cv2.destroyAllWindows()

