import os
import tensorflow as tf
import cv2
import face_recognition

# Ruta a la carpeta que contiene las imágenes de referencia
carpeta_imagenes = "images/"

# Cargamos las imágenes de referencia y sus nombres de archivo
reference_images = []  # Almacenará las imágenes de referencia
reference_encodings = []  # Almacenará las codificaciones de las caras en las imágenes de referencia
reference_image_names = []  # Almacenará los nombres de archivo de las imágenes de referencia

# Iteramos a través de los archivos en la carpeta de imágenes
for filename in os.listdir(carpeta_imagenes):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):  # Asegurarse de que solo procesamos archivos de imagen
        image_path = os.path.join(carpeta_imagenes, filename)
        reference_image = cv2.imread(image_path)  # Cargamos una imagen de referencia
        face_loc = face_recognition.face_locations(reference_image)[0]  # Localizamos una cara en la imagen
        face_image_encodings = face_recognition.face_encodings(reference_image, known_face_locations=[face_loc])[0]  # Codificamos la cara
        reference_images.append(reference_image)  # Agregamos la imagen a la lista
        reference_encodings.append(face_image_encodings)  # Agregamos la codificación a la lista
        reference_image_names.append(os.path.splitext(filename)[0])  # Obtenemos el nombre sin extensión y lo agregamos a la lista

# Video Streaming
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Iniciamos la captura de video desde la cámara

while True:
    ret, frame = cap.read()  # Capturamos un frame del video
    if ret == False:  # Si no se pudo capturar un frame, salimos del bucle
        break
    frame = cv2.flip(frame, 1)  # Volteamos horizontalmente el frame para evitar espejear

    face_locations = face_recognition.face_locations(frame)  # Localizamos las caras en el frame
    if face_locations:  # Si se encontraron caras en el frame
        for face_location in face_locations:
            face_frame_encodings = face_recognition.face_encodings(frame, known_face_locations=[face_location])[0]  # Codificamos la cara en el frame

            # Comparamos la codificación de la cara en el frame con las codificaciones de las imágenes de referencia
            results = face_recognition.compare_faces(reference_encodings, face_frame_encodings)

            recognized = False
            for i, result in enumerate(results):
                if result:
                    recognized = True
                    text = f"{reference_image_names[i]}"  # Obtenemos el nombre de la imagen de referencia correspondiente
                    color = (125, 220, 0)  # Color para la caja y el texto
                    break

            if not recognized:
                text = "Desconocido"
                color = (50, 50, 255)  # Color para la caja y el texto en caso de que no se reconozca

            cv2.rectangle(frame, (face_location[3], face_location[2]), (face_location[1], face_location[2] + 30), color, -1)  # Dibuja un rectángulo para el texto
            cv2.rectangle(frame, (face_location[3], face_location[0]), (face_location[1], face_location[2]), color, 2)  # Dibuja un rectángulo alrededor de la cara
            cv2.putText(frame, text, (face_location[3], face_location[2] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)  # Muestra el nombre "Desconocido" en el frame

    cv2.imshow("Frame", frame)  # Muestra el frame en una ventana
    k = cv2.waitKey(1)
    if k == 27 & 0xFF:  # Si se presiona la tecla ESC, salimos del bucle
        break


