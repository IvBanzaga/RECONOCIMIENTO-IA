import cv2
import face_recognition
import tensorflow as tf

# Imagen a comparar
image = cv2.imread("Images/yo.jpg")
face_loc = face_recognition.face_locations(image)[0]
face_image_encodings = face_recognition.face_encodings(image, known_face_locations=[face_loc])[0]

######################################################################################
# Video Streaming

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
     ret, frame = cap.read()
     if ret == False: break
     frame = cv2.flip(frame, 1)

     face_locations = face_recognition.face_locations(frame)
     if face_locations != []:
          for face_location in face_locations:
               face_frame_encodings = face_recognition.face_encodings(frame, known_face_locations=[face_location])[0]
               result = face_recognition.compare_faces([face_frame_encodings], face_image_encodings)
               print("Result:", result)

               if result[0] == True:
                    text = "Reconocido"
                    color = (125, 220, 0)
               else:
                    text = "Desconocido"
                    color = (50, 50, 255)

               cv2.rectangle(frame, (face_location[3], face_location[2]), (face_location[1], face_location[2] + 30), color, -1)
               cv2.rectangle(frame, (face_location[3], face_location[0]), (face_location[1], face_location[2]), color, 2)
               cv2.putText(frame, text, (face_location[3], face_location[2] + 20), 2, 0.7, (255, 255, 255), 1)

     cv2.imshow("Frame", frame)
     k = cv2.waitKey(1)
     if k == 27 & 0xFF:
          break

cap.release()
cv2.destroyAllWindows()

# AÑADIMOS

# Crear y guardar el modelo en formato h5
modelo = tf.keras.models.Sequential()
# Agrega las capas de tu modelo aquí
modelo.save('reconocimiento.h5')

# El equipo es Linux. Listemos el contenido de la carpeta actual para verificar que se exportó el modelo
ls

# Para convertirlo a TensorFlow.js, primero debemos instalar la librería
pip install tensorflowjs

# Crear carpeta donde se colocarán los archivos resultantes
!mkdir carpeta_salida

# Realizar la exportación a la carpeta de salida
!tensorflowjs_converter --input_format keras reconocimiento.h5 carpeta_salida

# Confirmar que en la carpeta de salida se hayan generado los archivos. Deben aparecer archivos "bin" y "json"
!ls carpeta_salida
