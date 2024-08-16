# import tensorflow as tf
# from tensorflow.keras import layers, models
# import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow.keras.preprocessing import image
# #import cv2

# # Cargar el conjunto de datos CIFAR-10
# (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# train_images, test_images = train_images / 255.0, test_images / 255.0


# class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
#                'dog', 'frog', 'horse', 'ship', 'truck']


# model = models.Sequential([
#     layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.Flatten(),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(10, activation='softmax')  #10 clases en CIFAR-10
# ])

# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])


# model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
# print(f'\nTest accuracy: {test_acc}')


# predictions = model.predict(test_images)

# def plot_image(i, predictions_array, true_label, img):
#     predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
#     plt.grid(False)
#     plt.xticks([])
#     plt.yticks([])

#     plt.imshow(img, cmap=plt.cm.binary)

#     predicted_label = np.argmax(predictions_array)
#     true_label = true_label[0] 
#     if predicted_label == true_label:
#         color = 'blue'
#     else:
#         color = 'red'

#     plt.xlabel(f"{class_names[predicted_label]} ({100*np.max(predictions_array):2.0f}%) ({class_names[true_label]})",
#                color=color)

# def plot_value_array(i, predictions_array, true_label):
#     predictions_array, true_label = predictions_array[i], true_label[i]
#     plt.grid(False)
#     plt.xticks(range(10))
#     plt.yticks([])
#     thisplot = plt.bar(range(10), predictions_array, color="#777777")
#     plt.ylim([0, 1])
#     predicted_label = np.argmax(predictions_array)
#     true_label = true_label[0]  # Extraer el valor del array true_label

#     thisplot[predicted_label].set_color('red')
#     thisplot[true_label].set_color('blue')


# i = 0
# plt.figure(figsize=(6,3))
# plt.subplot(1,2,1)
# plot_image(i, predictions, test_labels, test_images)
# plt.subplot(1,2,2)
# plot_value_array(i, predictions, test_labels)
# plt.show()

# model.save('mi_modelo.h5')

# model = tf.keras.models.load_model('mi_modelo.h5')
####################################################################
# import tensorflow as tf
# from tensorflow.keras import layers, models
# import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow.keras.preprocessing import image

# # Cargar el modelo guardado
# model = tf.keras.models.load_model('mi_modelo.h5')

# # Definir los nombres de las clases
# class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
#                'dog', 'frog', 'horse', 'ship', 'truck']

# # Función para cargar y preprocesar una imagen
# def load_and_preprocess_image(img_path):
#     img = image.load_img(img_path, target_size=(32, 32))  # Cargar imagen y redimensionar a 32x32
#     img_array = image.img_to_array(img)  # Convertir a array
#     img_array = np.expand_dims(img_array, axis=0)  # Añadir una dimensión extra (para el batch)
#     img_array = img_array / 255.0  # Normalizar al rango [0, 1]
#     return img_array

# # Función para hacer la predicción y mostrar el resultado
# def classify_image(img_path):
#     img = load_and_preprocess_image(img_path)
#     prediction = model.predict(img)

#     predicted_label = np.argmax(prediction[0])
#     confidence = 100 * np.max(prediction[0])

#     # Mostrar la imagen preprocesada (ya redimensionada y normalizada)
#     plt.imshow(img[0])
#     plt.title(f"Predicción: {class_names[predicted_label]} ({confidence:.2f}%)")
#     plt.axis('off')
#     plt.show()

# # Solicitar la ruta de la imagen al usuario
# img_path = input("Ingresa la ruta de la imagen: ")
# classify_image(img_path)

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import numpy as np
import matplotlib.pyplot as plt

model = ResNet50(weights='imagenet')

def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224)) 
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = preprocess_input(img_array)
    return img_array

def classify_image(img_path):
    img = load_and_preprocess_image(img_path)
    predictions = model.predict(img)

    decoded_predictions = decode_predictions(predictions, top=1)[0]  # Solo la predicción principal

    imagenet_id, label, score = decoded_predictions[0]
    plt.imshow(image.load_img(img_path))
    plt.title(f"Predicción: {label} ({score:.2f}%)")
    plt.axis('off')
    plt.show()
  
img_path = input("Ingresa la ruta de la imagen: ")
classify_image(img_path)
