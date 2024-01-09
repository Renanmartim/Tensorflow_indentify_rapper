import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np


# Example dataset structure:
# - dataset/
#   - train/
#     - you/
#       - your_face_1.jpg
#       - your_face_2.jpg
#     - uncle/
#       - uncle_face_1.jpg
#       - uncle_face_2.jpg

train_data_dir = r"C:\Users\renan\OneDrive\Área de Trabalho\Face_Recognition\dataset\train"
img_size = (128, 128)
batch_size = 16

train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'  # Two classes: you and your uncle
)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=19)

# Load an image for prediction
img_path = r'C:\Users\renan\OneDrive\Área de Trabalho\7Xnua7BM_400x400.jpg'
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Normalize the pixel values

# Make predictions
predictions = model.predict(img_array)

# Interpret the predictions
threshold = 0.5
predicted_class = 'Central cee' if predictions[0, 0] > threshold else 'Others'

print(f'The model predicts that the image contains: {predicted_class}')


# Now you can use the trained model for face recognition.

