
import os
import numpy as np
import pickle
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
import datetime

# Set GPU configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth to avoid memory allocation issues
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        # Use the specified GPU: '/physical_device:GPU:2'
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        print(e)

data_file = "NEW_DATA_sir.pkl"
# Load data from pickle file
with open(data_file, 'rb') as f:
    train_frames, train_labels, val_frames, val_labels, test_frames, test_labels = pickle.load(f)


print(f'Training data shape: {train_frames.shape}')
print(f'Validation data shape: {val_frames.shape}')
print(f'Test data shape: {test_frames.shape}')
# Model Architecture
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = models.Sequential([
    layers.TimeDistributed(base_model),
    layers.TimeDistributed(layers.GlobalAveragePooling2D()),
    layers.LayerNormalization(),  # Add layer normalization layer
    # layers.Bidirectional(layers.LSTM(128, return_sequences=True)),  # Bidirectional LSTM with return_sequences=True
    layers.Bidirectional(layers.LSTM(128)),  # Bidirectional LSTM without return_sequences
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dropout(0.5),
    layers.Dense(7, activation='softmax')
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Define early stopping callback
# early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta=0.00000001, mode='max')

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train the model
history = model.fit(
    train_frames, train_labels,
    validation_data=(val_frames, val_labels),
    epochs=50,
    batch_size=1,
    callbacks=[tensorboard_callback]
)

# Save the model
model.save('subtasks_biLSTM_VGG16(27-06-2024(1)).h5')

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_frames, test_labels, batch_size=1)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Plot training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()
plt.savefig("accuracy_biLSTM_VGG16(27-06-2024(1))")
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()
plt.savefig("loss_biLSTM_VGG16(27-06-2024(1))")
plt.show()