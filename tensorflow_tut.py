# Commented out IPython magic to ensure Python compatibility.
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
# %matplotlib inline

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt',
'Sneaker', 'Bag', 'Ankle Boot']

print(class_names.index('Ankle Boot'))
train_images = train_images/255.0
test_images = test_images/255.0

model = keras.Sequential([keras.layers.Flatten(input_shape=(28,28)),
                          keras.layers.Dense(128, activation='relu'),
                          keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

"""You Don't always have to evaluate the model as the model is trained we can use it straight up to the predicting other output"""

# tes_loss, test_acc = model.evaluate(test_images, test_labels)

# print("Test Accuracy: ", test_acc)

prediction = model.predict(test_images)

for i in range(5):
  plt.grid(False)
  plt.imshow(test_images[i], cmap=plt.cm.binary)
  plt.xlabel("Actual: "+ class_names[test_labels[i]])
  plt.title("Prediction: "+ class_names[np.argmax(prediction[i])])
  plt.show()





