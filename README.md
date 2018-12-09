# TensorFlow_Project
#### Read google fashion image classification data
```
import numpy as np
import tensorflow as tf
from tensorflow import keras
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
```
#### Converted pixel value to floating point number
```
train_images = train_images / 255.0
test_images = test_images / 255.0
```
#### Building a dataset import function using a NumPy array
```
train_input_fn = tf.estimator.inputs.numpy_input_fn(
   x={"pixels": train_images}, y=train_labels.astype(np.int32), shuffle=True)
test_input_fn = tf.estimator.inputs.numpy_input_fn(
   x={"pixels": test_images}, y=test_labels.astype(np.int32), shuffle=False)
```
#### Define feature columns (numeric_column is numeric)
```
feature_columns = [tf.feature_column.numeric_column("pixels", shape=[28, 28])]
```
#### Define a deep learning neural network classifier, create a new folder estimator_test to save checkpoints
```
classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns, hidden_units=[128, 128], 
    optimizer=tf.train.AdamOptimizer(1e-4), n_classes=10, model_dir = './estimator_test')
classifier.train(input_fn=train_input_fn, steps=20000) # learning
model_eval = classifier.evaluate(input_fn=test_input_fn) # Evaluation
```
#### INFO:tensorflow:Saving dict for global step 469: accuracy = 0.8015, average_loss = 0.5861637, global_step = 469, loss = 74.19793

#### Read google fashion image classification data
 ```
 import tensorflow as tf
from tensorflow import keras
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
```
#### Converted pixel value to floating point number
```
train_images = train_images / 255.0
test_images = test_images / 255.0
```
#### Building the input layer - hidden layer - output layer
```
model = keras.Sequential([
   keras.layers.Flatten(input_shape=(28, 28)),
   keras.layers.Dense(128, activation=tf.nn.relu),
   keras.layers.Dense(10, activation=tf.nn.softmax)
])
```
#### Set optimization algorithm, loss function
```
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])
```
#### Start learning（epochs=5）
```
model.fit(train_images, train_labels, epochs=5)
```
#### Model evaluation
```
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
```
#### prediction
```
predictions = model.predict(test_images)
```
#### Save mode and mode parameters
```
model.save_weights('./keras_test') # Create a new folder in the current path
model.save('my_model.h5')
```
#### Epoch 1/5
#### 60000/60000 [==============================] - 8s 126us/step - loss: 0.4971 - acc: 0.8252
#### Epoch 2/5
#### 60000/60000 [==============================] - 6s 95us/step - loss: 0.3747 - acc: 0.8653
#### Epoch 3/5
#### 60000/60000 [==============================] - 6s 97us/step - loss: 0.3396 - acc: 0.8763
#### Epoch 4/5
#### 60000/60000 [==============================] - 6s 102us/step - loss: 0.3119 - acc: 0.8862
#### Epoch 5/5
#### 60000/60000 [==============================] - 6s 94us/step - loss: 0.2971 - acc: 0.8907
#### 10000/10000 [==============================] - 0s 44us/step
#### Test accuracy: 0.8777
#### The lower the loss, the better a model (unless the model has over-fitted to the training data).
