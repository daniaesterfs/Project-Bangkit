import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split

def load_data():
    data = pd.read_csv("breast.csv")
    x = data[['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se',
                'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'symmetry_worst', 'fractal_dimension_worst']]
    y = data[['diagnosis']] 

    return x, y

x_data, y_data = load_data()

# print(x_data)
# print(y_data)

y_data = OneHotEncoder(sparse = False).fit(y_data).transform(y_data)

x_data = MinMaxScaler().fit(x_data).transform(x_data)

# print(x_data)
# print(y_data)

layer = {
    'input' : 29,
    'hidden' : 29,
    'output' : 2
}

weight = {
    'input_to_hidden' : tf.Variable(tf.random.normal([layer ['input'], layer['hidden']])),
    'hidden_to_output' : tf.Variable(tf.random.normal([layer ['hidden'], layer ['output']]))
}

bias = {
    'hidden' : tf.Variable(tf.random.normal([layer ['hidden']])),
    'output' : tf.Variable(tf.random.normal([layer ['output']]))
}

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.3)

# Buat tempat

x_curr = tf.placeholder(tf.float32, [None, layer['input']])
y_target = tf.placeholder(tf.float32, [None, layer['output']])

def predict():
  w1 = tf.matmul(x_curr, weight['input_to_hidden'])
  w1 = w1 + bias['hidden']
  u = tf.nn.sigmoid(w1)

  w2 = tf.matmul(u, weight['hidden_to_output'])
  w2 = w2 + bias['output']
  y = tf.nn.sigmoid(w2)

  return y

y_predict = predict()

learning_rate = 0.1
num_epoch = 5000

# minimize error

error = tf.reduce_mean(0.5 * (y_target - y_predict) ** 2)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(error)

init = tf.global_variables_initializer()

# Training model
with tf.Session() as sess:
  sess.run(init)

  for i in range(1, num_epoch + 1):
    sess.run(train, feed_dict = {x_curr: x_train, y_target: y_train })
    error_val = sess.run(error, feed_dict = {x_curr : x_train, y_target : y_train})

    print('Epoch: {}, Error: {}%'.format(i, error_val * 100))

# Testing model

with tf.Session() as sess:
  sess.run(init)

  for i in range(num_epoch + 1):
    sess.run(train, feed_dict = {x_curr : x_train, y_target : y_train})

    if i % 500 == 0:
      matches = tf.equal(tf.argmax(y_target, axis = 1), tf.argmax(y_predict, axis = 1))
      accuracy = tf.reduce_mean(tf.cast(matches, tf.float32))

      print('{} = {}%'.format(i, sess.run(accuracy * 100, feed_dict = {x_curr : x_test, y_target : y_test})))
