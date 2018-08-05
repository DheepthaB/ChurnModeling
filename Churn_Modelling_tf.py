#churn modeling using tensorflow

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf

#set up hyper parameters
learning_rate=0.5
epochs=100
batch_size=100

#import dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#encoding categorical variables
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

#scaling all tje variables
sc = StandardScaler()
X = sc.fit_transform(X)

#splitting into test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#features and labels placeholder
x=tf.placeholder(tf.float32, [None, 11])
y=tf.placeholder(tf.float32, [None, 1])

#weights and biases connection between input and first hidden layer
W1=tf.Variable(tf.random_normal([11, 6], stddev=0.03), name='w1')
b1=tf.Variable(tf.random_normal([6]), name='b1')

#weights and biases connection between  first hidden layer and second hidden layer
W2=tf.Variable(tf.random_normal([6, 6], stddev=0.03), name='w2')
b2=tf.Variable(tf.random_normal([6]), name='b2')

#weights and biases connection between second hidden layer and output layer
W3=tf.Variable(tf.random_normal([6, 1], stddev=0.03), name='w3')
b3=tf.Variable(tf.random_normal([1]), name='b3')

# calculate the output of the first hidden layer
h1_out=tf.nn.relu(tf.add(tf.matmul(x, W1),b1))

# calculate the output of the second hidden layer
h2_out=tf.nn.relu(tf.add(tf.matmul(h1_out, W2),b2))

# calculate the output of the output layer
y_pred=tf.nn.sigmoid(tf.add(tf.matmul(h2_out, W3),b3))

y_pred_clipped = tf.clip_by_value(y_pred, 1e-10, 0.9999999)

loss_func=-tf.reduce_mean(tf.reduce_sum(y*tf.log(y_pred_clipped)+ (1-y)*tf.log(1-y_pred_clipped), axis=1))

# add an optimiser
optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss_func)

# finally setup the initialisation operator
init_op=tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


X_train = np.array(X_train).reshape(8000, 11)
y_train = np.array(y_train).reshape(8000, 1)

X_test = np.array(X_test).reshape(2000, 11)
y_test = np.array(y_test).reshape(2000, 1)


# start the session
with tf.Session() as sess:
    sess.run(init_op)
    no_batches=int(len(X_train)/batch_size)
    for epoch in range(epochs):
        _, c = sess.run([optimiser, loss_func], 
                         feed_dict={x: X_train, y: y_train})
        print("Epoch:", (epoch + 1))
    print('accuracy', sess.run(accuracy, feed_dict={x: X_test, y: y_test}))

            






















