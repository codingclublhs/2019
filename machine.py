#Written by Aryan Bansal

#transfers code from python 2 to python 3 and vice versa
from __future__ import absolute_import, division, print_function, unicode_literals


import tensorflow as tf #calling tensorflow tf

#making numy able to be formatted into neat lists
import numpy as np
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

import matplotlib.pyplot as plt #used for graphing

#establishing arrays for celsius and farenheit
celsius_q    = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)

#printing data
for i,c in enumerate(celsius_q):
  print("{} degrees Celsius = {} degrees Fahrenheit".format(c, fahrenheit_a[i]))


l0 = tf.keras.layers.Dense(units=1, #units is the number of internal variables it has to solve for, only 1 since only fiding one value - Farenheit)
 input_shape=[1]) #number of types of inputs present, only 1 - Celsius

#only one layer l0, layers determine calculations
model = tf.keras.Sequential([l0])

''' (another way to write it)
model = tf.keras.Sequential([
  tf.keras.layers.Dense(units=1, input_shape=[1])
])
'''

model.compile(loss='mean_squared_error', #measuring how far off (average squares of errors)
              optimizer=tf.keras.optimizers.Adam(0.1)) #adjusting to reduce loss

history = model.fit(celsius_q, fahrenheit_a, epochs=10000, verbose=False) #epochs is the number of times program runs to get right value, verbose = how much output the method produces
print("Finished training the model")

#graphing epoch vs loss magnitude througout run time
plt.xlabel('Epoch Number') #x-value
plt.ylabel("Loss Magnitude") #y-value
plt.plot(history.history['loss'])


print(model.predict([100.0])) #prints output if input is 100.0

#prints algorithm
print("These are the layer variables: {}".format(l0.get_weights()))
