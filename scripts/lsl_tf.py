from __future__ import absolute_import, division, print_function

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


from pylsl import StreamInlet, resolve_stream, StreamOutlet, StreamInfo
import numpy as np
import progressbar
import math

print(tf.__version__)
sample_name = 'generator_sample'
label_name = 'Left_Hand_Command'

print("Searching for stream :", sample_name) 
streams_sample = resolve_stream('name', sample_name)
print("Searching for stream :", label_name)
streams_label = resolve_stream('name', label_name)

# create a new inlet to read from the stream
inlet_sample = StreamInlet(streams_sample[0])
inlet_label = StreamInlet(streams_label[0])

t=0


x_train = []
y_train = []
N= 100

bar = progressbar.ProgressBar(maxval=N, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

bar.start()
while t<N:
    sample, ts = inlet_sample.pull_sample()
    sample = np.array(sample,float)#.reshape((2,2))
    label, tl = inlet_label.pull_sample()
    
    if(ts and tl):
        bar.update(t)
        t += 1
        x_train.append(sample)
        y_train.append(label)

bar.finish()

x_train = np.array(x_train,float)
y_train = np.array(y_train,float)

def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation=tf.nn.relu, input_shape=[len(sample)]),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(len(label))
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    return model

model = build_model()

model.summary()

class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')


def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  #plt.ylim([0,5])
  plt.legend()
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  #plt.ylim([0,20])
  plt.legend()
  plt.show()



# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=200)

EPOCHS = 1000
history = model.fit(x_train, y_train, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

plot_history(history)
loss, mae, mse = model.evaluate(x_train, y_train, verbose=0)

print("Testing set Mean Abs Error: {:5.2f} ".format(mae))



"""
model.fit(x_train, y_train, epochs=5)

## Recreate the exact same model, including weights and optimizer.
#model = tf.keras.models.load_model('my_model.h5')
model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#model.evaluate(x_test, y_test)
"""

info_out = StreamInfo('Right_Hand_Command', 'result', 15, 0, 'float32')

# next make an outlet
outlet = StreamOutlet(info_out)

while True:
    sample, ts = inlet_sample.pull_sample()
    label, tl = inlet_label.pull_sample()
    
    if(ts and tl):
        x_train = np.append(x_train,[sample],axis=0)
        result = model.predict(x_train[-2:])[1]
        #print("R: " , result[:5])
        sum =0
        for i in range(len(result)):
            sum += (result[i]-label[i])**2
        sum = math.sqrt(sum)
        print(sum)
        #print("L: ",label[:5])
        outlet.push_sample(result)
        


