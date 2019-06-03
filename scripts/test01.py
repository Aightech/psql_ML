from __future__ import absolute_import, division, print_function


import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


from pylsl import StreamInlet, resolve_stream, StreamOutlet, StreamInfo
import numpy as np
import progressbar
import math

def update_line(l,x,y):
    print(y)
    l.set_xdata(np.append(l.get_xdata(), x))
    l.set_ydata(np.append(l.get_ydata(), y))
    plt.ion()
    plt.pause(1e-17)
    



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

info_out = StreamInfo('Right_Hand_Command', 'result', 15, 0, 'float32')

# next make an outlet
outlet = StreamOutlet(info_out)

t=0


x_train = [] #np.array([],float)
y_train = [] #np.array([],float)
N= 100

def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation=tf.nn.relu, input_shape=[streams_sample[0].channel_count()]),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(streams_label[0].channel_count())
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    return model

model = build_model()

model.summary()

bar = progressbar.ProgressBar(maxval=N, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

axes = plt.gca()
axes.set_ylim(0, 4)
line, = axes.plot([], [], 'r-')

#bar.start()
loop = True

class Index(object):
    def exit(self, event):
        loop = False

callback = Index()
a = plt.axes([0.7, 0.05, 0.1, 0.075])
btn = Button(a, 'Next')
btn.on_clicked(callback.exit)

while loop:
    sample, ts = inlet_sample.pull_sample()
    sample = np.array(sample,float)#.reshape((2,2))
    label, tl = inlet_label.pull_sample()    
    
    if(ts and tl):
        t += 1
        x_train.append(sample)
        x_t = np.array(x_train,float)
        y_train.append(label)
        y_t = np.array(y_train,float)
        if(len(x_train)>2):
            history = model.fit(x_t,
                                y_t,
                                epochs=1,
                                validation_split = 0.2,
                                verbose=0 )
            hist = pd.DataFrame(history.history)
            axes.set_xlim(0, t)
            if(axes.get_ylim()[1] < hist['mean_absolute_error'][0]):
                axes.set_ylim(0, hist['mean_absolute_error'])
            
            update_line(line,t, hist['mean_absolute_error'])
            if( hist['mean_absolute_error'][0] < 1):
                result = model.predict(x_t[-2:])[1]
                outlet.push_sample(result)





plt.show()


