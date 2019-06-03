import matplotlib.pyplot as plt
import numpy
import random
 
ysample = random.sample(range(-50, 50), 100)
 
xdata = []
ydata = []
 
plt.show()
 
axes = plt.gca()
axes.set_xlim(0, 100)
axes.set_ylim(-50, +50)
line, = plt.plot(xdata, ydata, 'r-')

def update_line(l,x,y):
    l.set_xdata(numpy.append(l.get_xdata(), x))
    l.set_ydata(numpy.append(l.get_ydata(), y))
    plt.draw()
    plt.pause(1e-17)         
 
for i in range(100):
    update_line(line, i, ysample[i])
    #time.sleep(0.1)
    
# add this if you don't want the window to disappear at the end
plt.show()
