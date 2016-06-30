import tkinter
from tkinter import Frame, X, LEFT, CENTER, BOTTOM, BOTH
import neural_network
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure

#Window settings
widthpixels = 500
heightpixels = 500

screen = tkinter.Tk()
screen.wm_title("Traffic Sign Recognition")
screen.geometry('{}x{}'.format(widthpixels, heightpixels))
screen.resizable(width=True, height=True)

#Frame 1 - set data
frame1 = Frame(screen)
frame1.pack(fill=X)

label = tkinter.Label(frame1, text="Epochs:")
#label.place(relx=0.15, rely = 0.05, anchor = CENTER)
label.pack(side=LEFT)
entry = tkinter.Entry(frame1, bd = 2)
#entry.place(relx=0.45, rely = 0.05, anchor = CENTER)
entry.pack(side=LEFT)
button = tkinter.Button(frame1, text="Start", command = lambda: callback(int(entry.get())))
#button.place(relx=0.75, rely = 0.05, anchor = CENTER)
button.pack(side=LEFT)

#Frame 2 - graph
frame2 = Frame(screen)
frame2.pack(fill=X)

x_axis = [0]
y_axis = [100]

fig = Figure(figsize=(5,5), dpi=100)
ax = fig.add_subplot(111)
ax.axis([0,30,0,100])
ax.set_xlabel('Num of epochs')
ax.set_ylabel('Failed')
line1, = ax.plot(x_axis, y_axis)

canvas = FigureCanvasTkAgg(fig, frame2)
toolbar = NavigationToolbar2TkAgg(canvas, frame2)
toolbar.pack(side=BOTTOM)
canvas.get_tk_widget().pack(side=BOTTOM, fill=BOTH, expand=True)
	
def callback(epochs):
	failed = neural_network.train(epochs)
	x_axis.append(epochs)
	y_axis.append(failed)
	line1.set_xdata(x_axis)
	line1.set_ydata(y_axis)
	fig.canvas.draw()

screen.mainloop()

