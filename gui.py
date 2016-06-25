import tkinter
import neural_network
import matpotlib

widthpixels = 400
heightpixels = 400

screen = tkinter.Tk()
screen.wm_title("traffic sign recognition")
screen.geometry('{}x{}'.format(widthpixels, heightpixels))
screen.resizable(width=False, height=False)

def callback():
    print(neural_network.train(int(entry.get())))

label = tkinter.Label(screen, text="Epochs")
label.place(relx=0.15, rely = 0.05, anchor = tkinter.CENTER)
entry = tkinter.Entry(screen, bd = 2)
entry.place(relx=0.45, rely = 0.05, anchor = tkinter.CENTER)
button = tkinter.Button(screen, text="Start", command=callback)
button.place(relx=0.75, rely = 0.05, anchor = tkinter.CENTER)

screen.mainloop()

