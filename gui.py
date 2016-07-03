import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, Frame, X, LEFT, RIGHT, CENTER, BOTTOM, BOTH
import neural_network
from neural_network import Brain
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
		
		
class App(tk.Tk):

	def __init__(self, *args, **kwargs):
		
		tk.Tk.__init__(self, *args, **kwargs)
		
		#Window settings
		widthpixels = 500
		heightpixels = 500

		tk.Tk.wm_title(self, "Traffic Sign Recognition")
		tk.Tk.geometry(self, '{}x{}'.format(widthpixels, heightpixels))
		tk.Tk.resizable(self, width=True, height=True)

		container = tk.Frame(self)
		container.pack(side="top", fill="both", expand = True)
		container.grid_rowconfigure(0, weight=1)
		container.grid_columnconfigure(0, weight=1)

		self.frames = {}
		self.widths = {}
		self.heights = {}
		
		for F in (StartPage, PageOne, PageTwo):

			frame = F(container, self)

			self.frames[F] = frame
			self.widths[F] = frame.width
			self.heights[F] = frame.height

			frame.grid(row=0, column=0, sticky="nsew")

		self.show_frame(StartPage)

	def show_frame(self, page_name):

		for frame in self.frames.values():
			frame.grid_remove()
		frame = self.frames[page_name]
		tk.Tk.geometry(self, '{}x{}'.format(self.widths[page_name], self.heights[page_name]))
		frame.grid()
	
	def change_res(self, width, height):
		tk.Tk.geometry(self, '{}x{}'.format(width, height))

class StartPage(tk.Frame):

	def __init__(self, parent, controller):
		tk.Frame.__init__(self,parent)
		self.width = 300
		self.height = 100
		label = tk.Label(self, text="Start Page")
		label.pack(pady=10,padx=10)

		button = ttk.Button(self, text="Epochs graph",
							command=lambda: controller.show_frame(PageOne))
		button.pack()

		button2 = ttk.Button(self, text="Test image",
							command=lambda: controller.show_frame(PageTwo))
		button2.pack()

class PageOne(tk.Frame):

	def __init__(self, parent, controller):
		tk.Frame.__init__(self, parent)
		self.width = 500
		self.height = 500
		label = tk.Label(self, text="Graph Page!")
		label.pack(pady=10,padx=10)
		
		button1 = ttk.Button(self, text="Back to Home",
					command=lambda: controller.show_frame(StartPage))
		button1.pack()
		
		#Frame 1 - set data
		frame1 = Frame(self)
		frame1.pack(fill=X)

		frame1_center = Frame(frame1)
		frame1_center.pack()
		
		label = ttk.Label(frame1_center, text="Epochs:")
		label.pack(side=LEFT, pady=10,padx=10)
		entry = tk.Entry(frame1_center, bd = 2)
		entry.pack(side=LEFT, pady=10,padx=10)
		button = ttk.Button(frame1_center, text="Test", command = lambda: self.callback(int(entry.get())))
		button.pack(side=LEFT, pady=10,padx=10)
		
		#Frame 2 - graph
		frame2 = Frame(self)
		frame2.pack(fill=X)

		self.x_axis = [0]
		self.y_axis = [100]

		self.fig = Figure(figsize=(5,5), dpi=100)
		ax = self.fig.add_subplot(111)
		ax.axis([0,30,0,100])
		ax.set_xlabel('Num of epochs')
		ax.set_ylabel('Failed')
		self.line1, = ax.plot(self.x_axis, self.y_axis)

		canvas = FigureCanvasTkAgg(self.fig, frame2)
		toolbar = NavigationToolbar2TkAgg(canvas, frame2)
		toolbar.pack(side=BOTTOM)
		canvas.get_tk_widget().pack(side=BOTTOM, fill=BOTH, expand=True)
		
		#brain
		self.brain = Brain()
		
	def callback(self, epochs):
		failed = self.brain.test_train(epochs)
		self.x_axis.append(epochs)
		self.y_axis.append(failed)
		self.line1.set_xdata(self.x_axis)
		self.line1.set_ydata(self.y_axis)
		self.fig.canvas.draw()
		print("Num failed: " + str(failed))

class PageTwo(tk.Frame):

	def __init__(self, parent, controller):
		tk.Frame.__init__(self, parent)
		self.width = 500
		self.height = 500
		label = tk.Label(self, text="Test image!")
		label.pack(pady=10,padx=10)

		button1 = ttk.Button(self, text="Back to Home",
							command=lambda: controller.show_frame(StartPage))
		button1.pack()
		
		#Frame - Network actions
		frame_net = Frame(self)
		frame_net.pack(fill=X)
		
		label_epochs = ttk.Label(frame_net, text="Network: ")
		label_epochs.pack(side=LEFT, pady=10,padx=10)
		
		button = ttk.Button(frame_net, text="Import", command=self.ask_import_net)
		button.pack(side=LEFT)
		
		button = ttk.Button(frame_net, text="Export", command=self.ask_export_net)
		button.pack(side=LEFT)
		
		#Frame 1 - set data
		frame1 = Frame(self)
		frame1.pack(fill=X)
		
		label_epochs = ttk.Label(frame1, text="Epochs:")
		label_epochs.pack(side=LEFT, pady=10,padx=10)
		entry_epochs = tk.Entry(frame1, bd = 2)
		entry_epochs.pack(side=LEFT, pady=10,padx=10)
		button = ttk.Button(frame1, text="Clean Train", command = lambda: self.train_clean(int(entry_epochs.get())))
		button.pack(side=LEFT, pady=10,padx=10)

		button = ttk.Button(frame1, text="Train", command = lambda: self.train(int(entry_epochs.get())))
		button.pack(side=LEFT, pady=10,padx=10)
		
		#Frame 2 - set data
		frame2 = Frame(self)
		frame2.pack(fill=X)
		
		label_file = ttk.Label(frame2, text="None")
		
		button = ttk.Button(frame2, text="File", command=lambda: self.askopenfile(label_file))
		button.pack(side=LEFT)
		
		label_file.pack(side=LEFT, pady=10,padx=10)
		
		#Frame - Test button
		frame_test = Frame(self)
		frame_test.pack(fill=X)
		
		button = ttk.Button(frame_test, text="Test image", command=self.test_image)
		button.pack(side=LEFT)
		
		#Frame - Result label
		frame_rl = Frame(self)
		frame_rl.pack(fill=X)
		
		label_r = ttk.Label(frame_rl, text="Result:")
		label_r.pack(side=LEFT, pady=10,padx=10)
	
		#Frame - Result frame
		frame_result = Frame(self)
		frame_result.pack(fill=X)
		
		self.label_result = ttk.Label(frame_result, text="Sign: None, Output number: None")
		self.label_result.pack(side=LEFT, pady=10,padx=10)
		
		#Frame - Result frame
		frame_testall = Frame(self)
		frame_testall.pack(fill=X)
		
		button = ttk.Button(frame_testall, text="Test all", command=self.test_all)
		button.pack(side=LEFT)
		
		#Frame - Result frame
		frame_allresult = Frame(self)
		frame_allresult.pack(fill=X)
		
		self.label_allresult = ttk.Label(frame_allresult, text="Avarage error:")
		self.label_allresult.pack(side=LEFT, pady=10,padx=10)
		
		#brain
		self.image_file = ""
		self.network_file = ""
		self.brain = Brain()
		
	def askopenfile(self, in_label):
		self.image_file = filedialog.askopenfilename()
		
		if self.image_file:
			print("Opened file: " + self.image_file)
			in_label["text"] = self.image_file
			
	def ask_import_net(self):
		self.network_file = filedialog.askopenfilename(filetypes = [("XML files", "*.xml")])
			
		if self.network_file:
			self.brain.import_network(self.network_file)
			print("Imported network from file: " + self.network_file)
	
	def ask_export_net(self):
		self.network_file = filedialog.asksaveasfilename(filetypes = [("XML files", "*.xml")])
		
		if self.network_file:
			self.brain.export_network(self.network_file)
			print("Exported network to file: " + self.network_file)
	
	def train_clean(self, epochs):
		self.brain.train_clean(epochs)
		
	def train(self, epochs):
		self.brain.train_more(epochs)
		
	def test_image(self):
		sign_names = ["20", "30", "50", "60", "70", "80", "80 out", "100", "120",
						"S predimstvo", "Bez predimstvo", "STOP", "B2 - Zabraneno vlizaneto v dvete posoki",
						"В1 - zabranetno vlizaneto", "A32 - uchastuk v remont"]
								
		result = self.brain.test_image(self.image_file)
		sign_name = sign_names[abs(int(round(result)))]
		self.label_result["text"] = "Sign: " + sign_name + ", Output number: " + str(result)
		
	def test_all(self):
		avarage = self.brain.test_allsamples()
		self.label_allresult["text"] = "Avarage error:" + str(avarage)

app = App()
app.mainloop()

