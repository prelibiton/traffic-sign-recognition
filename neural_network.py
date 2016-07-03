import cv2
import numpy as np
import csv
from pybrain.datasets.supervised import SupervisedDataSet 
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer, RPropMinusTrainer
from pybrain.tools.customxml import NetworkWriter
from pybrain.tools.customxml import NetworkReader

class Brain:		
	def __init__(self):
		classes = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 17, 25])
		self.samples = []
		self.labels = []
		for i in range(len(classes)):
			prefix = "GTSRB/" + format(classes[i], '05d') + '/'
			file = open(prefix + 'GT-' + format(classes[i], '05d') + '.csv')
			reader = csv.reader(file, delimiter=';')
			next(reader, None)

			for row in reader:
				image = cv2.imread(prefix + row[0])
				self.samples.append(image)
				self.labels.append(i)

		self.samples = [cv2.resize(s, (10, 10)) for s in self.samples]
		self.samples = np.array(self.samples).astype(np.float32) / 255
		self.samples = [s.flatten() for s in self.samples]
		
		np.random.seed(0)
		np.random.shuffle(self.samples)
		np.random.seed(0)
		np.random.shuffle(self.labels)
	
		self.totalEpochs = 0

	def test_train(self, epochs=1):
		print("Training...")

		split = int(len(self.samples) * 0.7)
		train_samples = self.samples[0:split]
		train_labels  = self.labels[0:split]

		test_samples = self.samples[split:]
		test_labels  = self.labels[split:]

		net = buildNetwork(300, 300, 1)	
		ds = SupervisedDataSet(300, 1)
		for i in range(len(train_samples)):  
			ds.addSample(tuple(np.array(train_samples[i], dtype='float64')), (train_labels[i],))
		
		trainer = BackpropTrainer(net, ds, verbose=True)
		trainer.trainEpochs(epochs)
		self.totalEpochs = epochs
		
		error = 0
		counter = 0
		for i in range(0, 100):
			output = net.activate(tuple(np.array(test_samples[i], dtype='float64')))
			if round(output[0]) != test_labels[i]:
				counter += 1
				print(counter, " : output : ", output[0], " real answer : ", test_labels[i])
				error += 1
			else:
				counter += 1
				print(counter, " : output : ", output[0], " real answer : ", test_labels[i])
		
		print("Trained with " + str(epochs) + " epochs; Total: " + str(self.totalEpochs) + ";")
		return error
	
	def train_clean(self, epochs=1):
		print("Training...")
		self.totalEpochs = epochs
		
		train_samples = self.samples
		train_labels  = self.labels

		self.net_shared = buildNetwork(300, 300, 1)	
		self.ds_shared = SupervisedDataSet(300, 1)
		for i in range(len(train_samples)):  
			self.ds_shared.addSample(tuple(np.array(train_samples[i], dtype='float64')), (train_labels[i],))
		
		self.trainer_shared = BackpropTrainer(self.net_shared, self.ds_shared, verbose=True)
		self.trainer_shared.trainEpochs(epochs)
		
		print("Trained with " + str(epochs) + " epochs; Total: " + str(self.totalEpochs) + ";")

	def train_more(self, epochs=1):
		print("Training...")
		self.totalEpochs += epochs
		self.trainer_shared.trainEpochs(epochs)
		print("Trained with " + str(epochs) + " epochs more; Total: " + str(self.totalEpochs) + ";")
	
	def test_image(self, filename):		
		image = cv2.imread(filename)
		images = [image]
		
		images = [cv2.resize(s, (10, 10)) for s in images]
		images = np.array(images).astype(np.float32) / 255
		images = [s.flatten() for s in images]
		
		output = self.net_shared.activate(tuple(np.array(images[0], dtype='float64')))
		print("Output: ", output[0])
		return output[0]
			
	def import_network(self, filename):
		train_samples = self.samples
		train_labels  = self.labels
		
		np.random.seed(0)
		np.random.shuffle(train_samples)
		np.random.seed(0)
		np.random.shuffle(train_labels)
		
		self.net_shared = NetworkReader.readFrom(filename)
		self.ds_shared = SupervisedDataSet(300, 1)
		for i in range(len(train_samples)):  
			self.ds_shared.addSample(tuple(np.array(train_samples[i], dtype='float64')), (train_labels[i],))
			
		self.trainer_shared = BackpropTrainer(self.net_shared, self.ds_shared, verbose=True)
		
	def export_network(self, filename):
		NetworkWriter.writeToFile(self.net_shared, filename)



