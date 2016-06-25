import cv2
import numpy as np
import csv
from pybrain.datasets.supervised import SupervisedDataSet 
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

def train(epochs=1):
    classes = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 17, 25])
    samples = []
    labels = []
    for i in range(len(classes)):
        prefix = "GTSRB/" + format(classes[i], '05d') + '/'
        file = open(prefix + 'GT-' + format(classes[i], '05d') + '.csv')
        reader = csv.reader(file, delimiter=';')
        next(reader, None)

        for row in reader:
            image = cv2.imread(prefix + row[0])
            samples.append(image)
            labels.append(i)

    samples = [cv2.resize(s, (10, 10)) for s in samples]
    samples = np.array(samples).astype(np.float32) / 255
    samples = [s.flatten() for s in samples]

    np.random.seed(0)
    np.random.shuffle(samples)
    np.random.seed(0)
    np.random.shuffle(labels)

    split = int(len(samples) * 0.7)
    train_samples = samples[0:split]
    train_labels  = labels[0:split]

    test_samples = samples[split:]
    test_labels  = labels[split:]

    net = buildNetwork(300, 300, 1)
    ds = SupervisedDataSet(300, 1)
    for i in range(len(train_samples)):  
        ds.addSample(tuple(np.array(train_samples[i], dtype='float64')), (train_labels[i],))

    #for inpt, target in ds:
    #    print(inpt, target)
    
    trainer = BackpropTrainer(net, ds)
    trainer.trainEpochs(epochs)

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
    return error



