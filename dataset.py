import cv2
import numpy as np

import csv
from matplotlib import cm
from matplotlib import pyplot as plt

def load_data():
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

    np.random.seed(0)
    np.random.shuffle(samples)
    np.random.seed(0)
    np.random.shuffle(labels)

    split = int(len(samples) * 0.7)
    train_samples = samples[0:split]
    train_labels  = labels[0:split]

    test_samples = samples[split:]
    test_labels  = labels[split:]

    return (train_samples, train_labels), (test_samples, test_labels)



