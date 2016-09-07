
import tensorflow as tf

import numpy as np
from numpy import random

import math
from enum import Enum

from nn import Trainer
from ga import *
from sda import *

import pickle

class ObjectType(Enum):
    spill = 0
    lookalike = 1

class OceanObject(object):
    """Represents an ocean dark spot, with basic features shared by oil spills
    and lookalikes.

    Features:

    type: spill, lookalike

    Geometric
        area (A): area of dark spot
        perimeter (P): perimeter of dark spot
        complexity (C): defined P / (2 * sqrt(pi * A))
        spread (S): defined 100 * lambda2 / (lambda1 + lambda2)

    Backscatter
        object standard deviation (OSd): std of spill candidate pixels in dB
        background standard deviation (BSd): std of background pixels in dB
        max contrast (ConMax): difference between background mean and the min
                               pixel value in the object
        mean contrast (ConMean): difference between background mean and object
                                 mean
        max gradient (GMax): max value of border gradient
        mean gradient (GMean): mean value of border gradient
        gradient standard deviation (GSd): std of border gradient values
    """

    def __init__(self, type=ObjectType.spill) :
        self.type = type

        self.A = None
        self.P = None
        self.P_A = None
        self.C = None
        self.SP1 = None
        self.SP2 = None
        self.OMe = None
        self.OSd = None
        self.Opm = None
        self.BMe = None
        self.BSd = None
        self.Bpm = None
        self.Opm_Bpm = None
        self.ConMax = None
        self.ConMean = None
        self.ConRaMe = None
        self.ConRaSd = None
        self.ConLa = None
        self.GMax = None
        self.GMean = None
        self.GSd = None
        self.NDm = None
        self.TSp = None
        self.TSh = None
        self.THm = None

    def vectorize(self) :
        return np.array([self.A, self.P, self.P_A, self.C, self.SP1, self.SP2,
                         self.OMe, self.OSd, self.Opm, self.BMe, self.BSd, self.Bpm,
                         self.Opm_Bpm, self.ConMax, self.ConMean, self.ConRaMe,
                         self.ConRaSd, self.ConLa, self.GMax, self.GMean, self.GSd,
                         self.NDm, self.TSp, self.TSh, self.THm])

    def one_hot_vector(self) :
        temp = [0, 0]
        temp[self.type] = 1
        return np.array(temp)

    @staticmethod
    def random_feature(feature_stats) :
        proposed = r.normal(feature_stats['MEAN'], feature_stats['STD'])
        return proposed

class OilSpill(OceanObject):
    """
    """

    A = { 'MIN': 382.00, 'MAX': 346550.00, 'MEAN': 30471.09, 'STD': 55930.64 }
    P = { 'MIN': 138.00, 'MAX': 25290.00, 'MEAN': 2812.06, 'STD': 3823.01 }
    P_A = { 'MIN': 0.02, 'MAX': 0.75, 'MEAN': 0.21, 'STD': 0.14 }
    C = { 'MIN': 1.51, 'MAX': 11.77, 'MEAN': 4.33, 'STD': 2.40 }
    SP1 = { 'MIN': 1.05, 'MAX': 22.99, 'MEAN': 5.72, 'STD': 4.80 }
    SP2 = { 'MIN': 0.15, 'MAX': 1.00, 'MEAN': 0.86, 'STD': 0.19 }
    OME = { 'MIN': 17.37, 'MAX': 130.33, 'MEAN': 43.57, 'STD': 18.07 }
    OSD = { 'MIN': 12.84, 'MAX': 43.35, 'MEAN': 27.92, 'STD': 7.19 }
    OPM = { 'MIN': 0.30, 'MAX': 1.31, 'MEAN': 0.72, 'STD': 0.29 }
    BME = { 'MIN': 78.94, 'MAX': 172.21, 'MEAN': 115.94, 'STD': 18.56 }
    BSD = { 'MIN': 26.83, 'MAX': 59.87, 'MEAN': 44.28, 'STD': 8.52 }
    BPM = { 'MIN': 0.25, 'MAX': 0.54, 'MEAN': 0.39, 'STD': 0.08 }
    OPM_BPM = { 'MIN': 1.04, 'MAX': 3.61, 'MEAN': 1.84, 'STD': 0.62 }
    CMEAN = { 'MIN': 23.36, 'MAX': 121.29, 'MEAN': 72.36, 'STD': 20.67 }
    CMAX = { 'MIN': 77.60, 'MAX': 157.21, 'MEAN': 113.08, 'STD': 18.02 }
    CRAME = { 'MIN': 0.15, 'MAX': 0.80, 'MEAN': 0.38, 'STD': 0.14 }
    CRASD = { 'MIN': 0.39, 'MAX': 0.91, 'MEAN': 0.63, 'STD': 0.10 }
    CLA = { 'MIN': 0.15, 'MAX': 0.76, 'MEAN': 0.39, 'STD': 0.13 }
    GMEAN = { 'MIN': 54.19, 'MAX': 149.36, 'MEAN': 79.85, 'STD': 14.38 }
    GMAX = { 'MIN': 141.00, 'MAX': 255.00, 'MEAN': 226.62, 'STD': 34.11 }
    GSD = { 'MIN': 21.10, 'MAX': 51.60, 'MEAN': 37.90, 'STD': 7.87 }
    NDM = { 'MIN': 10.56, 'MAX': 65.98, 'MEAN': 35.86, 'STD': 12.44 }
    TSP = { 'MIN': 14.79, 'MAX': 51.47, 'MEAN': 33.84, 'STD': 8.44 }
    TSH = { 'MIN': 0.17, 'MAX': 0.24, 'MEAN': 0.22, 'STD': 0.01 }
    THM = { 'MIN': 18.36, 'MAX': 130.41, 'MEAN': 44.52, 'STD': 17.69 }

    def __init__(self) :
        OceanObject.__init__(self, ObjectType.spill)

        self.A = self.random_feature(OilSpill.A)
        while self.A <= 0:
            self.A = self.random_feature(OilSpill.A)

        # self.radius = math.sqrt(self.A / math.pi)
        # self.P = 2.0 * math.pi * self.radius

        self.P = self.random_feature(OilSpill.P)
        while self.P <= 0:
            self.P = self.random_feature(OilSpill.P)
        self.P_A = self.P / self.A

        self.C = self.P / (2.0 * math.sqrt(math.pi * self.A))

        self.SP1 = self.random_feature(OilSpill.SP1)
        self.SP2 = self.random_feature(OilSpill.SP2)

        self.OMe = self.random_feature(OilSpill.OME)
        self.OSd = self.random_feature(OilSpill.OSD)
        self.Opm = self.random_feature(OilSpill.OPM)

        self.BMe = self.random_feature(OilSpill.BME)
        self.BSd = self.random_feature(OilSpill.BSD)
        self.Bpm = self.random_feature(OilSpill.BPM)

        self.Opm_Bpm = self.Opm / self.Bpm

        self.ConMax = self.random_feature(OilSpill.CMAX)
        self.ConMean = self.random_feature(OilSpill.CMEAN)
        while self.GMax < self.GMean:
            self.ConMax = self.random_feature(OilSpill.CMAX)

        self.ConRaMe = self.random_feature(OilSpill.CRAME)
        self.ConRaSd = self.random_feature(OilSpill.CRASD)
        self.ConLa = self.random_feature(OilSpill.CLA)

        self.GMean = self.random_feature(OilSpill.GMEAN)
        self.GMax = self.random_feature(self.GMAX)
        while self.GMax < self.GMean:
            self.GMax = self.random_feature(OilSpill.GMAX)
        self.GSd = self.random_feature(OilSpill.GSD)

        self.NDm = self.random_feature(OilSpill.NDM)
        self.TSp = self.random_feature(OilSpill.TSP)
        self.TSh = self.random_feature(OilSpill.TSH)
        self.THm = self.random_feature(OilSpill.THM)

class Lookalike(OceanObject):
    """
    """

    A = { 'MIN': 401.00, 'MAX': 907090.00, 'MEAN': 47242.99, 'STD': 116988.16 }
    P = { 'MIN': 192.00, 'MAX': 32326.00, 'MEAN': 4233.16, 'STD': 5038.28 }
    P_A = { 'MIN': 0.03, 'MAX': 0.95, 'MEAN': 0.25, 'STD': 0.19 }
    C = { 'MIN': 1.58, 'MAX': 16.28, 'MEAN': 6.02, 'STD': 2.56 }
    SP1 = { 'MIN': 1.02, 'MAX': 22.99, 'MEAN': 4.06, 'STD': 4.51 }
    SP2 = { 'MIN': 0.05, 'MAX': 1.00, 'MEAN': 0.72, 'STD': 0.26 }
    OME = { 'MIN': 17.44, 'MAX': 91.45, 'MEAN': 46.40, 'STD': 15.39 }
    OSD = { 'MIN': 15.90, 'MAX': 47.09, 'MEAN': 30.32, 'STD': 6.51 }
    OPM = { 'MIN': 0.27, 'MAX': 1.60, 'MEAN': 0.72, 'STD': 0.26 }
    BME = { 'MIN': 78.12, 'MAX': 168.74, 'MEAN': 113.97, 'STD': 17.52 }
    BSD = { 'MIN': 26.57, 'MAX': 57.90, 'MEAN': 46.80, 'STD': 7.81 }
    BPM = { 'MIN': 0.25, 'MAX': 0.54, 'MEAN': 0.42, 'STD': 0.07 }
    OPM_BPM = { 'MIN': 1.06, 'MAX': 3.08, 'MEAN': 1.73, 'STD': 0.54 }
    CMEAN = { 'MIN': 23.84, 'MAX': 116.05, 'MEAN': 67.58, 'STD': 21.27 }
    CMAX = { 'MIN': 71.54, 'MAX': 163.53, 'MEAN': 112.83, 'STD': 17.56 }
    CRAME = { 'MIN': 0.17, 'MAX': 0.75, 'MEAN': 0.41, 'STD': 0.14 }
    CRASD = { 'MIN': 0.47, 'MAX': 0.87, 'MEAN': 0.65, 'STD': 0.10 }
    CLA = { 'MIN': 0.18, 'MAX': 0.75, 'MEAN': 0.45, 'STD': 0.14 }
    GMEAN = { 'MIN': 42.83, 'MAX': 110.51, 'MEAN': 79.11, 'STD': 11.66 }
    GMAX = { 'MIN': 140.00, 'MAX': 255.00, 'MEAN': 234.92, 'STD': 18.72 }
    GSD = { 'MIN': 18.79, 'MAX': 51.98, 'MEAN': 38.24, 'STD': 7.50 }
    NDM = { 'MIN': 6.29, 'MAX': 63.74, 'MEAN': 31.84, 'STD': 13.36 }
    TSP = { 'MIN': 18.14, 'MAX': 58.06, 'MEAN': 36.37, 'STD': 7.58 }
    TSH = { 'MIN': 0.21, 'MAX': 0.29, 'MEAN': 0.23, 'STD': 0.01 }
    THM = { 'MIN': 19.22, 'MAX': 91.58, 'MEAN': 47.42, 'STD': 14.85 }

    def __init__(self) :
        OceanObject.__init__(self, ObjectType.lookalike)

        self.A = self.random_feature(Lookalike.A)
        while self.A <= 0:
            self.A = self.random_feature(Lookalike.A)

        # self.radius = math.sqrt(self.A / math.pi)
        # self.P = 2.0 * math.pi * self.radius

        self.P = self.random_feature(Lookalike.P)
        while self.P <= 0:
            self.P = self.random_feature(Lookalike.P)
        self.P_A = self.P / self.A

        self.C = self.P / (2.0 * math.sqrt(math.pi * self.A))

        self.SP1 = self.random_feature(Lookalike.SP1)
        self.SP2 = self.random_feature(Lookalike.SP2)

        self.OMe = self.random_feature(Lookalike.OME)
        self.OSd = self.random_feature(Lookalike.OSD)
        self.Opm = self.random_feature(Lookalike.OPM)

        self.BMe = self.random_feature(Lookalike.BME)
        self.BSd = self.random_feature(Lookalike.BSD)
        self.Bpm = self.random_feature(Lookalike.BPM)

        self.Opm_Bpm = self.Opm / self.Bpm

        self.ConMax = self.random_feature(Lookalike.CMAX)
        self.ConMean = self.random_feature(Lookalike.CMEAN)
        while self.GMax < self.GMean:
            self.ConMax = self.random_feature(Lookalike.CMAX)

        self.ConRaMe = self.random_feature(Lookalike.CRAME)
        self.ConRaSd = self.random_feature(Lookalike.CRASD)
        self.ConLa = self.random_feature(Lookalike.CLA)

        self.GMean = self.random_feature(Lookalike.GMEAN)
        self.GMax = self.random_feature(self.GMAX)
        while self.GMax < self.GMean:
            self.GMax = self.random_feature(Lookalike.GMAX)
        self.GSd = self.random_feature(Lookalike.GSD)

        self.NDm = self.random_feature(Lookalike.NDM)
        self.TSp = self.random_feature(Lookalike.TSP)
        self.TSh = self.random_feature(Lookalike.TSH)
        self.THm = self.random_feature(Lookalike.THM)

if __name__ == "__main__":

    use_cache = True
    cache = None

    newfile = "cache.pk"

    if use_cache is True:
        with open(newfile, 'rb') as fi:
            cache = pickle.load(fi)

    # r.seed(50)

    # for i in range(4000, 4200):
    r.seed(4026)
    # print(i)

    training = [OilSpill() for i in range(300)]
    training.extend([Lookalike() for i in range(300)])
    r.shuffle(training)

    train_matrix_x = np.array([sample.vectorize() for sample in training]).reshape((-1, Trainer.n_features))
    train_matrix_y = np.array([sample.one_hot_vector() for sample in training]).reshape((-1, Trainer.n_classes))
    print(train_matrix_x.shape, train_matrix_y.shape)

    testing = [OilSpill() for i in range(100)]
    testing.extend([Lookalike() for i in range(100)])
    r.shuffle(testing)

    test_matrix_x = np.array([sample.vectorize() for sample in testing]).reshape((-1, Trainer.n_features))
    test_matrix_y = np.array([sample.one_hot_vector() for sample in testing]).reshape((-1, Trainer.n_classes))

    trainer = Trainer(train_matrix_x, train_matrix_y, test_matrix_x, test_matrix_y)

    # trainer.train([1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0])
    # trainer.train()
    #
    # positive = []
    # print("ONEHOTVECTORS -------------")
    # for i in range(len(test_matrix_y)):
    #     if test_matrix_y[i][0] == 1:
    #         positive.append(i)
    #
    # # accuracy, correct = trainer.percent_accuracy([1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0])
    # accuracy, correct = trainer.percent_accuracy()
    #
    # print("Correct Prediction? -------------")
    # count = 0.0
    # for index in positive:
    #     if correct[index] == True:
    #         count += 1.0
    #
    # print(count, len(positive))
    # print(count / float(len(positive)))
    #
    # negative = []
    # count = 0.0
    # for i in range(len(test_matrix_y)):
    #     if test_matrix_y[i][1] == 1:
    #         negative.append(i)
    #
    # print("Correct Prediction? -------------")
    # count = 0.0
    # for index in negative:
    #     if correct[index] == True:
    #         count += 1.0
    #
    # print(count, len(negative))
    # print(count / float(len(negative)))
    #
    # print(accuracy)

    # sda = SdA(dims=[12, 11], activations=['relu', 'relu'], epoch=[1000, 500],
    #           loss='cross-entropy', lr=0.007, batch_size=50, print_step=200)
    #
    # sda.fit(train_matrix_x)

    ga = GA(trainer)
    generation, output = 0, None
    if use_cache:
        ga.history = cache[2]
        ga.present_features = cache[3]
        generation, output = ga.evolve(cache[1], cache[0])
    else:
        generation, output = ga.evolve()

    dump = [generation, output, ga.history, ga.present_features]
    with open(newfile, 'wb') as fi:
        pickle.dump(dump, fi)

    ga.graph()
