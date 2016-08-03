
import tensorflow as tf

import numpy as np
from numpy import random

import math
from enum import Enum

from nn import Trainer
from ga import *

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
        self.C = None
        self.S = None
        self.OSd = None
        self.BSd = None
        self.ConMax = None
        self.ConMean = None
        self.GMax = None
        self.GMean = None
        self.GSd = None

    def vectorize(self) :
        return np.array([self.A, self.P, self.C, self.S, self.OSd, self.BSd,
                         self.ConMax, self.ConMean, self.GMax, self.GMean, self.GSd])

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

    A = { 'MIN': 0.4, 'MAX': 40.6, 'MEAN': 6.4, 'STD': 7.6 }
    P = { 'MIN': 4.2, 'MAX': 117.6, 'MEAN': 28.1, 'STD': 22.1 }
    C = { 'MIN': 1.1, 'MAX': 6.8, 'MEAN': 3.2, 'STD': 1.1 }
    S = { 'MIN': 0.1, 'MAX': 40.8, 'MEAN': 4.2, 'STD': 7.4 }
    OSD = { 'MIN': 0.8, 'MAX': 3.8, 'MEAN': 1.7, 'STD': 0.6 }
    BSD = { 'MIN': 0.8, 'MAX': 3.0, 'MEAN': 1.1, 'STD': 0.4 }
    CMAX = { 'MIN': 3.2, 'MAX': 15.7, 'MEAN': 9.2, 'STD': 2.6 }
    CMEAN = { 'MIN': -0.4, 'MAX': 10.9, 'MEAN': 4.8, 'STD': 1.9 }
    GMAX = { 'MIN': 2.8, 'MAX': 15.5, 'MEAN': 7.2, 'STD': 2.2 }
    GMEAN = { 'MIN': 0.0, 'MAX': 6.5, 'MEAN': 3.0, 'STD': 1.1 }
    GSD = { 'MIN': 0.8, 'MAX': 2.7, 'MEAN': 1.4, 'STD': 0.4 }

    def __init__(self) :
        OceanObject.__init__(self, ObjectType.spill)

        self.A = self.random_feature(OilSpill.A)
        while self.A < 0:
            self.A = self.random_feature(OilSpill.A)

        self.radius = math.sqrt(self.A / math.pi)

        # self.P = 2.0 * math.pi * self.radius
        self.P = self.random_feature(Lookalike.A)
        while self.P <= 0:
            self.P = self.random_feature(Lookalike.A)

        self.C = self.P / (2.0 * math.sqrt(math.pi * self.A))
        self.S = self.random_feature(OilSpill.S)

        self.OSd = self.random_feature(OilSpill.OSD)
        self.BSd = self.random_feature(OilSpill.BSD)
        self.ConMax = self.random_feature(OilSpill.CMAX)
        self.ConMean = self.random_feature(OilSpill.CMEAN)

        self.GMean = self.random_feature(OilSpill.GMEAN)
        self.GMax = self.random_feature(self.GMAX)
        while self.GMax > self.GMean:
            self.GMax = self.random_feature(OilSpill.GMAX)
        self.GSd = self.random_feature(OilSpill.GSD)

class Lookalike(OceanObject):
    """
    """

    A = { 'MIN': 1.1, 'MAX': 115.6, 'MEAN': 13.3, 'STD': 17.0 }
    P = { 'MIN': 7.1, 'MAX': 396.4, 'MEAN': 52.4, 'STD': 57.2 }
    C = { 'MIN': 1.1, 'MAX': 10.4, 'MEAN': 3.9, 'STD': 1.7 }
    S = { 'MIN': 0.1, 'MAX': 45.2, 'MEAN': 11.8, 'STD': 11.4 }
    OSD = { 'MIN': 0.9, 'MAX': 3.2, 'MEAN': 2.0, 'STD': 0.6 }
    BSD = { 'MIN': 0.9, 'MAX': 2.3, 'MEAN': 1.5, 'STD': 0.4 }
    CMAX = { 'MIN': 2.6, 'MAX': 14.9, 'MEAN': 10.8, 'STD': 2.2 }
    CMEAN = { 'MIN': -0.4, 'MAX': 9.3, 'MEAN': 5.3, 'STD': 1.7 }
    GMAX = { 'MIN': 3.6, 'MAX': 16.8, 'MEAN': 8.5, 'STD': 2.6 }
    GMEAN = { 'MIN': 0.0, 'MAX': 5.2, 'MEAN': 2.7, 'STD': 1.0 }
    GSD = { 'MIN': 0.6, 'MAX': 2.6, 'MEAN': 1.5, 'STD': 0.5 }

    def __init__(self) :
        OceanObject.__init__(self, ObjectType.lookalike)

        self.A = self.random_feature(Lookalike.A)
        while self.A <= 0:
            self.A = self.random_feature(Lookalike.A)

        self.radius = math.sqrt(self.A / math.pi)

        # self.P = 2.0 * math.pi * self.radius
        self.P = self.random_feature(Lookalike.A)
        while self.P <= 0:
            self.P = self.random_feature(Lookalike.A)

        self.C = self.P / (2.0 * math.sqrt(math.pi * self.A))
        self.S = self.random_feature(Lookalike.S)

        self.OSd = self.random_feature(Lookalike.OSD)
        self.BSd = self.random_feature(Lookalike.BSD)
        self.ConMax = self.random_feature(Lookalike.CMAX)
        self.ConMean = self.random_feature(Lookalike.CMEAN)

        self.GMean = self.random_feature(Lookalike.GMEAN)
        self.GMax = self.random_feature(self.GMAX)
        while self.GMax > self.GMean:
            self.GMax = self.random_feature(Lookalike.GMAX)
        self.GSd = self.random_feature(Lookalike.GSD)

if __name__ == "__main__":

    r.seed(123456)

    training = [OilSpill() for i in range(200)]
    training.extend([Lookalike() for i in range(400)])
    r.shuffle(training)

    train_matrix_x = np.array([sample.vectorize() for sample in training]).reshape((-1, Trainer.n_features))
    train_matrix_y = np.array([sample.one_hot_vector() for sample in training]).reshape((-1, Trainer.n_classes))

    testing = [OilSpill() for i in range(50)]
    testing.extend([Lookalike() for i in range(100)])
    r.shuffle(testing)

    test_matrix_x = np.array([sample.vectorize() for sample in testing]).reshape((-1, 11))
    test_matrix_y = np.array([sample.one_hot_vector() for sample in testing]).reshape((-1, 2))

    trainer = Trainer(train_matrix_x, train_matrix_y, test_matrix_x, test_matrix_y)

    # trainer.train()
    # print(trainer.percent_accuracy())
    ga = GA(trainer)
    ga.evolve()
    ga.graph()
