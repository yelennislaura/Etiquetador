__authors__ = ['1635979','1636581','1558589']
__group__ = 'DL.10 && DJ.12'

import numpy as np
import math
import operator
from scipy.spatial.distance import cdist


class KNN:
    def __init__(self, train_data, labels):
        self._init_train(train_data)
        self.labels = np.array(labels)
        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_train(self, train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        self.train_data = np.array(train_data, dtype=float).reshape(len(train_data), -1)

    def get_k_neighbours(self, test_data, k):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        test_data = np.asarray(test_data)
        test_data = np.reshape(test_data, (len(test_data), test_data[0].size)).astype(float)
        distances = cdist(test_data, self.train_data)
        sorted_indices = np.argsort(distances, axis=1)
        k_sorted_indices = sorted_indices[:, :k]
        self.neighbors = self.labels[k_sorted_indices]

    def get_class(self):
        """
        Get the class by maximum voting
        :return: 2 numpy array of Nx1 elements.
                1st array For each of the rows in self.neighbors gets the most voted value
                            (i.e. the class at which that row belongs)
                2nd array For each of the rows in self.neighbors gets the % of votes for the winning class
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        diccionari = dict()
        elements = []
        mes_votat = np.array([], dtype='<U8')
        percentatge = np.empty([len(self.neighbors), 1])

        # diccionari amb la quantitat de vegades que apareix cada element
        for i, element in enumerate(self.neighbors):
            for el in element:
                if el not in diccionari:
                    diccionari[el] = 0
                diccionari[el] += 1

            # afegim els elements que tenen el valor màxim a ELEMENTS
            valor_maxim = max(diccionari.values())
            for el in diccionari.keys():
                if diccionari[el] == valor_maxim:
                    elements.append(el)
            # ens quedem amb el primer de ELEMENTS i calculem el percentage d'aparicio del valor maxim dins el diccionari
            mes_votat = np.append(mes_votat, elements[0])
            total = sum(diccionari.values())
            percentatge[i] = valor_maxim / total

            # anem utilitzanr les mateixes varibale de manera que per no omplir-ho amb dades innecesaries ho netejem i ens quedem amb el q ens interessa
            diccionari.clear()
            elements.clear()

        return mes_votat

    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the output form get_class (2 Nx1 vector, 1st the class 2nd the  % of votes it got
        """

        self.get_k_neighbours(test_data, k)
        return self.get_class()
