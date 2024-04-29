__authors__ = ['1635979','1636581','1558589']
__group__ = 'DL.10 && DJ.12'

import numpy as np
import utils


class KMeans:

    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictionary with options
            """
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)  # DICT options

    #############################################################
    ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
    #############################################################

    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the sample space is the length of
                    the last dimension
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        if X.ndim == 3:
            F, C, D = X.shape
            N = F * C
            self.X = X.reshape((N, D))
        else:
            self.X = X
    pass

    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options is None:
            options = {}
        if 'km_init' not in options:
            options['km_init'] = 'first'
        if 'verbose' not in options:
            options['verbose'] = False
        if 'tolerance' not in options:
            options['tolerance'] = 0
        if 'max_iter' not in options:
            options['max_iter'] = np.inf
        if 'fitting' not in options:
            options['fitting'] = 'WCD'  # within class distance.

        # If your methods need any other parameter you can add it to the options dictionary
        self.options = options

        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################


    def _init_centroids(self):
        """
        Initialization of centroids
        """

        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        # inicializamos el contador que nos ayudará a determinar el final del array de numpy
        contador = 0
        # inicializamos los diferentes atributos de la clase
        self.old_centroids = np.empty([self.K, 3], dtype=float)
        self.old_centroids[:] = np.nan
        self.centroids = np.empty([self.K, 3], dtype=float)
        self.centroids[:] = np.nan

        # permutaciones aleatorias
        randPermutations = np.random.permutation(self.X)

        if self.options['km_init'] == 'first':
            for p in self.X:  # iteramos por los pixeles de forma ordenada
                if not any(np.equal(p, self.centroids).all(1)):  # miramos que no se repita la p para poder agregarla al numpy array de centroides
                    self.centroids[contador] = p  # guardamos el pixel que hará de centroide no repetido en la array numpy
                    contador += 1  # incrementamos el contador hasta llegar al maxio de centroides delimitado por self.k
                    if contador == self.K:  # cuando tengamos todos los centroides entonces salimos del bucle for
                        break

        elif self.options['km_init'] == 'random':
            for p in randPermutations:  # iteramos por los pixeles de forma aleatoria
                if not any(np.equal(p, self.centroids).all(1)):  # miramos que no se repita la p para poder agregarla al numpy array de centroides
                    self.centroids[contador] = p  # guardamos el pixel que hará de centroide no repetido en la array numpy
                    contador += 1  # incrementamos el contador hasta llegar al maxio de centroides delimitado por self.k
                    if contador == self.K:  # cuando tengamos todos los centroides entonces salimos del bucle for
                        break

        elif self.options['km_init'].lower() == 'custom':  # usmamos la funcion next + el rand como criterio
            for p in randPermutations:  # iteramos por los pixeles de forma aleatoria
                if not any(np.equal(next(p), self.centroids).all(1)):  # miramos que no se repita la p para poder agregarla al numpy array de centroides
                    self.centroids[contador] = next(p)  # guardamos el pixel que hará de centroide no repetido en la array numpy
                    contador += 1  # incrementamos el contador hasta llegar al maxio de centroides delimitado por self.k
                    if contador == self.K:  # cuando tengamos todos los centroides entonces salimos del bucle for
                        break


    def get_labels(self):
        """        Calculates the closest centroid of all points in X
        and assigns each point to the closest centroid
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        matriu = distance(self.X, self.centroids)
        self.labels = np.argmin(matriu, axis=1)


    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        centroid_sums = []
        self.old_centroids = np.array(self.centroids)
        var = np.bincount(self.labels)
        var = var.reshape(-1, 1)
        for index, n in enumerate(var):
            if n[0] == 0:
                var[index] += 1

        for index in range(len(self.centroids)):
            indices = np.where(self.labels == index)[0]
            centroid_sum = np.sum(self.X[indices], axis=0)
            centroid_sums.append(centroid_sum)
        self.centroids = centroid_sums / var
        pass


    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        return np.array_equal(self.old_centroids, self.centroids)


    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number
        of iterations is smaller than the maximum number of iterations.
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        self._init_centroids()  # inicializa la estructura
        while self.num_iter != self.options['max_iter'] and self.converges() is False:
            self.get_labels()  # primero calcular el "label"
            self.get_centroids()  # después obtener los nuevos centroides
            self.num_iter += 1  # sumar el número de iteraciones para ver si llega al máximo
        pass


    def withinClassDistance(self):
        """
         returns the within class distance of the current clustering
        """

        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        distancia = 0
        for counter in range(len(self.centroids)):
            puntsCluster = np.where(self.labels == counter)[0]  # usamos where porque devuelve los indices que cumplen la condición
            distancia = distancia + np.sum((self.X[puntsCluster] - self.centroids[counter]) ** 2)  # formula para calcular la distancia entre los puntos del clúster y el centroide
        classDist = distancia / len(self.X)
        return classDist



    def interClassDistance(self):

        d = 0
        for i in range(0, len(self.centroids)):
            p = np.where(self.labels == i)[0] #obtenemos los indices de los pixeles que se encuentran en el centroide actual (i)
            aux = self.centroids[np.where(np.array(range(0, len(self.centroids))) != i)] #creamos una variable auxiliar que nos guardara todos nuestros centroides excepto el actaul
            for centroide in aux: #calcularemos la distancia de los puntos de nuestro centroide actual a los punt de un nuevo centroide de aux para asi poder hacer un calculo promedio de la distancia entre clases.
                d = d + np.sum((self.X[p[:]] - centroide) ** 2)
        icd = d / len(self.X)
        return icd

    def fisherDiscriminant(self):
        """Discriminant: (distancia intra class) / (distancia inter class) """

        fd = self.withinClassDistance() / self.interClassDistance()
        return fd


    def find_bestK(self, max_K):
        """
         sets the best k anlysing the results up to 'max_K' clusters
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        self.K = 2  # iniciamos en numero de clusters a 2
        self.fit()
        aux = self.withinClassDistance()  # calculamos distancia intraclase
        self.K += 1  # incrementamos k para probar valores mas altos
        flag = False
        while self.K <= max_K and not flag:
            self.fit()
            w = self.withinClassDistance()  # calcula distancia intraclase actual
            percent = (w / aux) * 100  # calculamos el porcentaje de cambio que hay entre la distancia intraclase
            # actual y la distancia intraclase anterior
            if abs(100 - percent) < 20:  # si el porcentaje es menor al 20%, establecemos el valor anterior de k
                self.K -= 1
                flag = True
            else:  # sino, establecemos el valor actual de k en el valor siguiente y actualizamos el valor
                # de distancia intraclase anterior
                self.K += 1
                aux = w
        if not flag:  # si el flag no es cierto, asignamos el valor de k como valor maximo de k
            self.K = max_K
        self.fit()

    def find_bestK_millora(self, max_K, tipus, llindar):
        # tipus : nos dira la metodologia
        # llindar : valor que usaremos para controlar nuestro umbral
        self.K = 2  # iniciamos en numero de clusters a 2
        self.fit()

        if tipus == 'Fisher':
            dist = self.fisherDiscriminant()
        elif tipus == 'Inter':
            dist = self.interClassDistance()
        else:
            dist = self.withinClassDistance()

        self.K += 1  # incrementamos k para probar valores mas altos
        flag = False
        while self.K <= max_K and not flag:
            self.fit()

            if tipus == 'Inter':
                w = self.interClassDistance()
                porcentaje = (dist / w) * 100    #interclass promedio > interclass actual --> disminución de distancia
                                                #interclass actual > interclass promedio --> la distancia ha aumentado
            elif tipus == 'Fisher':
                w = self.fisherDiscriminant()
                porcentaje = (w / dist) * 100
            else:
                w = self.withinClassDistance()
                porcentaje = (w / dist) * 100

            if (100 - porcentaje) < llindar:
                self.K -= 1
                flag = True
            else:
                self.K += 1
                dist = w
            if flag is False:
                self.K = max_K

            self.fit()


def distance(X, C):
    """
    Calculates the distance between each pixel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
    """

    #########################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #########################################################
    dCalculada = np.zeros((X.shape[0], C.shape[0]))
    for index in range(C.shape[0]):
        d = np.sqrt(np.sum((X - C[index]) ** 2, axis=1))  # axis fa que sumi x columnes
        dCalculada[:, index] = d  # seleccionem totes les files e introduim d a la columna que indica l'index
    return dCalculada


def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color label following the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroid points)

    Returns:
        labels: list of K labels corresponding to one of the 11 basic colors
    """

    #########################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #########################################################
    colors = []
    for c in range(len(centroids)):
        color_prob = utils.get_color_prob(centroids)[c]
        max_index = np.argmax(color_prob)
        color = utils.colors[max_index]
        colors.append(color)
    return colors
