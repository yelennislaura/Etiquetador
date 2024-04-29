__authors__ = ['1635979','1636581','1558589']
__group__ = 'DL.10 && DJ.12'

import numpy as np
from utils_data import read_dataset, read_extended_dataset, crop_images, visualize_retrieval, visualize_k_means
import Kmeans as km
from Kmeans import *

def retrieval_by_color(images, tags, question):

    """
    Retrieve images that contain the specified color tags based on a question.

    Args:
        images (list): List of images.
        tags (list): List of tags obtained by applying the K-means algorithm to the images.
        question (str or list): Color(s) we want to search for.

    Returns:
        list: List of images that contain the specified color tags.
    """

    # Filter images based on color tags
    filtered_images = []
    for image, tag in zip(images, tags):
        if all(color in tag for color in question):
            filtered_images.append(image)

    visualize_retrieval(filtered_images, 8)

    return filtered_images

def retrieval_by_shape(images, labels, shapes):

    shape_counts = []
    shape_percentages = []

    for i in range(len(labels)):
        shape_count = dict(zip(*np.unique(labels[i], return_counts=True)))
        total_shapes = np.sum(list(shape_count.values()))

        shape_counts.append(shape_count)
        shape_percentage = {shape: count / total_shapes for shape, count in shape_count.items()}
        shape_percentages.append(shape_percentage)

    image_list = {}
    for index, image in enumerate(images):
        shape_sum = 0
        found = False
        m = 0
        while m < len(shapes) and not found:
            if shapes[m] not in shape_counts[index]:
                found = True
                if index in image_list:
                    image_list.pop(index)
            else:
                if index not in image_list:
                    image_list[index] = 0
                shape = shapes[m]
                shape_sum += shape_percentages[index].get(shape, 0)
                m += 1

        image_list[index] = shape_sum

    sorted_images = sorted(image_list.items(), key=lambda x: x[1], reverse=True)
    sorted_image_indices = [index for index, _ in sorted_images]
    return_list = [images[index] for index in sorted_image_indices]

    visualize_retrieval(return_list, 16)

    return return_list
"""
def retrieval_by_color(images, labels, query_colors):

    
    llista_return = []
    for ix, image in enumerate(images):
        for goal in query_colors:
            if goal in labels[ix]:
                llista_return.append(image)
                break
    etiquetes = []

    for test in test_imgs:
        Km = km.KMeans(test,K=2)  # dependiendo de la K dará unos resultados u otros, pero funciona (puedes poner la que quieras)
        Km.fit()
        etiquetes.append(km.get_colors(Km.centroids))

    return etiquetes
"""

def kmean_statistics(kmeans, kmax):
    # Results for 2 <= x < Kmax

    '''
    for i in range(2, kmax):
        start = time.time()
        kmeans.find_bestKImprovement(i, 50, 'Intra')
        end = time.time()
        res = end-start
        visualize_k_means(kmeans, [80,60,3], i, res)
    '''

    # Result for the 'Kmax' given

    kmeans.find_bestK(kmax)

    # Should work, but does not. This way we do not fix the image's size
    visualize_k_means(kmeans, [80, 60, 3])

if __name__ == '__main__':

    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset(root_folder='./images/', gt_json='./images/gt.json')

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # Load extended ground truth
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_images = crop_images(imgs, upper, lower)

    # You can start coding your functions here

    #Llamada K_means_statistics

    '''for i in range(20):
        element_kmeans = KMeans(test_imgs[i])
        kmean_statistics(element_kmeans, 7)'''

    #Llamada retrieval_by_color
    #retrieval_by_color(test_imgs,test_color_labels, ['Black'])

    #retrieval_by_shape(test_imgs, test_class_labels, ['Shorts'])


    """
    kMeans.fit()
    lista_cosas = get_colors(kMeans.centroids)
    retrieval_by_color(test_imgs, test_color_labels, ['Yellow'])
    """
    for i in range(20):
        element_kmeans = KMeans(train_imgs[i],4)
        element_kmeans.fit()
        element_kmeans.find_bestK_millora(4,'Fisher',10)
        mida_img=np.shape(train_imgs[i])
        visualize_k_means(element_kmeans,[mida_img[0],mida_img[1],mida_img[2]])

