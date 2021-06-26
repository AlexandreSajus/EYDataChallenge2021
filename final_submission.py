from random import randint
import numpy as np
import cv2


def directionalDilate(image):
    """
    A morphology transformation designed to dilate the image only towards the area where the pixels have a higher value
    :param image: an array representing the image to transform
    :type image: Array
    :return: the transformed array
    :rtype: Array
    """
    initr = randint(0, 10)  # Initialize the position of the kernels randomly
    result = np.zeros(image.shape)
    # For each kernel
    for i in np.arange(0, image.shape[0] - 10 - initr, 11):
        for j in np.arange(0, image.shape[1] - 10 - initr, 11):
            # A is the vector pointing towards the area where the pixels have a higher value
            A = [0, 0]
            center_x = i + 5 + initr
            center_y = j + 5 + initr
            for x in np.arange(-5, 6, 1):
                for y in np.arange(-5, 6, 1):
                    A += image[center_x + x, center_y + y]*np.array([x, y])
            # A has now been calculated

            # Here we split the image into two
            # The half in the direction of A will have only ones, the other half only zeroes
            for x in np.arange(-5, 6, 1):
                for y in np.arange(-5, 6, 1):
                    if A[0]*x + A[1]*y > 0:
                        result[center_x + x, center_y + y] = 1
    return result


def predict_old(src, lower=240, dilationIt=50, erosionIt=44):
    """
    This prediction relies only on the dilate-erode process to predict the location of fires on the image
    :param src: an array representing the image to predict
    :type src: Array
    :param lower: a threshold to detect pixels of high intensity
    :type lower: Int
    :param dilationIt: the number of dilation iterations
    :type dilationIt: Int
    :param erosionIt: the number of erosion iterations
    :type erosionIt: Int
    :return: the predicted label
    :rtype: Array
    """
    vis2 = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    lowerBound = np.array([lower, lower, lower])
    upperBound = np.array([255, 255, 255])
    rgb_mask = cv2.inRange(vis2, lowerBound, upperBound)  # Threshold

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
    dilate = cv2.dilate(rgb_mask, kernel, iterations=dilationIt)  # Dilate

    # We do this quick manipulation to allow erosion on sides of the image where all the pixels have 1 as a value
    temp1 = np.zeros((dilate.shape[0] + 2, dilate.shape[1] + 2))
    temp1[1:-1, 1:-1] = dilate
    temp2 = cv2.erode(temp1, kernel, iterations=erosionIt)  # Erode
    erode = temp2[1:-1, 1:-1]

    erode = np.where(erode != 0, 1, 0)
    return erode


def predict_new(src, lower=240, directionalDilateIt=5, dilationIt=15, erosionIt=14, failureThres=120000):
    """
    This prediction relies on directional dilation and the dilate-erode process to predict the location of fires on the image
    :param src: a dict containing the image to predict
    :type src: Dict
    :param lower: a threshold to detect pixels of high intensity
    :type lower: Int
    :param dilationIt: the number of dilation iterations
    :type dilationIt: Int
    :param erosionIt: the number of erode iterations
    :type erosionIt: Int
    :param failureThres: a threshold to revert back to the old method
    :type failureThres: Int
    :return: the predicted label
    :rtype: Array
    """
    a = src['linescan'].values[0, :, :]
    vis2 = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
    lowerBound = np.array([lower, lower, lower])
    upperBound = np.array([255, 255, 255])
    rgb_mask = cv2.inRange(vis2, lowerBound, upperBound)  # Threshold

    # Directional Dilate
    filtered = directionalDilate(rgb_mask)
    filtered_list = [filtered]
    filtered_it = filtered

    for i in range(directionalDilateIt - 1):
        filtered_it = directionalDilate(filtered_it)
        filtered_list.append(filtered_it)

    kernelOpen = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
    dilate = cv2.dilate(filtered_it, kernelOpen,
                        iterations=dilationIt)  # Dilate
    erode = cv2.erode(dilate, kernelOpen, iterations=erosionIt)  # Erode

    erode = np.where(erode != 0, 1, 0)
    # If the amount of "true" pixels is too low, we assume that this method is less effective than the old one and we revert to the old one
    if np.sum(erode) < failureThres:
        return predict_old(src)
    return erode
