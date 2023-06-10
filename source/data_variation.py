from globals import *
import cv2 as cv
import numpy as np
import os
from data_division import clear_folder
import random

def add_rotation(image, deg):

    # Get the image dimensions
    height, width = image.shape[:2]

    # Calculate the rotation matrix
    rotation_matrix = cv.getRotationMatrix2D((width / 2, height / 2), deg, 1)

    noise = np.random.uniform(0, 255, image.shape).astype(np.uint8)

    rotated_image = cv.warpAffine(image, rotation_matrix, (width, height))
    noise = cv.subtract(noise, cv.warpAffine(np.ones(image.shape).astype(np.uint8) * 255, rotation_matrix, (width, height)))
    noise = np.clip(noise, 0, 255).astype(np.uint8)
    rotated_image = cv.add(rotated_image, noise)


    # cv.imshow('Original Image', image)
    # cv.imshow('Rotated Image', rotated_image)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    return rotated_image

def add_noise(image, amount):
    noise = np.random.uniform(0, 255, image.shape).astype(np.uint8)

    noisy_image = cv.addWeighted(image, (1 - amount), noise, amount, 0)
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    # cv.imshow('Original Image', image)
    # cv.imshow('Noisy Image', noisy_image)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    return noisy_image

def distribution(folder_path):
    hist = np.zeros(20, dtype=int)
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            label = float(filename.split('_')[0])
            hist[int(label) // 5] += 1
    max = hist.max()
    #print(hist / max)
    return hist / max

def main():
    distrib = distribution(path_training_set)
    max_angle = 15 # degrees
    noise = 0.3 # from 0 to 1
    pid = os.getpid()
    random.seed(pid)
    clear_folder(path_training_set_augmented)
    for filename in os.listdir(path_training_set):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            label = float(filename.split('_')[0])
            file_path = os.path.join(path_data_set, filename)
            image = cv.imread(file_path)
            for i in range(int(min(5, float(1 / distrib[int(label) // 5])))):
                augmented_image = add_rotation(image, max_angle * random.uniform(-1, 1))
                augmented_image = add_noise(augmented_image, noise * random.uniform(0, 1))
                if 0.5 < random.uniform(0,1):
                    augmented_image = cv.flip(augmented_image, 1)
                cv.imwrite(os.path.join(path_training_set_augmented, filename + '_' + str(i) + '.jpg'), augmented_image)
                # cv.imshow('Original Image', image)
                # cv.imshow('Noisy Image', augmented_image)
                # cv.waitKey(0)
                # cv.destroyAllWindows()


if __name__ == "__main__":
    #distribution(path_training_set_augmented)
    main()
