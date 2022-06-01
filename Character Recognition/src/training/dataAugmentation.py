import PIL.Image
#import tensorflow as tf
import keras
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import random
import skimage.transform
import numpy as np
import cv2



def rderodedilate(img):
    i = random.randint(-7,1)
    kernel = np.ones((2, 2), np.uint8)
    if i == 0:
        return img
    elif i > 0:
        img = cv2.erode(img, kernel, iterations=i)
    elif i < 0:
        img = cv2.dilate(img, kernel, iterations=-i)
    return img


def rdrotateimage(img):
    degrees = random.randint(-40,40)
    # rotates at degrees angle and pads with whitespace
    img = skimage.transform.rotate(img, angle=degrees, resize=False, cval=1)
    # transform back to a uint8-type image
    img = img * 255
    img = img.astype(np.uint8)
    return img


def addbox(img, xsize, ysize):
    x = random.randint(0, img.shape[0]-xsize)
    y = random.randint(0, img.shape[1]-ysize)
    for i in range(0,xsize):
        for j in range(0,ysize):
            img[x+i][y+j] = 0
    return img

def randomerase(img, prob):
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if random.random()<prob:
                img[i][j]=255
    return img


def randomcrop(img):
    cropleft = random.randint(1, 4)
    cropright = random.randint(1, 4)
    cropbottom = random.randint(1, 4)
    croptop = random.randint(1, 4)
    img = img[0+cropleft:img.shape[0]-cropright, 0+cropbottom:img.shape[1]-croptop]
    return img


def randomexpand(img):
    img = cv2.copyMakeBorder(img, random.randint(0,5), random.randint(0,5), random.randint(0,5), random.randint(0,5), cv2.BORDER_REPLICATE)
    return img

def randomaugmentimage(img):
    methods = ['rdrotateimage', 'randomexpand', 'randomcrop', 'rderodedilate', 'randomerase', 'addbox']
    probs = createaugmentprobabilities(0.5, 0.2, 0.8, 0.2, 0.1, 0.3)

    if random.random()<probs[1]:
        img = randomexpand(img)
    if random.random()<probs[5]:
        img = addbox(img, 5, 5)
    elif random.random()<probs[2]:
        img = randomcrop(img)
    if random.random()<probs[0]:
        img = rdrotateimage(img)
    if random.random()<probs[4]:
        img = rderodedilate(img)
    if random.random()<probs[3]:
        img = randomerase(img, 0.3)



    return img

def createaugmentprobabilities(protate, pexpand, pcrop, perodedilate, perase, paddbox):
    probs = [protate, pexpand, pcrop, perodedilate, perase, paddbox]
    return probs

