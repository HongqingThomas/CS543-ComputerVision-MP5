# import lxr as love

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import time
import cv2 as cv

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def load_image(Input):
    if Input == 1:
        Img1 = mpimg.imread('Part2_pictures/tsukuba1.jpg')
        Img2 = mpimg.imread('Part2_pictures/tsukuba2.jpg')
        gray1 = rgb2gray(Img1)
        gray2 = rgb2gray(Img2)
    elif Input == 2:
        Img1 = mpimg.imread('Part2_pictures/moebius1.png')#[160:620,900:,:]
        Img2 = mpimg.imread('Part2_pictures/moebius2.png')#[160:620,900:,:]
        gray1 = rgb2gray(Img1)
        gray2 = rgb2gray(Img2)
    return gray1, gray2

def disparity_map(Img1, Img2, w, d, method):
    row, col = Img1.shape[0], Img1.shape[1]
    disparity_value = np.zeros(shape = d)
    depth = np.zeros(shape=(row,col))
    for i in range(row-w+1): # 2~n-2 0-6
        for j in range(col-w+1):
            _Img1 = Img1[i:i+w,j:j+w]
            if method == 1: #SSD
                for _slide in range(int((-d + 1) / 2), int((d + 1) / 2)):
                    if j + _slide < 0 or j + _slide > col-w :
                        disparity_value[_slide-int((-d+1)/2)] = 10000000
                    else:
                        _Img2 = Img2[i:i+w,j+_slide:j+_slide+w]
                        disparity_value[_slide-int((-d+1)/2)] = np.sum((_Img1 - _Img2)**2)
                    dispa_ = abs(np.where(disparity_value == np.min(disparity_value))[0][0] + int((-d+1)/2))
                    print('dispa_:', dispa_)
            if method == 2: #SAD
                for _slide in range(int((-d + 1) / 2), int((d + 1) / 2)):
                    if j + _slide < 0 or j + _slide > col - w:
                        disparity_value[_slide - int((-d + 1) / 2)] = 10000000
                    else:
                        _Img2 = Img2[i:i + w, j + _slide:j + _slide + w]
                        disparity_value[_slide - int((-d + 1) / 2)] = np.sum(abs(_Img1 - _Img2))
                    dispa_ = abs(np.where(disparity_value == np.min(disparity_value))[0][0] + int((-d + 1) / 2))
            if method == 3: # normalized correlation
                for _slide in range(int((-d + 1) / 2), int((d + 1) / 2)):
                    if j + _slide < 0 or j + _slide > col - w:
                        disparity_value[_slide - int((-d + 1) / 2)] = 10
                    else:
                        _Img2 = Img2[i:i + w, j + _slide:j + _slide + w]
                        disparity_value[_slide - int((-d + 1) / 2)] = 1 - np.mean(np.multiply((_Img1 - np.mean(_Img1)), (_Img2 - np.mean(_Img2)))) / ((np.std(_Img1) * np.std(_Img2))+0.00001)
                    dispa_ = abs(np.where(disparity_value == np.min(disparity_value))[0][0] + int((-d + 1) / 2))
                    print('dispa_:', dispa_)

            if dispa_ != 0:
                depth[i][j] = 5/dispa_
    return depth

if __name__ == '__main__':

    start_time = time.time()

    # since all pictures are rectified in this project, there is only 2 parts need to be implemented:
    # 1. find the best match for each window in picture1
    # 2. compute disparity and depth
    Input = 2 # 1 rsukuba; 2 moebius
    gray1, gray2 = load_image(Input=Input)
    windowsize = 25 # 5, 11,15, 25, 35; 45； 55; 25
    disparity_range = 151 # odd 11, 21, 31, 41, 51; 21 31;111
    method = 1 # 1 SSD 2 SAD 3 normalized correlation
    depth = disparity_map(gray1, gray2, w=windowsize, d = disparity_range, method = method)

    end_time = time.time()
    total_time = end_time - start_time
    print('total time:', total_time, 's')

    # visualize
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(depth,cmap='gray', vmin=0, vmax=1)
    plt.show()

    # 49171.47095298767s