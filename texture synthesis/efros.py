#!/usr/bin/env python2
#-*-coding:utf-8-*-
'''
Efros Algorithm for CS534 HW1
'''
import random
import time
import numpy as np
from skimage import io, morphology, color
import os
from Gauss import gaussian
import matplotlib.pyplot as plt

class Efros:
    output_path = os.path.abspath(os.path.dirname(__file__))+'/output/'
    err_thres = 0.1
    MaxErrThreshold = 0.3

    window_size = 5
    margin = 2
    sample = None
    visited = None
    def __init__(self, path, file_name, window_size):
        self.name = file_name
        self.sample = io.imread(path+file_name).astype('float64')
        self.sample *= (1.0/self.sample.max())

        self.window_size = window_size
        self.margin = self.window_size / 2


    def find_matches(self, template, image, GaussMask, SampleImg_list_of_block, list_of_coordinate, err_thres = -1.0):
        ValidMask = self.visited[template[0]:template[1], template[2]: template[3]]
        blocks = image[template[0]:template[1], template[2]: template[3]]

        #if GaussMask.shape != ValidMask.shape:
            #print "[ERROR] GaussMask shape " + str(GaussMask.shape) + " is not equal to the ValidMask shape " + str(ValidMask.shape) +""
        mat_for_weight = np.multiply(GaussMask, ValidMask)
        total_weight = mat_for_weight.sum()
        blocks_list = np.tile(blocks, (len(SampleImg_list_of_block),1,1))
        SSD = np.sum(np.sum(np.multiply(mat_for_weight, np.square(blocks_list - SampleImg_list_of_block)), axis=1), axis=1) / total_weight
        threshold = SSD.min()
        if err_thres != -1.0:
            threshold= threshold*(1+err_thres)
            pixel_list = []
            for error,coordinate in zip(SSD, list_of_coordinate):
                if error <= threshold:
                    pixel_list.append((coordinate, error))
            return pixel_list
        else:
            for error,coordinate in zip(SSD, list_of_coordinate):
                if error == threshold:
                    return (coordinate, error)

    def neighbor_pixel(self, image, pos):
        return [pos[0] - self.margin, pos[0]+1 + self.margin, pos[1]-self.margin, pos[1]+1+self.margin]

    def find_neighbor(self):
        mat_for_dilate = np.ones((3,3))
        mast_for_dilate = morphology.dilation(self.visited, selem=mat_for_dilate) - self.visited
        count_mask = np.ones((3,3))
        count_mask[1][1] = 0
        count_dict = {}
        x,y = mast_for_dilate.shape
        for (i,j),k in np.ndenumerate(mast_for_dilate):
            if k == 1 and i >= self.margin and i < x - self.margin and j >= self.margin and j < y-self.margin:
                temp = self.visited[i-1:i+2 , j-1:j+2]
                count = int(np.multiply(temp, count_mask).sum())
                if count_dict.has_key(count):
                    count_dict[count].append((i,j))
                else:
                    count_dict.setdefault(count, [(i,j)])
        list_for_count = count_dict.keys()
        list_for_count.sort(reverse=True)
        remaining_list = []
        for key in list_for_count:
            remaining_list += count_dict[key]
        return remaining_list

    def image_filling(self, image, SampleImg_list_of_block, list_of_coordinate):

        sigma = self.window_size / 6.4
        GaussMask = gaussian((self.window_size, self.window_size), sigma)
        while 1:
            flag = 0
            pixel_list = self.find_neighbor()
            if len(pixel_list) == 0:
                break
            for pixel in pixel_list:
                template = self.neighbor_pixel(image, pixel)
                matches_list = self.find_matches(template, image, GaussMask, np.asarray(SampleImg_list_of_block), list_of_coordinate, self.err_thres)
                if matches_list == 1:
                    match_pixel = matches_list[0]
                else:
                    match_pixel = matches_list[random.randrange(len(matches_list))]
                if match_pixel[1] < self.MaxErrThreshold:
                    image[pixel[0],pixel[1]] = self.sample[match_pixel[0]]
                    self.visited[pixel[0],pixel[1]] = 1
                    flag = 1
            if flag == 0:
                self.MaxErrThreshold *= 1.1
            #io.imshow(image)
            #io.show()

    def efros_synthesis(self, img_x, img_y):
        height, width = self.sample.shape
        img_x += self.margin*2
        img_y += self.margin*2
        new_img = np.zeros((img_x,img_y))
        self.visited = np.zeros((img_x,img_y))

        begin_x = img_x / 2 - 1
        begin_y = img_y / 2 - 1
        rand_x = random.randrange(height-3)
        rand_y = random.randrange(width-3)
        new_img[begin_x: (begin_x + 3), begin_y:(begin_y + 3)] = self.sample[rand_x: rand_x+3, rand_y:rand_y + 3]
        self.visited[begin_x: (begin_x + 3), begin_y:(begin_y + 3)] = 1
        # block list        # coorsponding coordinate of block list
        SampleImg_list_of_block = []
        list_of_coordinate = []
        for x in range(self.margin, height - self.margin):
            for y in range(self.margin, width - self.margin):
                SampleImg_list_of_block.append(self.sample[(x - self.margin):(x+1 + self.margin),(y-self.margin):(y+1+self.margin)])
                list_of_coordinate.append((x,y))
        self.image_filling(new_img, SampleImg_list_of_block, list_of_coordinate)
        new_img = new_img[self.margin: img_x - self.margin, self.margin: img_y - self.margin] #*255
        io.imshow(new_img, cmap='gray')
        #io.show()
        io.imsave(self.output_path+self.name.split('.')[0]+"_size"+str(self.window_size)+".gif",new_img)

    def efros_inpainting(self):
        height, width = self.sample.shape
        img_x = height + self.margin*2
        img_y = width + self.margin*2

        new_img = np.zeros((img_x,img_y))
        self.visited = np.zeros((img_x,img_y))
        new_img[self.margin: img_x - self.margin, self.margin:img_y - self.margin] = self.sample
        # block list         # coorsponding coordinate of block list
        SampleImg_list_of_block = []
        list_of_coordinate = []
        for (x,y),v in np.ndenumerate(new_img):
            if v == 0:
                continue
            temp = new_img[(x - self.margin):(x+1 + self.margin),(y-self.margin):(y+1+self.margin)]
            if temp[temp==0].shape[0] == 0:
                SampleImg_list_of_block.append(temp)
                list_of_coordinate.append((x - self.margin,y - self.margin))
            self.visited[x,y] = 1
        self.image_filling(new_img, SampleImg_list_of_block, list_of_coordinate)
        new_img = new_img[self.margin: img_x - self.margin, self.margin: img_y - self.margin] #*255
        io.imshow(new_img, cmap='gray')
        #io.show()
        io.imsave(self.output_path+self.name.split('.')[0]+"_size"+str(self.window_size)+".bmp",new_img)

    def efros_remove(self, blocks):
        margin = self.margin
        self.sample = color.rgb2gray(self.sample)
        height, width = self.sample.shape
        visited = np.ones(self.sample.shape)
        for block in blocks:
            if len(block) < 4:
                print "invalid input"
            visited[block[0]:block[1], block[2]:block[3]] = 0
            self.sample = np.multiply(self.sample, visited)
        self.visited = np.zeros((height + margin * 2, width+ margin*2))
        self.visited[margin:height + margin, margin: width + margin] = visited
        io.imshow(self.sample)
        io.show()
        img_x = height + self.margin*2
        img_y = width + self.margin*2

        new_img = np.zeros((img_x,img_y))
        new_img[self.margin: img_x - self.margin, self.margin:img_y - self.margin] = self.sample
        # block list         # coorsponding coordinate of block list
        SampleImg_list_of_block = []
        list_of_coordinate = []
        for (x,y),value in np.ndenumerate(new_img):
            if value == 0:
                continue
            temp = new_img[(x - margin):(x+1 + margin),(y-margin):(y+1+margin)]
            if temp[temp==0].shape[0] == 0:
                SampleImg_list_of_block.append(temp)
                list_of_coordinate.append((x - margin,y - margin))
        #print len(SampleImg_list_of_block)
        self.image_filling(new_img, SampleImg_list_of_block, list_of_coordinate)
        new_img = new_img[self.margin: img_x - self.margin, self.margin: img_y - self.margin] #*255
        io.imshow(new_img, cmap='gray')
        #io.show()
        io.imsave(self.output_path+self.name.split('.')[0]+"_Efros_remove person"+"_size"+str(self.window_size)+".bmp",new_img)
        #io.imsave(self.output_path+self.name.split('.')[0]+"_Efros_remove ground"+"_size"+str(self.window_size)+".bmp",new_img)
        #io.imsave(self.output_path+self.name.split('.')[0]+"_Efros_remove sign"+"_size"+str(self.window_size)+".bmp",new_img)
