#!/usr/bin/env python2
#-*-coding:utf-8-*-
'''
CPT(Criminisi, Perez and Toyama) algorithm for CS534 HW2
'''
import numpy as np
import math
import time
from skimage import filters
from skimage import io, morphology, color
import os
from Gauss import gaussian

class CPT:
    output_path = os.path.abspath(os.path.dirname(__file__))+'/output/'

    def __init__(self, path, file_name, window_size):
        self.sample = io.imread(path+file_name).astype('float64')
        #print(type(self.sample))
        self.sample_rgb = self.sample
        self.img_rgb = self.sample
        self.sample = color.rgb2gray(self.sample)
        self.OriginSample = self.sample
        self.window_size = window_size
        self.margin = self.window_size / 2
        self.name = file_name

        self.visited = None
        self.img = None

        self.x_grad = None
        self.y_grad = None
        self.MatBound = None
        self.GaussMask = None

    def find_matches(self, template, img, GaussMask, SampleImg_list_of_block, list_of_coordinate, err_thres = -1.0):
        ValidMask = self.visited[template[0]:template[1], template[2]: template[3]]
        blocks = img[template[0]:template[1], template[2]: template[3]]
        # get the GaussMask, shift it make it same to the shape of ValidMask
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

    def remove_blocks(self, block_list):
        margin = self.margin
        height, width = self.sample.shape
        visited = np.ones(self.sample.shape)
        for block in block_list:
            if len(block) < 4:
                print "invalid block input"
            visited[block[0]:block[1], block[2]:block[3]] = 0
            self.sample = np.multiply(self.sample, visited)
            for (x,y),value in np.ndenumerate(visited):
                if value == 0:
                    self.img_rgb[x,y] = (0,0,0)
        self.visited = np.zeros((height + margin * 2, width+ margin*2))
        self.visited[margin:height + margin, margin: width + margin] = visited
        io.imshow(self.sample,cmap='gray')
        io.show()
        temp_rgb = np.zeros((height + margin * 2, width+ margin*2,3))
        temp_rgb[margin:height + margin, margin: width + margin,] = self.img_rgb
        self.img_rgb = temp_rgb

    def generating_mat(self):
        enlarge_x, enlarge_y = self.visited.shape
        margin = self.margin
        self.x_grad = filters.scharr_h(self.OriginSample)
        self.y_grad = filters.scharr_v(self.OriginSample)
        self.img = np.zeros((enlarge_x, enlarge_y))
        self.img[margin: enlarge_x - margin, margin: enlarge_y - margin] = self.sample
        # specified for boundary and gradient get
        self.MatBound = np.ones((enlarge_x, enlarge_y))
        self.MatBound[margin: enlarge_x - margin, margin: enlarge_y - margin] = self.visited[margin: enlarge_x - margin, margin: enlarge_y - margin]
        # get sample block list
        SampleImg_list_of_block = []
        # get coorsponding coordinate
        list_of_coordinate = []
        for (x,y),v in np.ndenumerate(self.img):
            if v == 0:
                continue
            tmp = self.img[(x - margin):(x+1 + margin),(y-margin):(y+1+margin)]
            if tmp[tmp==0].shape[0] == 0:
                SampleImg_list_of_block.append(tmp)
                list_of_coordinate.append((x - margin,y - margin))
            self.visited[x,y] = 1
        sigma = self.window_size / 6.4
        self.GaussMask = gaussian((self.window_size,self.window_size), sigma)
        return [SampleImg_list_of_block, list_of_coordinate]

    def get_order(self, remaining_list):
        MinPrior = 0.0
        MinPixel = (self.margin, self.margin)
        margin = self.margin
        self.OriginSample.shape[0]
        grad_x = filters.scharr_h(self.MatBound[margin:self.OriginSample.shape[0] + margin, margin:self.OriginSample.shape[1] + margin])
        grad_y = filters.scharr_v(self.MatBound[margin:self.OriginSample.shape[0] + margin, margin:self.OriginSample.shape[1] + margin])
        for pixel in remaining_list:
            if pixel[0] < margin or pixel[0] >= self.OriginSample.shape[0] + margin or pixel[1] < margin or pixel[1] < margin or pixel[1] >= self.OriginSample.shape[1] + margin:
                continue
            temp = (pixel[0] - margin, pixel[0] + 1 + margin, pixel[1] - margin, pixel[1] + 1 + margin)
            conf = self.visited[temp[0]:temp[1], temp[2]:temp[3]].sum()
            sch_dx = self.x_grad[pixel[0] - margin, pixel[1] - margin]
            sch_dy = self.y_grad[pixel[0] - margin, pixel[1] -margin]
            norm = math.sqrt(sch_dx*sch_dx + sch_dy*sch_dy)
            if norm != 0:
                sch_dx /= norm
                sch_dy /= norm
            dx = grad_y[pixel[0] - margin, pixel[1] - margin]
            dy = grad_x[pixel[0] - margin, pixel[1] - margin]
            norm = math.sqrt(dx*dx + dy*dy)
            if norm != 0:
                dx /= norm
                dy /= norm
            v1 = math.fabs(-dx*sch_dx + dy*sch_dy)
            v2 = math.fabs(dx*sch_dx + -dy*sch_dy)
            priority = max(v2, v1)*conf
            if priority >= MinPrior:
                MinPixel = pixel
                MinPrior = priority
        return MinPixel

    def get_remaining_list(self):

        Bound = filters.laplace(self.MatBound)

        return zip(*np.where(np.multiply(Bound, self.MatBound) != 0))

    def CPT_synthesize(self):
        SampleImg_list_of_block, list_of_coordinate = self.generating_mat()
        margin = self.margin
        while True:
            remaining_list = self.get_remaining_list()
            if len(remaining_list) == 0:
                break
            pixel = self.get_order(remaining_list)
            template = (pixel[0] - margin, pixel[0] + 1 + margin, pixel[1] - margin, pixel[1] + 1 + margin)
            picked,error = self.find_matches(template, self.img, self.GaussMask, np.asarray(SampleImg_list_of_block), list_of_coordinate)
            fangwen = abs(self.MatBound[template[0]: template[1], template[2]: template[3]] - 1)
            fangwen_rgb = np.ones((fangwen.shape[0],fangwen.shape[1],3))
            for (x,y),value in np.ndenumerate(fangwen):
                if value == 1:
                    fangwen_rgb[x,y] = np.array([1,1,1])
                else:
                    fangwen_rgb[x,y] = np.array([0,0,0])
            #print pixel, picked
            self.img[template[0]:template[1], template[2]:template[3]] += np.multiply(self.sample[picked[0]-margin:picked[0]+1+margin, picked[1]-margin: picked[1]+1+margin], fangwen)
            self.img_rgb[template[0]:template[1], template[2]:template[3]] += np.multiply(self.sample_rgb[picked[0]-margin:picked[0]+1+margin, picked[1]-margin: picked[1]+1+margin], fangwen_rgb)
            self.visited[template[0]: template[1], template[2]: template[3]] += fangwen
            self.MatBound[template[0]: template[1], template[2]: template[3]] += fangwen
        self.img = self.img[margin: self.img.shape[0] - margin, margin: self.img.shape[1] - margin]/256.0
        self.img_rgb = self.img_rgb[margin: self.img.shape[0] - margin, margin: self.img.shape[1] - margin,]
        #io.imshow(self.sample,cmap='gray')
        #io.show()
        io.imshow(self.img, cmap='gray')
        #print(self.img)
        #print(self.img.max())
        #print(self.img.min())
        #io.show()
        #io.imsave(self.output_path+self.name.split('.')[0]+"_CPT_remove person"+"_size"+str(self.window_size)+".jpg",self.img)
        io.imsave(self.output_path+self.name.split('.')[0]+"_CPT_remove ground"+"_size"+str(self.window_size)+".jpg",self.img)
        #io.imsave(self.output_path+self.name.split('.')[0]+"_CPT_remove sign"+"_size"+str(self.window_size)+".jpg",self.img)
