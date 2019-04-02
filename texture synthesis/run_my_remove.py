#!/usr/bin/env python2
#-*-coding:utf-8-*-

from efros import *
from CPT import *
import time
import sys

if __name__ == '__main__':
    # remove and inpainting by Efros algo and CPT algo
    script,log = sys.argv
    f = open(log,'a +')
    __con__ = sys.stdout
    sys.stdout = f
    # efors part
    '''
    tests = ['test_im3.jpg']
    for test in tests:
        for size in [9]:
            start_time = time.time()
            print 'now processing',test
            input_path = os.path.abspath(os.path.dirname(__file__))+'/pics/'
            efros = Efros(input_path,test,size)
            #person
            efros.efros_remove([(350,480,225,255)])
            end_time = time.time()
            print 'remove person and inpaint by Efros algo with window size of ',size,' takes ',end_time-start_time

            #sign
            #efros.efros_remove([(510,570,765,830),(560,670,785, 810)])
            #end_time = time.time()
            #print 'remove sign and inpaint by Efros algo with window size of ',size,' takes ',end_time-start_time

            #ground
            #efros.efros_remove([(620, 670,0,400), (580, 620, 40, 470), (530,580,140,535), (490, 530, 320,600), (465,490, 420, 630),(440,465, 515,660)])
            #end_time = time.time()
            #print 'remove ground and inpaint by Efros algo with window size of ',size,' takes ',end_time-start_time

            #raw_input()
    '''
    #CPT_algorithm
    tests = ['test_im3.jpg']
    for test in tests:
        for size in [9]:
            start_time = time.time()
            print 'now processing',test
            input_path = os.path.abspath(os.path.dirname(__file__))+'/pics/'
            # crinimis part
            cpt = CPT(input_path,test,size)
            # person
            #cpt.remove_blocks([(350,480,225,255)])
            #cpt.CPT_synthesize()
            #end_time = time.time()
            #print 'remove person and inpaint by CPT algo with window size of ',size,' takes ',end_time-start_time

            # sign
            #cpt.remove_blocks([(513,567,770,830),(566,664,788, 803)])
            #cpt.CPT_synthesize()
            #end_time = time.time()
            #print 'remove sign and inpaint by CPT algo with window size of ',size,' takes ',end_time-start_time

            # ground
            cpt.remove_blocks([(620, 670,0,400), (580, 620, 40, 470), (530,580,140,535), (490, 530, 320,600), (465,490, 420, 630),(440,465, 515,660)])
            cpt.CPT_synthesize()
            end_time = time.time()
            print 'remove ground and inpaint by CPT algo with window size of ',size,' takes ',end_time-start_time
