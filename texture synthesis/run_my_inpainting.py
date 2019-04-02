#!/usr/bin/env python2
#-*-coding:utf-8-*-

from efros import *
import time
import sys

if __name__ == '__main__':
    # inpainting by Efros algo
    script,log = sys.argv
    f = open(log,'w+')
    __con__ = sys.stdout
    sys.stdout = f

    tests = ['test_im1.bmp','test_im2.bmp']
    for test in tests:
        for size in [5,9,11]:
            start_time = time.time()
            print 'now processing',test
            input_path = os.path.abspath(os.path.dirname(__file__))+'/pics/'
            efros = Efros(input_path, test, size)
            efros.efros_inpainting()
            end_time = time.time()
            print 'inpainting of ',test,'under window size ',size,'takes ',end_time-start_time
            print '\n'
            #raw_input()
