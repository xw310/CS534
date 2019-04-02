#!/usr/bin/env python2
#-*-coding:utf-8-*-

from efros import *
import time
import sys

if __name__ == '__main__':
    # texture synthesize by Efros algo
    script,log = sys.argv
    f = open(log,'w+')
    __con__ = sys.stdout
    sys.stdout = f

    tests = ['T1.gif','T2.gif','T3.gif','T4.gif','T5.gif']
    for test in tests:
        for size in [5,9,11,15]:
            start_time = time.time()
            print 'now processing',test
            input_path = os.path.abspath(os.path.dirname(__file__))+'/pics/'
            efros = Efros(input_path, test, size)
            efros.efros_synthesis(200, 200)
            end_time = time.time()
            print 'synthesis of ',test,'under window size ',size,'takes ',end_time-start_time
            print '\n'
            #raw_input()
