#! /usr/bin/env python
# -*- coding: iso-8859-1 -*-

import numpy as np
import scipy as sp
import Source as src
import SSLFilter as ssl
import Test as ts
import math
import time
import thread
#import scipy as sp
#from scipy import linalg

#-----------------------------------------------------------#

data=np.loadtxt("X.txt");
label=np.loadtxt("Y.txt");

int_tam_data=1500;  # Data Set size, number of data points
int_per_tr=10;       # Percentage of data to be labeled
int_K=5;            # K o the K-graph

#-----------------------------------------------------------#

dt,lb1,lb2=ts.GetData(data,label,int_tam_data,int_per_tr)

try:
   thread.start_new_thread(ts.EIF_Regularizer,(dt,lb2,0.2,200,0.1,"EIFtest.txt",lb1))
   thread.start_new_thread(ts.NYS_Regularizer,(dt,lb2,int_K,30,5,"NYStest.txt",lb1))
except:
   print "Error: unable to start thread"

gr=ts.BuildKGraph(dt,int_K)

try:
   thread.start_new_thread(ts.FIR_Regularizer,(gr,lb2,2,5,0.5,"FIRtest(2-5).txt",lb1))
   thread.start_new_thread(ts.FIR_Regularizer,(gr,lb2,2,10,0.5,"FIRtest(2-10).txt",lb1))
   thread.start_new_thread(ts.FIR_Regularizer,(gr,lb2,2,15,0.5,"FIRtest(2-15).txt",lb1))
   thread.start_new_thread(ts.FIR_Regularizer,(gr,lb2,2,20,0.5,"FIRtest(2-20).txt",lb1))
   
   thread.start_new_thread(ts.FIR_Regularizer,(gr,lb2,3,5,0.5,"FIRtest(3-5).txt",lb1))
   thread.start_new_thread(ts.FIR_Regularizer,(gr,lb2,3,10,0.5,"FIRtest(3-10).txt",lb1))
   thread.start_new_thread(ts.FIR_Regularizer,(gr,lb2,3,15,0.5,"FIRtest(3-15).txt",lb1))
   thread.start_new_thread(ts.FIR_Regularizer,(gr,lb2,3,20,0.5,"FIRtest(3-20).txt",lb1))
   
   thread.start_new_thread(ts.FIR_Regularizer,(gr,lb2,4,5,0.5,"FIRtest(4-5).txt",lb1))
   thread.start_new_thread(ts.FIR_Regularizer,(gr,lb2,4,10,0.5,"FIRtest(4-10).txt",lb1))
   thread.start_new_thread(ts.FIR_Regularizer,(gr,lb2,4,15,0.5,"FIRtest(4-15).txt",lb1))
   thread.start_new_thread(ts.FIR_Regularizer,(gr,lb2,4,20,0.5,"FIRtest(4-20).txt",lb1))
   
   thread.start_new_thread(ts.FIR_Regularizer,(gr,lb2,5,5,0.5,"FIRtest(5-5).txt",lb1))
   thread.start_new_thread(ts.FIR_Regularizer,(gr,lb2,5,10,0.5,"FIRtest(5-10).txt",lb1))
   thread.start_new_thread(ts.FIR_Regularizer,(gr,lb2,5,15,0.5,"FIRtest(5-15).txt",lb1))
   thread.start_new_thread(ts.FIR_Regularizer,(gr,lb2,5,20,0.5,"FIRtest(5-20).txt",lb1))
   
   thread.start_new_thread(ts.FIR_Regularizer,(gr,lb2,10,5,0.5,"FIRtest(10-5).txt",lb1))
   thread.start_new_thread(ts.FIR_Regularizer,(gr,lb2,10,10,0.5,"FIRtest(10-10).txt",lb1))
   thread.start_new_thread(ts.FIR_Regularizer,(gr,lb2,10,15,0.5,"FIRtest(10-15).txt",lb1))
   thread.start_new_thread(ts.FIR_Regularizer,(gr,lb2,10,20,0.5,"FIRtest(10-20).txt",lb1))
   
   thread.start_new_thread(ts.FIR_Regularizer,(gr,lb2,15,5,0.5,"FIRtest(15-5).txt",lb1))
   thread.start_new_thread(ts.FIR_Regularizer,(gr,lb2,15,10,0.5,"FIRtest(15-10).txt",lb1))
   thread.start_new_thread(ts.FIR_Regularizer,(gr,lb2,15,15,0.5,"FIRtest(15-15).txt",lb1))
   thread.start_new_thread(ts.FIR_Regularizer,(gr,lb2,15,20,0.5,"FIRtest(15-20).txt",lb1))   
   
   thread.start_new_thread(ts.FIR_Regularizer,(gr,lb2,20,5,0.5,"FIRtest(20-5).txt",lb1))
   thread.start_new_thread(ts.FIR_Regularizer,(gr,lb2,20,10,0.5,"FIRtest(20-10).txt",lb1))
   thread.start_new_thread(ts.FIR_Regularizer,(gr,lb2,20,15,0.5,"FIRtest(20-15).txt",lb1))
   thread.start_new_thread(ts.FIR_Regularizer,(gr,lb2,20,20,0.5,"FIRtest(20-20).txt",lb1))
   
   thread.start_new_thread(ts.HAR_Regularizer,(gr,lb2,"HARtest.txt",lb1))
   thread.start_new_thread(ts.KRY_Regularizer,(gr,lb2,30,"KRYtest.txt",lb1))
   thread.start_new_thread(ts.SMO_Regularizer,(gr,lb2,0.9,"SMOtest.txt",lb1))
except:
   print "Error: unable to start thread"
      
print "WORK COMPLETE !";

while 1:
   pass

 



  
