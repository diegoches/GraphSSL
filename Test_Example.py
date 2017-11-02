#!/usr/bin/python
import math
import numpy as np
import Source as src
from time import time
#import scipy as sp
#from scipy import linalg

#-------------------------------------------------------------------------------------------------------#      
#-------------------------------------------------------------------------------------------------------#
  
# Test SECTION

#----- (begin) LOADING DATA AND SETTING EXPERIMENTS PARAMETERS (begin) -----#
data=np.loadtxt("gistpca64.txt");
label=np.loadtxt("labels.txt");
int_data=1000;
int_per=10;
string_nome="k-Graph1000G";
string_nome_data="k-Graph1000D";
string_nome_label="k-Graph1000La";
#----- (end) LOADING DATA AND SETTING EXPERIMENTS PARAMETERS (end) -----#

for i in range(0,5):

	#----- (begin) SELECTING DATA (begin) -----#
	int_data_train=int(int_data*int_per/100);
	int_data_uc=int(int_data_train/2);
	int_data_uc_tot=int(int_data/2);
	print int_per,"% ",int_data_uc," ",int_data_uc_tot;
	data_proc,labelfinal,label_proc=src.filterDataRandomly(data,label,1,int_data_uc,int_data_uc,int_data_uc_tot,int_data_uc_tot);
	string_nome_label_temp=string_nome_label+"L"+str(int_per)+"%.txt";
	np.savetxt(string_nome_label_temp,(labelfinal));
	#----- (end) SELECTING DATA (end) -----#
      
	#----- (begin) K GRAPH CONSTRUCTION (begin) -----#
	str_graph_type="K";
	floint_graph_hyper=2;
	timestart=time();
	Gr_K=src.buildKGraph(data_proc,2); # ----> Construction Method
	timefinal_gc=time()-timestart; 
	#----- (end) FULL GRAPH CONSTRUCTION (end) -----#
      
	#----- (begin) HARMONIC REGULARIZATION (begin) -----#
	print "< < Harmonic > >"; # Checking Regularization
	str_reg_type="Har";
	
	timestart=time(); 
	respH=src.HarmonicRegularizer(Gr_K,label_proc); # --------------------> Regulating
	timefinal_gr=time()-timestart;
	
	respH=src.convert_to_list(respH); # Converting to list, the result of the regulation
	
	dataH=src.float_list_normalization(respH); # Mapping results as floats
	temp_respH=dataH;
	string_nome_data_temp=string_nome_data+str_graph_type+str(floint_graph_hyper)+str_reg_type+"L"+str(int_per)+"%.txt"; # Saving results of data
	np.savetxt(string_nome_data_temp,(temp_respH));
	
	repH=src.generalizeAlphaCut(np.array(respH),0.5,0,1); # Alpha Cut in 0.5
	datH=src.int_list_normalization(repH); # Mapping cut results as ints
	
	acc=src.accuracyTest(datH,labelfinal); # Getting performance metrics
	TP,TN,FP,FN=src.confussionMatrix(datH,labelfinal);
	
	string_nome_temp=string_nome+str_graph_type+str(floint_graph_hyper)+str_reg_type+"L"+str(int_per)+"%.txt"; # Saving Performance metrics and other results
	np.savetxt(string_nome_temp,(TP,TN,FP,FN,acc,timefinal_gc,timefinal_gr));
	
	print acc;
	#----- (end) HARMONIC REGULARIZATION (end) -----#

	#----- (begin) SMOOTH OPERATOR REGULARIZATION (begin) -----#
	print "< < Smooth Operator > >"; # Checking Regularization
	str_reg_type="Smoo";
	
	timestart=time(); 
	respS=src.SmoothOperatorRegularizer(Gr_K,label_proc,int(len(Gr_K)*0.3)); # --------------------> Regulating
	timefinal_gr=time()-timestart;
	
	respS=src.convert_to_list(respS); # Converting to list, the result of the regulation
	
	dataS=src.float_list_normalization(respS); # Mapping results as floats
	temp_respS=dataS;
	string_nome_data_temp=string_nome_data+str_graph_type+str(floint_graph_hyper)+str_reg_type+"L"+str(int_per)+"%.txt"; # Saving results of data
	np.savetxt(string_nome_data_temp,(temp_respS));
	
	repS=src.generalizeAlphaCut(np.array(respS),0,0,1); # Alpha Cut in 0.5 of results
	datS=src.int_list_normalization(repS); # Mapping cut results as ints
	
	acc=src.accuracyTest(datS,labelfinal); # Getting performance metrics
	TP,TN,FP,FN=src.confussionMatrix(datS,labelfinal);
	
	string_nome_temp=string_nome+str_graph_type+str(floint_graph_hyper)+str_reg_type+"L"+str(int_per)+"%.txt"; # Saving Performance metrics and other results
	np.savetxt(string_nome_temp,(TP,TN,FP,FN,acc,timefinal_gc,timefinal_gr));
	
	print acc;
	#----- (end) SMOOTH OPERATOR REGULARIZATION (end) -----#
	
	#----- (begin) KRYLOV REGULARIZATION (begin) -----#
	print "< < Krylov > >"; # Checking Regularization
	str_reg_type="Kry";
	
	timestart=time(); 
	respK=src.KrylovRegularizer(Gr_K,label_proc,100); # --------------------> Regulating
	timefinal_gr=time()-timestart;
	
	respK=src.convert_to_list(respK); # Converting to list, the result of the regulation
	
	dataK=src.float_list_normalization(respK); # Mapping results as floats
	temp_respK=dataK;
	string_nome_data_temp=string_nome_data+str_graph_type+str(floint_graph_hyper)+str_reg_type+"L"+str(int_per)+"%.txt"; # Saving results of data
	np.savetxt(string_nome_data_temp,(temp_respK));
	
	repK=src.generalizeAlphaCut(np.array(respK),0.5,0,1); # Alpha Cut in 0.5 of results 
	datK=src.int_list_normalization(repK); # Mapping cut results as ints
	
	acc=src.accuracyTest(datK,labelfinal); # Getting performance metrics
	TP,TN,FP,FN=src.confussionMatrix(datK,labelfinal);
	
	string_nome_temp=string_nome+str_graph_type+str(floint_graph_hyper)+str_reg_type+"L"+str(int_per)+"%.txt"; # Saving Performance metrics and other results
	np.savetxt(string_nome_temp,(TP,TN,FP,FN,acc,timefinal_gc,timefinal_gr));
	
	print acc;
    #----- (end) KRYLOV REGULARIZATION (end) -----#
    
	int_per=int_per+10;
      
print "WORK COMPLETE !";
 
#-------------------------------------------------------------------------------------------------------#  


  
