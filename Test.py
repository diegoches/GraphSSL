#!/usr/bin/python

import math
import numpy as np
import Source as src
from time import time
import SSLFilter as ssl
import scipy as sp
import scipy.sparse as spsp

#--------------------------------------------------------------------#

# STANDARS:

# int_ integer
# flo_ float
# ls_ python list (vector)
# tp_ tuple
# ar_ python list of python list (array, matrix)
# mt_ scipy array
# mtx_ scipy matrix
# mts_ sparse matrix
# str_ string
# val_ structure, special value (some class)

# NOTAS:

#--------------------------------------------------------------------#

def GetData(mt_data,mt_label,int_tam,int_per):
	int_dt_tr=int(int_tam*int_per/100);
	int_dt_cl=int(int_dt_tr/2);
	int_dt_tl=int(int_tam/2);
	ls_data,ls_lb_tl,ls_lb_tr=src.filterDataRandomly(mt_data,mt_label,1,int_dt_cl,int_dt_cl,int_dt_tl,int_dt_tl);
	return (ls_data,ls_lb_tl,ls_lb_tr);

def GetDataVar(mt_data,mt_label,int_tam_pos,int_tam_neg,int_per):
	int_dt_tr=int(int_tam*int_per/100);
	int_dt_cl_pos=int(int_tam_pos*int_per/100);
	int_dt_cl_neg=int(int_tam_neg*int_per/100);
	ls_data,ls_lb_tl,ls_lb_tr=src.filterDataRandomly(mt_data,mt_label,1,int_dt_cl_pos,int_dt_cl_neg,int_tam_pos,int_tam_neg);
	return (ls_data,ls_lb_tl,ls_lb_tr);
      
def BuildKGraph(ls_data,int_K):
	ls_graph=src.buildKGraph(ls_data,int_K);
	return ls_graph;
      
def BuildFullGraph(ls_data):
	ls_graph=src.buildFullGraph(ls_data);
	return ls_graph
      
def FIR_Regularizer(ls_graph,ls_lb_tr,int_fs,int_ti,flo_smo,str_file,ls_lb_tl):
  
	mt_graph=np.array(ls_graph);
	mts_graph=spsp.csr_matrix(mt_graph);
	
	# Preparing the labels vectors
	ls_lb_nu=src.copy_list(ls_lb_tr);
	ls_lb_tlnu=src.copy_list(ls_lb_tl);
	src.configureLabels(ls_lb_nu,0,-1)
	src.configureLabels(ls_lb_tlnu,0,-1)
	
	int_zad=len(ls_graph)-len(ls_lb_nu);
	for i in range(0,int_zad):
		ls_lb_nu.append(0);
	
	val_fir=ssl.SSLFilter(int_fs,ls_lb_nu,mts_graph,flo_smo);
	
	flo_timestart=time();
	mtx_lb_pr=val_fir.regularize(int_ti);
	flo_timefinal=time()-flo_timestart;
	
	# Getting Results
	
	ls_lb_pr=src.convert_to_list(mtx_lb_pr);
	ls_lb_prflo=src.float_list_normalization(ls_lb_pr);
	ls_temp_lb=ls_lb_prflo;
	
	np.savetxt("Pr"+str_file,(tuple(ls_temp_lb)+tuple(ls_lb_tlnu)))
	
	ls_lb_alpha=src.generalizeAlphaCut(np.array(ls_lb_pr),0,-1,1);
	ls_lb_alphaint=src.int_list_normalization(ls_lb_alpha);
	
	flo_acc=src.accuracyTest(ls_lb_alphaint,ls_lb_tlnu);
	flo_TP,flo_TN,flo_FP,flo_FN=src.confussionMatrix(ls_lb_alphaint,ls_lb_tlnu);
	
	np.savetxt("Rs"+str_file,(flo_TP,flo_TN,flo_FP,flo_FN,flo_acc,flo_timefinal));
	
	print str_file+" done!"
	return ls_lb_pr
	
def HAR_Regularizer(ls_graph,ls_lb_tr,str_file,ls_lb_tl):
	
	ls_lb_nu=src.copy_list(ls_lb_tr); 
	ls_lb_tlnu=src.copy_list(ls_lb_tl);
	src.configureLabels(ls_lb_nu,0,-1)
	src.configureLabels(ls_lb_tlnu,0,-1)
	
	#raw_input("--pausa-- ");
	
	flo_timestart=time(); 
	mtx_lb_pr=src.HarmonicRegularizer(ls_graph,ls_lb_nu);
	flo_timefinal=time()-flo_timestart;
	
	# Getting Results
	
	ls_lb_pr=src.convert_to_list(mtx_lb_pr)
	ls_lb_prflo=src.float_list_normalization(ls_lb_pr)
	ls_temp_lb=ls_lb_prflo
	
	np.savetxt("Pr"+str_file,(tuple(ls_temp_lb)+tuple(ls_lb_tlnu)))
	
	ls_lb_alpha=src.generalizeAlphaCut(np.array(ls_lb_pr),0,-1,1)
	ls_lb_alphaint=src.int_list_normalization(ls_lb_alpha)
	
	flo_acc=src.accuracyTest(ls_lb_alphaint,ls_lb_tlnu)
	flo_TP,flo_TN,flo_FP,flo_FN=src.confussionMatrix(ls_lb_alphaint,ls_lb_tlnu)
	
	np.savetxt("Rs"+str_file,(flo_TP,flo_TN,flo_FP,flo_FN,flo_acc,flo_timefinal))
	
	print str_file+" done!"
	return ls_lb_pr
	
def KRY_Regularizer(ls_graph,ls_lb_tr,int_iter,str_file,ls_lb_tl):
	
	ls_lb_nu=src.copy_list(ls_lb_tr);
	ls_lb_tlnu=src.copy_list(ls_lb_tl);
	
	flo_timestart=time();
	mtx_lb_pr=src.KrylovRegularizer(ls_graph,ls_lb_nu,int_iter);
	flo_timefinal=time()-flo_timestart;
	
	# Getting Results
	
	ls_lb_pr=src.convert_to_list(mtx_lb_pr);
	ls_lb_prflo=src.float_list_normalization(ls_lb_pr);
	ls_temp_lb=ls_lb_prflo;
	
	np.savetxt("Pr"+str_file,(tuple(ls_temp_lb)+tuple(ls_lb_tlnu)))
	
	ls_lb_alpha=src.generalizeAlphaCut(np.array(ls_lb_pr),0.5,0,1);
	ls_lb_alphaint=src.int_list_normalization(ls_lb_alpha);
	
	flo_acc=src.accuracyTest(ls_lb_alphaint,ls_lb_tlnu)
	flo_TP,flo_TN,flo_FP,flo_FN=src.confussionMatrix(ls_lb_alphaint,ls_lb_tlnu)
	
	np.savetxt("Rs"+str_file,(flo_TP,flo_TN,flo_FP,flo_FN,flo_acc,flo_timefinal))
	
	print str_file+" done!"
	return ls_lb_pr
	
def SMO_Regularizer(ls_graph,ls_lb_tr,flo_per,str_file,ls_lb_tl):
	
	ls_lb_nu=src.copy_list(ls_lb_tr);
	ls_lb_tlnu=src.copy_list(ls_lb_tl);
	src.configureLabels(ls_lb_nu,0,-1)
	src.configureLabels(ls_lb_tlnu,0,-1)
	
	#raw_input("--pausa-- ");
	
	flo_timestart=time();
	mtx_lb_pr=src.SmoothOperatorRegularizer(ls_graph,ls_lb_nu,int(len(ls_graph)*flo_per));
	flo_timefinal=time()-flo_timestart;
	
	# Getting Results
	
	ls_lb_pr=src.convert_to_list(mtx_lb_pr);
	ls_lb_prflo=src.float_list_normalization(ls_lb_pr);
	ls_temp_lb=ls_lb_prflo;
	
	np.savetxt("Pr"+str_file,(tuple(ls_temp_lb)+tuple(ls_lb_tlnu)))
	
	ls_lb_alpha=src.generalizeAlphaCut(np.array(ls_lb_pr),0,-1,1);
	ls_lb_alphaint=src.int_list_normalization(ls_lb_alpha);
	
	flo_acc=src.accuracyTest(ls_lb_alphaint,ls_lb_tlnu);
	flo_TP,flo_TN,flo_FP,flo_FN=src.confussionMatrix(ls_lb_alphaint,ls_lb_tlnu);
	
	np.savetxt("Rs"+str_file,(flo_TP,flo_TN,flo_FP,flo_FN,flo_acc,flo_timefinal));
	
	print str_file+" done!"
	return ls_lb_pr
	
def EIF_Regularizer(ls_data,ls_lb_tr,flo_sigma,int_eig,flo_epsilon,str_file,ls_lb_tl):
  
	ls_lb_nu=src.copy_list(ls_lb_tr);
	ls_lb_tlnu=src.copy_list(ls_lb_tl);
	src.configureLabels(ls_lb_nu,0,-1)
	src.configureLabels(ls_lb_tlnu,0,-1)
	
	#raw_input("--pausa-- ");
	
	flo_timestart=time();
	mtx_lb_pr=src.EigenFunctionRegularizer(ls_data,ls_lb_nu,flo_sigma,int_eig,flo_epsilon);
	flo_timefinal=time()-flo_timestart;
	
	# Getting Results
	
	ls_lb_pr=src.convert_to_list(mtx_lb_pr);
	ls_lb_prflo=src.float_list_normalization(ls_lb_pr);
	ls_temp_lb=ls_lb_prflo;
	
	np.savetxt("Pr"+str_file,(tuple(ls_temp_lb)+tuple(ls_lb_tlnu)))
	
	ls_lb_alpha=src.generalizeAlphaCut(np.array(ls_lb_pr),0,-1,1);
	ls_lb_alphaint=src.int_list_normalization(ls_lb_alpha);
	
	flo_acc=src.accuracyTest(ls_lb_alphaint,ls_lb_tlnu);
	flo_TP,flo_TN,flo_FP,flo_FN=src.confussionMatrix(ls_lb_alphaint,ls_lb_tlnu);
	
	np.savetxt("Rs"+str_file,(flo_TP,flo_TN,flo_FP,flo_FN,flo_acc,flo_timefinal));
	
	print str_file+" done!"
	return ls_lb_pr;
	
def NYS_Regularizer(ls_data,ls_lb_tr,int_Kg,int_sampling,int_Knn,str_file,ls_lb_tl):
  
	ls_lb_nu=src.copy_list(ls_lb_tr);
	ls_lb_tlnu=src.copy_list(ls_lb_tl);
	
	flo_timestart=time();
	mtx_lb_pr=src.NystromIsomapKNN(ls_data,ls_lb_nu,int_Kg,int_sampling,int_Knn);
	flo_timefinal=time()-flo_timestart;
	
	# Getting Results
	
	ls_lb_pr=src.convert_to_list(mtx_lb_pr);
	ls_lb_prflo=src.float_list_normalization(ls_lb_pr);
	ls_temp_lb=ls_lb_prflo;
	
	np.savetxt("Pr"+str_file,(tuple(ls_temp_lb)+tuple(ls_lb_tlnu)))
	
	ls_lb_alpha=src.generalizeAlphaCut(np.array(ls_lb_pr),0.5,0,1);
	ls_lb_alphaint=src.int_list_normalization(ls_lb_alpha);
	
	flo_acc=src.accuracyTest(ls_lb_alphaint,ls_lb_tlnu);
	flo_TP,flo_TN,flo_FP,flo_FN=src.confussionMatrix(ls_lb_alphaint,ls_lb_tlnu);
	
	np.savetxt("Rs"+str_file,(flo_TP,flo_TN,flo_FP,flo_FN,flo_acc,flo_timefinal));
	
	print str_file+" done!"
	return ls_lb_pr;
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
      