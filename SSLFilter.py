#! /usr/bin/env python
# -*- coding: iso-8859-1 -*-

import numpy as np
import scipy as sp
import graphssl as gssl

#--------------------------------------------------------------------#

# STANDARS:

# int_ integer
# flo_ float
# ls_ python list (vector)
# tp_ tuple
# ar_ python list of python list (array, matrix)
# mt_ scipy array
# mtx_ scipy matrix
# str_ string
# val_ structure, special value (some class)

# NOTAS:

#--------------------------------------------------------------------#

class SSLFilter:
	"""SSLFilter"""
	def __init__(self,int_fs,val_labels,val_sim,flo_s):
		self.__val_X=val_labels # Labels of a few of the vertices (0 unlabeled, +-1 for classes)
		self.__int_K=int_fs 	# Filter size
		self.__val_W=val_sim 	# Symilarity matrix [sparce scipy_matrix]
		self.__flo_w=flo_s 	# Weight of Smoothness (0) versus Labels (1)
		
	def __str__(self):
		return "Filter size: "+str(self.__int_K)
		
	def set_filter_size(self,int_fs):
		self.__int_K=int_fs
		
	def get_filter_size(self):
		return self.__int_K
		
	def set_labels(self,val_labels):
		self.__val_X=val_labels
		
	def get_labels(self):
		return self.__val_labels
		
	def set_similarity_matrix(self,val_sim):
		self.__val_W=val_sim
		
	def get_similarity_matrix(self):
		return self.__val_W
		
	def set_smoothness(self,flo_s):
		self.__flo_w=flo_s
		
	def get_smoothness(self):
		return self.__flo_w
		
	def regularize(self, int_t):
		val_L = gssl.symNormLaplacian(self.__val_W)
		ls_f = gssl.calcFilterCoef(val_L , self.__val_W , np.array(self.__val_X) , self.__int_K , self.__flo_w)
		##
		mt_den_L=val_L.todense()
		mt_eig=np.linalg.eig(mt_den_L);
		flo_max_eig=max(mt_eig[0]);
		np.savetxt(str(self.__int_K)+"-"+str(int_t)+"-"+str(self.__flo_w)+"FIReig.txt",(flo_max_eig,flo_max_eig))
		np.savetxt(str(self.__int_K)+"-"+str(int_t)+"-"+str(self.__flo_w)+"FIRCoeffi.txt",(ls_f))
		##
		val_lab=gssl.fastFIRGraph(ls_f,val_L,self.__val_X)
		for i in range(0,int_t-1):
			val_lab=gssl.fastFIRGraph(ls_f,val_L,val_lab)	
		return val_lab

		
		
		
		
		
		
		
		
		
		
		    
	      
	      
	      
	      
	      