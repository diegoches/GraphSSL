# -*- coding: utf-8 -*-
#!/usr/bin/python

import Source as src
import random

class Neuron:
	"""Neuron"""
	def __init__(self,vec_l,float_e):
		self.__vector=vec_l; #lista, vector
		self.__error=float_e; #flotante
		
	def __str__(self):
		return "coordinates: "+str(self.__vector)+"\nerror: "+str(self.__error);

	def get_vector(self):
		return self.__vector;
	
	def get_error(self):
		return self.__error;
		
	def set_vector(self,vec_l):
		self.__vector=vec_l;
	
	def set_error(self,float_e):
		self.__error=float_e;
		
	def get_distance(self,neuron_o):
		if(isinstance(neuron_o,Neuron)):
			return src.distance(self.__vector,neuron_o.get_vector());
		else:
			print "ERROR (Neuron.get_distance): input parameter is not a neuron -> ", type(neuron_o);
		
	def update_error(self,vec_x):
		int_t1=len(self.__vector);
		int_t2=len(vec_x);
		if(int_t1==int_t2):
			self.__error=self.__error+src.distance(self.__vector,vec_x);
		else:
			print "ERROR (Neuron.update_error): input vector of wrong size -> ", int_t2;
	
	def update_vector(self,vec_x,float_f):
		int_t1=len(self.__vector);
		int_t2=len(vec_x);
		if(int_t1==int_t2 and float_f<=1 and float_f>0):
			for i in range(0,int_t1):
				self.__vector[i]=self.__vector[i]+( float_f*(vec_x[i]-self.__vector[i]) );
		else:
			print "ERROR (Neuron.update_vector): worng input parameters ->  vector size: ", int_t2," error factor: ", float_f;
	
	def decrease_error(self,float_f):
		if(float_f<=1 and float_f>0):
			self.__error=self.__error-(float_f*self.__error);
		else:
			print "ERROR (Neuron.decrease_error): error factor not allowed -> ",float_f;

	def become_random(self,float_i,float_f,int_tam):
		self.__vector=[];
		for i in range(0,int_tam):
			float_temp=src.randomfloat(float_i,float_f);
			self.__vector.append(float_temp);
			
	def become_middle(self,neu_n1,neu_n2,float_alpha):
		int_t1=len(neu_n1.get_vector());
		int_t2=len(neu_n2.get_vector());
		if(int_t1==int_t2 and float_alpha<=1 and float_alpha>0):
			self.__vector=[];
			for i in range(0,int_t1):
				float_temp=0.5*(neu_n1.get_vector()[i]+neu_n2.get_vector()[i]);
				self.__vector.append(float_temp);
			neu_n1.set_error( neu_n1.get_error()-(float_alpha*neu_n1.get_error()) );
			neu_n2.set_error( neu_n2.get_error()-(float_alpha*neu_n2.get_error()) );
			self.__error=0.5*(neu_n1.get_error()+neu_n2.get_error());
		else:
			print "ERROR (Neuron.update_vector): worng input parameters ->  vector sizes: ", int_t1, int_t2," error alpha: ", float_alpha;

##------------------------------------------------------------------------##

