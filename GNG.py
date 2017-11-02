# -*- coding: utf-8 -*-
#!/usr/bin/python

import Source as src
import random
import Neuron as neu
import time
import Tkinter
import tkMessageBox
#from collections import Counter
import numpy as np

# NOTAS DE PROGRAMACION #
# Revisar el comportamiento del vector de activaciones al eliminarse una neurona. (listo)
# Tomar en cuenta si una activacion queda en -1 debe ser recalculada con las neuronas restantes
#	tomando en cuenta la neurona mas cercana. (listo)
# Krylov no aguanta grafos no conexos.
# --------------------- #


class GNG:
	"""GNG"""
	def __init__(self):
		self.__epsilon_s=0.5; # float, proporcion de modificacion de la neurona ganadora
		self.__epsilon_t=0.2; # float, proporcion de modificacion de las neuronas vecinas a la ganadora
		self.__lambda=30; # int, numero de iteraciones en la que se inserta una neurona
		self.__alpha=0.2; # float, proporcion en la que se reduce el error al insertarse una nueva neurona
		self.__beta=8; # int, maxima posible vida de una arista
		self.__delta=0.05; # float, proporcion de reduccion del error en cada iteraccion
		
		self.__neurons=[]; # vector, lista de neuronas
		self.__pesos=[]; # vector de vectores, lista de listas de float, matriz de adyacencia.
		self.__vidas=[]; # vector de vectores, lista de listas de int, matriz de vida de las aristas.
		
		self.__patterns=[]; 
		self.__activaciones=[]; # vector, guarda los indices de las neuronas que activaron cada patron;
		
		self.__tamano=0; # cantidad de neuronas
		self.__dimension=0;
		self.__max_iterations=200;
		
	def set_hyperparameters(self,float_eps,float_ept,int_lam,float_alp,int_be,float_del):
		self.__epsilon_s=float_eps;
		self.__epsilon_t=float_ept;
		self.__lambda=int_lam;
		self.__alpha=float_alp;
		self.__beta=int_be;
		self.__delta=float_del;
		
	def set_max_iterations(self,int_max):
		self.__max_iterations=int_max;
		
	def set_patterns(self,vec_vec_float_pat):
		if(isinstance(vec_vec_float_pat,list)):
			self.__patterns=vec_vec_float_pat;
			self.__dimension=len(self.__patterns[0]);
			self.__activaciones=[-1]*len(self.__patterns);
		else:
			print "ERROR (GNG.set_patterns): incorrect patterns format -> ", type(vec_vec_float_pat);
			
	def get_patterns(self):
		return self.__patterns;
		
	def get_neurons(self):
		return self.__neurons;
	      
	def get_tamano(self):
		return self.__tamano;
		
	def are_connected(self,int_i,int_j):
		if(int_i>=self.__tamano or int_i<0):# previous validations
			print "ERROR (GNG.are_connected): incorrect int_i value -> ", int_i;
			return;
		if(int_j>=self.__tamano or int_j<0):
			print "ERROR (GNG.are_connected): incorrect int_j value -> ", int_j;
			return;
		# algorithm
		bool_resp=self.__vidas[int_i][int_j]>-1;
		return bool_resp;
		
	def set_live(self,int_i,int_j,int_val):
		if(int_i>=self.__tamano or int_i<0):# previous validations
			print "ERROR (GNG.set_live): incorrect int_i value -> ", int_i;
			return;
		if(int_j>=self.__tamano or int_j<0):
			print "ERROR (GNG.set_live): incorrect int_j value -> ", int_j;
			return;
		if(isinstance(int_val,int)==False):
			print "ERROR (GNG.set_live): incorrect int_val value -> ", type(int_val);
			return;
		# algorithm
		self.__vidas[int_i][int_j]=int_val;
		self.__vidas[int_j][int_i]=int_val;
		
	def print_neurons(self):
		int_tam=len(self.__neurons);
		for i in range(0,int_tam):
			print self.__neurons[i];
	
	def print_pesos(self):
		int_tam=len(self.__pesos);
		for i in range(0,int_tam):
			print self.__pesos[i];
			
	def print_vidas(self):
		int_tam=len(self.__vidas);
		for i in range(0,int_tam):
			print self.__vidas[i];
			
	def print_GNG(self):
		print "self.__tamano: ", self.__tamano;
		print "self.__dimension: ", self.__dimension;
		print "self.__neurons:";
		self.print_neurons();
		print "self.__pesos:";
		self.print_pesos();
		print "self.__vidas:";
		self.print_vidas();
		
	def print_GNG_data(self):
		print "self.__activaciones[i] | self.__patterns[i]:";
		int_tam_patterns=len(self.__patterns);
		for i in range(0,int_tam_patterns):
			print self.__activaciones[i],"|",self.__patterns[i];
	
	def initialize(self):
		float_max=(max(self.__patterns[0]))+1;
		float_min=(min(self.__patterns[0]))-1;
		int_tam_patt=len(self.__patterns[0]);
		neuron_fn=neu.Neuron([],0.0);
		neuron_sn=neu.Neuron([],0.0);
		neuron_fn.become_random(float_min,float_max,int_tam_patt);
		neuron_sn.become_random(float_min,float_max,int_tam_patt);
		self.__neurons.append(neuron_fn);
		self.__neurons.append(neuron_sn);
		self.__tamano=2;
		self.__pesos.append(([-1.0]*2));
		self.__vidas.append(([-1]*2));
		self.__pesos.append(([-1.0]*2));
		self.__vidas.append(([-1]*2));
		
	def add_neuron(self,neuron_new,int_p,int_q):
		if(int_p>=self.__tamano or int_p<0):# previous validations
			print "ERROR (GNG.add_neuron): incorrect int_p value -> ", int_p;
			return;
		if(int_q>=self.__tamano or int_q<0):
			print "ERROR (GNG.add_neuron): incorrect int_q value -> ", int_q;
			return;
		if(isinstance(neuron_new,neu.Neuron)==False):
			print "ERROR (GNG.add_neuron): incorrect neuron_new type -> ", type(neuron_new);
			return;
		# adding algorithm
		self.__neurons.append(neuron_new);
		for i in range(0,self.__tamano):
			self.__pesos[i].append(-1.0);
			self.__vidas[i].append(-1);
		self.__tamano=self.__tamano+1;
		vec_float_new=[-1.0]*(self.__tamano);
		vec_int_new=[-1]*(self.__tamano);
		self.__pesos.append(vec_float_new);
		self.__vidas.append(vec_int_new);
		float_dist_p=neuron_new.get_distance(self.__neurons[int_p]);
		float_dist_q=neuron_new.get_distance(self.__neurons[int_q]);
		self.__pesos[self.__tamano-1][int_p]=float_dist_p;
		self.__pesos[self.__tamano-1][int_q]=float_dist_q;
		self.__pesos[int_p][self.__tamano-1]=float_dist_p;
		self.__pesos[int_q][self.__tamano-1]=float_dist_q;
		self.__vidas[self.__tamano-1][int_p]=0;
		self.__vidas[self.__tamano-1][int_q]=0;
		self.__vidas[int_p][self.__tamano-1]=0;
		self.__vidas[int_q][self.__tamano-1]=0;
					
	def delete_neuron(self,int_i):
		if(int_i>=self.__tamano or int_i<0):# previous validations
			print "ERROR (GNG.delete_neuron): incorrect int_i value -> ", int_i;
			return;
		# deleting algorithm
		self.__neurons.pop(int_i);
		self.__pesos.pop(int_i);
		self.__vidas.pop(int_i);
		for i in range(0,self.__tamano-1):
			self.__pesos[i].pop(int_i);
			self.__vidas[i].pop(int_i);
		self.__tamano=self.__tamano-1;
		
	def connect_neurons(self,int_a,int_b):
		if(int_a>=self.__tamano or int_a<0):# previous validations
			print "ERROR (GNG.delete_neuron): incorrect int_a value -> ", int_a;
			return;
		if(int_b>=self.__tamano or int_b<0):
			print "ERROR (GNG.delete_neuron): incorrect int_b value -> ", int_b;
			return;
		# connecting algorithm
		float_dist=self.__neurons[int_a].get_distance(self.__neurons[int_b]);
		self.__pesos[int_a][int_b]=float_dist;
		self.__pesos[int_b][int_a]=float_dist;
		self.__vidas[int_a][int_b]=0;
		self.__vidas[int_b][int_a]=0;
		
	def disconnect_neurons(self,int_a,int_b):
		if(int_a>=self.__tamano or int_a<0):# previous validations
			print "ERROR (GNG.delete_neuron): incorrect int_a value -> ", int_a;
			return;
		if(int_b>=self.__tamano or int_b<0):
			print "ERROR (GNG.delete_neuron): incorrect int_b value -> ", int_b;
			return;
		# connecting algorithm
		self.__pesos[int_a][int_b]=-1.0;
		self.__pesos[int_b][int_a]=-1.0;
		self.__vidas[int_a][int_b]=-1;
		self.__vidas[int_b][int_a]=-1;
		
	def close_neurons(self,vec_pat):
		if(isinstance(vec_pat,list)==False):# previous validations
			print "ERROR (GNG.close_neurons): incorrect input value -> ", type(vec_pat);
			return;
		# algorithm
		float_ini=src.distance(self.__neurons[0].get_vector(),vec_pat);
		float_min=float_ini; 
		int_min_ind=0;
		for i in range(1,self.__tamano):
			float_temp=src.distance(self.__neurons[i].get_vector(),vec_pat);
			if(float_temp<float_min):
				float_min=float_temp;
				int_min_ind=i;
		float_min_second=9999999;	
		int_min_ind_second=-1;			
		int_ind_begin=0;
		if(int_min_ind==0):
			float_min_second=src.distance(self.__neurons[1].get_vector(),vec_pat);	
			int_min_ind_second=1;
			int_ind_begin=2;			
		else:
			float_min_second=float_ini;	
			int_min_ind_second=0;			
			int_ind_begin=1;
		for j in range(int_ind_begin,self.__tamano):
			if(j != int_min_ind):
				float_temp2=src.distance(self.__neurons[j].get_vector(),vec_pat);
				if(float_temp2<float_min_second):
					float_min_second=float_temp2;
					int_min_ind_second=j;				
		return (int_min_ind,float_min,int_min_ind_second,float_min_second);	
		
	def one_close_neuron(self,vec_pat):# previous validations
		if(isinstance(vec_pat,list)==False):
			print "ERROR (GNG.one_close_neuron): incorrect input value -> ", type(vec_pat);
			return;
		# algorithm	
		float_ini=src.distance(self.__neurons[0].get_vector(),vec_pat);
		float_min=float_ini; 
		int_min_ind=0;
		for i in range(1,self.__tamano):
			float_temp=src.distance(self.__neurons[i].get_vector(),vec_pat);
			if(float_temp<float_min):
				float_min=float_temp;
				int_min_ind=i;
		return int_min_ind;
					
	def max_error_neuron(self):
		float_max_error=self.__neurons[0].get_error();
		int_ind=0;
		for i in range(1,self.__tamano):
			float_cur_error=self.__neurons[i].get_error();
			if(float_cur_error>float_max_error):
				float_max_error=float_cur_error;
				int_ind=i;
		return int_ind;
		
	def max_error_by_neighborhood(self,int_neuron):
		if(isinstance(int_neuron,int)==False and int_cur_neuron<self.__tamano and int_cur_neuron>=0):# previous validations
			print "ERROR (GNG.increment_age_by_neighborhood): incorrect input value -> ", int_cur_neuron;
			return;
		# algorithm
		float_max_error=-1;
		int_ind=-1;
		bool_flag=False;
		for i in range(0,self.__tamano):
			if(self.__vidas[int_neuron][i]>-1):
				if(bool_flag==False):
					float_max_error=self.__neurons[i].get_error();
					int_ind=i;
					bool_flag=True;
				else:
					float_cur_error=self.__neurons[i].get_error();
					if(float_cur_error>float_max_error):
						float_max_error=float_cur_error;
						int_ind=i;
		return int_ind;
		
	def increment_age(self):
		for i in range(0,self.__tamano):
			for j in range(0,self.__tamano):
				if(self.__vidas[i][j]>-1):
					self.__vidas[i][j]=self.__vidas[i][j]+1;
					
	def increment_age_by_neighborhood(self,int_cur_neuron):
		if(isinstance(int_cur_neuron,int)==False and int_cur_neuron<self.__tamano and int_cur_neuron>=0):# previous validations
			print "ERROR (GNG.increment_age_by_neighborhood): incorrect input value -> ", int_cur_neuron;
			return;
		# algorithm
		for j in range(0,self.__tamano):
			if(self.__vidas[int_cur_neuron][j]>-1):
					self.__vidas[int_cur_neuron][j]=self.__vidas[int_cur_neuron][j]+1;
					self.__vidas[j][int_cur_neuron]=self.__vidas[j][int_cur_neuron]+1;
	
	def check_to_remove_connections(self): # this method just set to -1 all the old connections.
		for i in range(0,self.__tamano):
			for j in range(0,self.__tamano):
				if(self.__vidas[i][j]>self.__beta):
					self.__vidas[i][j]=-1;
					self.__pesos[i][j]=-1.0;
					
	def check_to_remove_neurons(self): # this method eliminates de neurons with no connections.
		vec_int_elim=[];
		int_elim=0;
		for i in range(0,self.__tamano):
			int_cont=0;
			for j in range(0,self.__tamano):
				if(self.__vidas[i][j]==-1):
					int_cont=int_cont+1;
			if(int_cont==self.__tamano):
				vec_int_elim.append(i);
				int_elim=int_elim+1;
		vec_int_elim.reverse();
		for k in range(0,int_elim):
			self.delete_neuron(vec_int_elim[k]);
			self.normalize_activations(vec_int_elim[k]);
	
	def normalize_activations(self,int_k):
		int_pat_size=len(self.__activaciones);
		for i in range(0,int_pat_size):
			if(self.__activaciones[i]>int_k):
				self.__activaciones[i]=self.__activaciones[i]-1;
			elif(self.__activaciones[i]==int_k):
				self.__activaciones[i]=-1;
	
	def regularize_activations(self):
		int_tam=len(self.__activaciones);
		for i in range(0,int_tam):
			if(self.__activaciones[i]==-1):
				int_neu_clo=self.one_close_neuron(self.__patterns[i]);
				self.__activaciones[i]=int_neu_clo;
				
	def update_neighborhood(self,vec_x,float_f,int_i):
		if(isinstance(vec_x,list)==False):# previous validations
			print "ERROR (GNG.update_neighborhood): incorrect input value -> ", type(vec_x);
			return;
		if(float_f<=0.0 and float_f>1.0):
			print "ERROR (GNG.update_neighborhood): incorrect float_f value -> ", float_f;
			return;
		if(int_i<0 and int_i>=self.__tamano):
			print "ERROR (GNG.update_neighborhood): incorrect int_i value -> ", int_i;
			return;	
		# algorithm
		for i in range(1,self.__tamano):
			if(self.__vidas[int_i][i]>-1):
				self.__neurons[i].update_vector(vec_x,float_f);
			
	def decrease_error_to_all(self,float_f):
		if(float_f<=0.0 and float_f>1.0):# previous validations
			print "ERROR (GNG.decrease_error_to all): incorrect float_f value -> ", float_f;
			return;
		# algorithm
		for i in range(0,self.__tamano):
			self.__neurons[i].decrease_error(float_f);
	
	def train(self):
		self.initialize();		
		int_tam_patterns=len(self.__patterns);	
		for i in range(0,self.__max_iterations):
			for j in range(0,int_tam_patterns):
				int_s,float_s,int_t,float_t=self.close_neurons(self.__patterns[j]);
				self.__activaciones[j]=int_s;
				if(self.are_connected(int_s,int_t)):
					self.set_live(int_s,int_t,0);
				else:
					self.connect_neurons(int_s,int_t);
				self.__neurons[int_s].update_error(self.__patterns[j]);
				self.__neurons[int_s].update_vector(self.__patterns[j],self.__epsilon_s);
				self.update_neighborhood(self.__patterns[j],self.__epsilon_t,int_s);
				self.increment_age_by_neighborhood(int_s);
				self.check_to_remove_connections();
				self.check_to_remove_neurons();
				if(((i*int_tam_patterns)+j)%self.__lambda==0):
					int_q=self.max_error_neuron();
					int_f=self.max_error_by_neighborhood(int_q);
					neuron_temp=neu.Neuron([],0);
					neuron_temp.become_middle(self.__neurons[int_q],self.__neurons[int_f],self.__alpha);
					self.add_neuron(neuron_temp,int_q,int_f);
				self.decrease_error_to_all(self.__delta);			
		self.regularize_activations();
				
	def index_vertex_influence(self):
		float_sum=0.0;
		int_pattern_size=len(self.__patterns);
		for i in range(0,int_pattern_size):
			float_dist_temp=src.distance(self.__patterns[i],self.__neurons[self.__activaciones[i]].get_vector());
			float_sum=float_sum+float_dist_temp;
		float_resp=float_sum/int_pattern_size;
		return float_resp;
		
	def index_vertex_distribution(self):
		float_sum=0.0;
		for i in range(0,self.__tamano):
			float_max_cur_dist=max(self.__pesos[i]);
			float_sum=float_sum+float_max_cur_dist;
		float_resp=float_sum/self.__tamano;
		return float_resp;
		
	def index_graph_mapping	(self):
		float_Ivi=self.index_vertex_influence();
		float_Ivd=self.index_vertex_distribution();
		float_Igm=float_Ivi/float_Ivd;
		return float_Igm;
			
	def train_wsc(self,float_error,int_factor):
		self.initialize();		
		int_tam_patterns=len(self.__patterns);	
		float_index_gm_ant=999999;
		int_cont=0;
		for i in range(0,self.__max_iterations):
			for j in range(0,int_tam_patterns):
				int_s,float_s,int_t,float_t=self.close_neurons(self.__patterns[j]);
				self.__activaciones[j]=int_s;
				if(self.are_connected(int_s,int_t)):
					self.set_live(int_s,int_t,0);
				else:
					self.connect_neurons(int_s,int_t);
				self.__neurons[int_s].update_error(self.__patterns[j]);
				self.__neurons[int_s].update_vector(self.__patterns[j],self.__epsilon_s);
				self.update_neighborhood(self.__patterns[j],self.__epsilon_t,int_s);
				self.increment_age_by_neighborhood(int_s);
				self.check_to_remove_connections();
				self.check_to_remove_neurons();
				if(((i*int_tam_patterns)+j)%self.__lambda==0):
					int_q=self.max_error_neuron();
					int_f=self.max_error_by_neighborhood(int_q);
					neuron_temp=neu.Neuron([],0);
					neuron_temp.become_middle(self.__neurons[int_q],self.__neurons[int_f],self.__alpha);
					self.add_neuron(neuron_temp,int_q,int_f);
				self.decrease_error_to_all(self.__delta);
			self.regularize_activations();
			float_index_gm=self.index_graph_mapping();
			float_cur_error=abs(float_index_gm-float_index_gm_ant);
			#print float_cur_error, int_cont;
			if(float_cur_error<=abs(float_error)):
				int_cont=int_cont+1;
				if(int_cont>=int_factor):
					#print "Graph Mapping Index ------> ",float_index_gm;
					self.gm_connect_graph();
					return True;
			else:
				int_cont=0;
			float_index_gm_ant=float_index_gm;
		#print "Graph Mapping Index not converge ------> ",float_index_gm_ant;
		self.gm_connect_graph();
		return False;
	      
	def train_wsc_fixed(self,float_error,int_factor,int_max_tam):
		self.initialize();		
		int_tam_patterns=len(self.__patterns);	
		float_index_gm_ant=999999;
		int_cont=0;
		for i in range(0,self.__max_iterations):
			for j in range(0,int_tam_patterns):
				int_s,float_s,int_t,float_t=self.close_neurons(self.__patterns[j]);
				self.__activaciones[j]=int_s;
				if(self.are_connected(int_s,int_t)):
					self.set_live(int_s,int_t,0);
				else:
					self.connect_neurons(int_s,int_t);
				self.__neurons[int_s].update_error(self.__patterns[j]);
				self.__neurons[int_s].update_vector(self.__patterns[j],self.__epsilon_s);
				self.update_neighborhood(self.__patterns[j],self.__epsilon_t,int_s);
				self.increment_age_by_neighborhood(int_s);
				self.check_to_remove_connections();
				self.check_to_remove_neurons();
				if(((i*int_tam_patterns)+j)%self.__lambda==0 and self.__tamano<int_max_tam):
					int_q=self.max_error_neuron();
					int_f=self.max_error_by_neighborhood(int_q);
					neuron_temp=neu.Neuron([],0);
					neuron_temp.become_middle(self.__neurons[int_q],self.__neurons[int_f],self.__alpha);
					self.add_neuron(neuron_temp,int_q,int_f);
				self.decrease_error_to_all(self.__delta);
			self.regularize_activations();
			float_index_gm=self.index_graph_mapping();
			float_cur_error=abs(float_index_gm-float_index_gm_ant);
			#print float_cur_error, int_cont;
			if(float_cur_error<=abs(float_error)):
				int_cont=int_cont+1;
				if(int_cont>=int_factor):
					#print "Graph Mapping Index ------> ",float_index_gm;
					self.gm_connect_graph();
					return True;
			else:
				int_cont=0;
			float_index_gm_ant=float_index_gm;
		#print "Graph Mapping Index not converge ------> ",float_index_gm_ant;
		self.gm_connect_graph();
		return False;
	
##--------------------------------------------------------------------------------------##
##	Graph Manipulation Functions														##
##--------------------------------------------------------------------------------------##

	def gm_get_clusters(self):
		vec_clus=[-1]*self.__tamano;
		int_num=0;
		for i in range(0,self.__tamano):
			if(vec_clus[i]==-1):
				int_num=int_num+1;
				self.gm_dfs_exploration(i,int_num,vec_clus);		
		return vec_clus;

	def gm_dfs_exploration(self,int_ini,int_clus,vec_clus):
		if(int_ini>=self.__tamano or int_ini<0):# Validating input
			print "ERROR (GNG.gm_dfs_exploration): incorrect int_ini value -> ", int_ini;
			return;
		# Algorithm
		vec_vis=[0]*self.__tamano;
		Stack=[];
		Stack.append(int_ini);
		vec_clus[int_ini]=int_clus;
		vec_vis[int_ini]=1;
		while(len(Stack)!=0):
			int_node_cur=Stack.pop();
			for i in range(0,self.__tamano):
				if(self.__pesos[int_node_cur][i]!=-1): # check if a connections exists
					if(vec_vis[i]==0):
						Stack.append(i);
						vec_clus[i]=int_clus;
						vec_vis[i]=1;

	def gm_closest_point_out_cluster(self,vec_clus,int_point):
		if(isinstance(vec_clus,list)==False):# Validating input
			print "ERROR (GNG.gm_closest_point_out_cluster): incorrect vec_clus value -> ", type(vec_clus);
			return;
		if(int_point>=self.__tamano or int_point<0):
			print "ERROR (GNG.gm_closest_point_out_cluster): incorrect int_point value -> ", int_point;
			return;
		# Algorithm
		bool_flag=True;
		float_dst_min=9999999;
		int_ind_min=-1;
		for i in range(0,self.__tamano):
			if(i!=int_point and vec_clus[i]!=vec_clus[int_point]):
				if(bool_flag):
					float_dst_min=src.distance(self.__neurons[int_point].get_vector(),self.__neurons[i].get_vector());
					int_ind_min=i;
					bool_flag=False;
				else:
					float_dst_cur=src.distance(self.__neurons[int_point].get_vector(),self.__neurons[i].get_vector());
					if(float_dst_cur<float_dst_min):
						float_dst_min=float_dst_cur;
						int_ind_min=i;
		if(int_ind_min!=-1):
			return (int_ind_min,float_dst_min,vec_clus[int_ind_min]);
		else:
			print "WARNING (GNG.gm_closest_point_out_cluster): one cluster"
			return (-1,0,1);

	def gm_closest_clusters(self,vec_clus):
		if(isinstance(vec_clus,list)==False):# Validating input
			print "ERROR (GNG.gm_closest_clusters): incorrect vec_clus value -> ", type(vec_clus);
			return;
		# Algorithm
		vec_closest_point=[-1]*self.__tamano;
		vec_closest_dst=[-1]*self.__tamano;
		vec_closest_cluster=[-1]*self.__tamano;
		for i in range(0,self.__tamano):
			int_ind,float_dst,int_clus=self.gm_closest_point_out_cluster(vec_clus,i);
			if(int_ind!=-1):
				vec_closest_point[i]=int_ind;
				vec_closest_dst[i]=float_dst;
				vec_closest_cluster[i]=int_clus;
			else:
				print "WARNING (GNG.gm_closest_clusters): one cluster";
				return;
		return (vec_closest_point,vec_closest_dst,vec_closest_cluster);
			
	def gm_join_clusters(self,vec_clusters,vec_closest_p,vec_dst,vec_closest_c):
		if(isinstance(vec_clusters,list)==False):# Validating input
			print "ERROR (GNG.gm_join_clusters): incorrect vec_clusters value -> ", type(vec_clusters);
			return;
		if(isinstance(vec_closest_p,list)==False):
			print "ERROR (GNG.gm_join_clusters): incorrect vec_closest_p value -> ", type(vec_closest_p);
			return;
		if(isinstance(vec_dst,list)==False):
			print "ERROR (GNG.gm_join_clusters): incorrect vec_dst value -> ", type(vec_dst);
			return;
		if(isinstance(vec_clusters,list)==False):
			print "ERROR (GNG.gm_join_clusters): incorrect vec_closest_c value -> ", type(vec_closest_c);
			return;
		# Algorithm
		int_total_clusters=len(set(vec_clusters));
		if(int_total_clusters==1):
			print "WARNING (GNG.gm_join_clusters): one cluster";
			return;
		int_ind_min_dst=vec_dst.index(min(vec_dst));
		int_ind_fn=int_ind_min_dst;
		int_ind_sn=vec_closest_p[int_ind_min_dst];
		self.connect_neurons(int_ind_fn,int_ind_sn);
		int_clus_fn=vec_clusters[int_ind_fn];
		int_clus_sn=vec_clusters[int_ind_sn];
		for i in range(0,self.__tamano):
			if(vec_clusters[i]==int_clus_sn):
				vec_clusters[i]=int_clus_fn;
				
	def gm_connect_graph(self):
		vec_clusters=self.gm_get_clusters();
		if(len(set(vec_clusters))>1):
			vec_cp,vec_cd,vec_cc=self.gm_closest_clusters(vec_clusters);
			self.gm_join_clusters(vec_clusters,vec_cp,vec_cd,vec_cc);
			while(len(set(vec_clusters))>1):
				vec_cp,vec_cd,vec_cc=self.gm_closest_clusters(vec_clusters);
				self.gm_join_clusters(vec_clusters,vec_cp,vec_cd,vec_cc);


#-------------------------------------------------------------------------------------
#	Semi-Supervised Functions
#-------------------------------------------------------------------------------------

	def ssl_most_common(self,vec_lab,int_q):
		if(isinstance(vec_lab,list)==False):
			print "ERROR (GNG.ssl_most_common): input parameter is not a list -> ", type(vec_lab);
			return;
		int_tam=len(vec_lab);
		dict_col={};
		for i in range(0,int_tam):
			if(dict_col.has_key(vec_lab[i])):
				dict_col[vec_lab[i]]=dict_col[vec_lab[i]]+1;
			else:
				dict_col[vec_lab[i]]=1;
		vec_resp=[];
		for i in range(0,int_q):
			if(len(dict_col)>0):
				vec_keys=dict_col.keys();
				vec_values=dict_col.values();
				int_ind=vec_values.index(max(vec_values));
				val_max_key=vec_keys[int_ind];
				val_max_value=vec_values[int_ind];
				vec_resp.append((val_max_key,val_max_value));
				del dict_col[vec_keys[int_ind]];
		return vec_resp;

	def ssl_most_common_label(self,vec_lab,int_default=-99):
		if(len(vec_lab)==0):
			return int_default;
		#counter_data = Counter(vec_lab)
		#val_struct=counter_data.most_common(2);
		val_struct=self.ssl_most_common(vec_lab,2);
		if(len(val_struct)==1):
			return val_struct[0][0];
		elif(len(val_struct)==2):
			int_lab_1=val_struct[0][0];
			int_lab_2=val_struct[1][0];
			int_cant_1=val_struct[0][1];
			int_cant_2=val_struct[1][1];
			if(int_cant_1==int_cant_2):
				float_random=random.random();
				if(float_random>0.5):
					return int_lab_1;
				else:
					return int_lab_2;
			else:
				return int_lab_1;

	def ssl_get_all(self,vec_label):
		arr_gr=src.copy_list(self.__pesos);# Taking care of similarities
		for i in range(0,self.__tamano):
			for j in range(0,self.__tamano):
				if (arr_gr[i][j]==-1):
					arr_gr[i][j]=0;
				else:
					arr_gr[i][j]=src.weightFunc(arr_gr[i][j]);
		# Taking care of activations
		arr_irregular=[];
		vec_eti=[];
		for i in range(0,self.__tamano):
			arr_irregular.append([]);
			vec_eti.append(-99);
		int_tam_label=len(vec_label);
		for i in range(0,int_tam_label):
			arr_irregular[self.__activaciones[i]].append(vec_label[i]);
		vec_ind_lab=[];
		for i in range(0,self.__tamano):
			vec_eti[i]=self.ssl_most_common_label(arr_irregular[i]);
			if(vec_eti[i]!=-99):
				vec_ind_lab.append(i);
		vec_label_ready=src.reOrganizeVector(vec_eti,vec_ind_lab);
		arr_graph_return=src.reOrganizeMatrix(arr_gr,vec_ind_lab);
		vec_label_return=vec_label_ready[0:len(vec_ind_lab)];
		return (arr_graph_return,vec_label_return,vec_ind_lab); # grafo, etiquetas, indices mapeados.
		
	def ssl_wrap_regularizer(self,regularizer_func,*vec_full_data):
		if (len(vec_full_data)>=2):
			int_tam_gr=len(vec_full_data[0]);
			int_tam_lab=len(vec_full_data[1]);
			if(int_tam_gr<=int_tam_lab):
				return vec_full_data[1];
			else:
				return regularizer_func(*vec_full_data);
				
	def ssl_recover_all(self,vec_class,vec_ind):
		vec_clases=[];# Validating Input
		if(isinstance(vec_class,list)):
			vec_clases=vec_class;
		elif(isinstance(vec_class,np.matrix)):
			vec_clases=vec_class.flatten().tolist()[0];
		elif(isinstance(vec_class,np.ndarray)):
			vec_clases=vec_class.tolist();
		else:
			print "ERROR (GNG.ssl_recover_all): input parameter is not a list -> ", type(vec_class);
			return;
		# Algorithm
		vec_ind_todos=range(0,self.__tamano);
		vec_ind_listo=src.reOrganizeVector(vec_ind_todos,vec_ind);
		int_tam_data=len(self.__patterns);
		vec_mapped_labels=[];
		for i in range(0,int_tam_data):
			int_cur_ind=vec_ind_listo.index(self.__activaciones[i]);
			int_cur_label=vec_clases[int_cur_ind];
			vec_mapped_labels.append(int_cur_label);
		return vec_mapped_labels;


#-----------------------------------------------------------------------------------
#	Graphic Functions				
#-----------------------------------------------------------------------------------

	def draw_data(self,float_res):
		top = Tkinter.Tk()
		C = Tkinter.Canvas(top, bg="white", height=600, width=600)
		int_tam=len(self.__patterns);
		for i in range(0,int_tam):
			coord = (self.__patterns[i][0])*float_res,(self.__patterns[i][1])*float_res,(self.__patterns[i][0]+0.1)*float_res,(self.__patterns[i][1]+0.1)*float_res;
			C.create_oval(coord, fill="red",width=1);
		C.pack()
		top.mainloop()

	def draw_neurons(self,float_res):
		top = Tkinter.Tk()
		C = Tkinter.Canvas(top, bg="white", height=600, width=600)
		for i in range(0,self.__tamano):
			coord = (self.__neurons[i].get_vector()[0])*float_res,(self.__neurons[i].get_vector()[1])*float_res,(self.__neurons[i].get_vector()[0]+0.1)*float_res,(self.__neurons[i].get_vector()[1]+0.1)*float_res;
			C.create_oval(coord, fill="blue",width=1);
			for j in range(0,self.__tamano):
				if(self.__vidas[i][j]>-1):
					C.create_line((self.__neurons[i].get_vector()[0])*float_res,(self.__neurons[i].get_vector()[1])*float_res,(self.__neurons[j].get_vector()[0])*float_res,(self.__neurons[j].get_vector()[1])*float_res,fill="blue");
		C.pack()
		top.mainloop()
		
	def draw_all(self,float_res):
		top = Tkinter.Tk()
		C = Tkinter.Canvas(top, bg="white", height=600, width=600)
		int_tam=len(self.__patterns);
		for i in range(0,int_tam):
			coord = (self.__patterns[i][0])*float_res,(self.__patterns[i][1])*float_res,(self.__patterns[i][0]+0.2)*float_res,(self.__patterns[i][1]+0.2)*float_res;
			C.create_oval(coord, fill="red",width=1);
		for i in range(0,self.__tamano):
			coord = (self.__neurons[i].get_vector()[0])*float_res,(self.__neurons[i].get_vector()[1])*float_res,(self.__neurons[i].get_vector()[0]+0.1)*float_res,(self.__neurons[i].get_vector()[1]+0.1)*float_res;
			C.create_oval(coord, fill="blue",width=1);
			for j in range(0,self.__tamano):
				if(self.__vidas[i][j]>-1):
					C.create_line((self.__neurons[i].get_vector()[0])*float_res,(self.__neurons[i].get_vector()[1])*float_res,(self.__neurons[j].get_vector()[0])*float_res,(self.__neurons[j].get_vector()[1])*float_res,fill="blue");
		#C.pack()
		top.mainloop()
		
	def draw_current(self,float_res,C):
		C.delete("all");
		int_tam=len(self.__patterns);
		for i in range(0,int_tam):
			coord = (self.__patterns[i][0])*float_res,(self.__patterns[i][1])*float_res,(self.__patterns[i][0]+0.2)*float_res,(self.__patterns[i][1]+0.2)*float_res;
			C.create_oval(coord, fill="red",width=1);
		for i in range(0,self.__tamano):
			coord = (self.__neurons[i].get_vector()[0])*float_res,(self.__neurons[i].get_vector()[1])*float_res,(self.__neurons[i].get_vector()[0]+0.1)*float_res,(self.__neurons[i].get_vector()[1]+0.1)*float_res;
			C.create_oval(coord, fill="blue",width=1);
			for j in range(0,self.__tamano):
				if(self.__vidas[i][j]>-1):
					C.create_line((self.__neurons[i].get_vector()[0])*float_res,(self.__neurons[i].get_vector()[1])*float_res,(self.__neurons[j].get_vector()[0])*float_res,(self.__neurons[j].get_vector()[1])*float_res,fill="blue");
		C.after(10);
		C.update();
		
	def draw_class_data(self,float_res,C,vec_clas):
		C.delete("all");
		int_tam=len(self.__patterns);
		int_tam_clas=len(vec_clas);
		for i in range(0,int_tam):
			coord = (self.__patterns[i][0])*float_res,(self.__patterns[i][1])*float_res,(self.__patterns[i][0]+0.2)*float_res,(self.__patterns[i][1]+0.2)*float_res;
			if(i<int_tam_clas):
				if(vec_clas[i]==1):
					C.create_oval(coord, fill="green",width=1);
				else:
					C.create_oval(coord, fill="yellow",width=1);
			else:
				C.create_oval(coord, fill="red",width=1);
		for i in range(0,self.__tamano):
			coord = (self.__neurons[i].get_vector()[0])*float_res,(self.__neurons[i].get_vector()[1])*float_res,(self.__neurons[i].get_vector()[0]+0.1)*float_res,(self.__neurons[i].get_vector()[1]+0.1)*float_res;
			C.create_oval(coord, fill="blue",width=1);
			for j in range(0,self.__tamano):
				if(self.__vidas[i][j]>-1):
					C.create_line((self.__neurons[i].get_vector()[0])*float_res,(self.__neurons[i].get_vector()[1])*float_res,(self.__neurons[j].get_vector()[0])*float_res,(self.__neurons[j].get_vector()[1])*float_res,fill="blue");
		C.after(10);
		C.update();
		
	def draw_class_neuron(self,float_res,C,vec_clas_neu):
		C.delete("all");
		int_tam=len(self.__patterns);
		int_tam_clas=len(vec_clas_neu);
		for i in range(0,int_tam):
			coord = (self.__patterns[i][0])*float_res,(self.__patterns[i][1])*float_res,(self.__patterns[i][0]+0.2)*float_res,(self.__patterns[i][1]+0.2)*float_res;
			C.create_oval(coord, fill="red",width=1);
		for i in range(0,self.__tamano):
			coord = (self.__neurons[i].get_vector()[0])*float_res,(self.__neurons[i].get_vector()[1])*float_res,(self.__neurons[i].get_vector()[0]+0.1)*float_res,(self.__neurons[i].get_vector()[1]+0.1)*float_res;
			if(i<int_tam_clas):
				if(vec_clas_neu[i]==1):
					C.create_oval(coord, fill="green",width=1);
				else:
					C.create_oval(coord, fill="yellow",width=1);
			else:
				C.create_oval(coord, fill="blue",width=1);
			for j in range(0,self.__tamano):
				if(self.__vidas[i][j]>-1):
					C.create_line((self.__neurons[i].get_vector()[0])*float_res,(self.__neurons[i].get_vector()[1])*float_res,(self.__neurons[j].get_vector()[0])*float_res,(self.__neurons[j].get_vector()[1])*float_res,fill="blue");
		C.after(10);
		C.update();
	
	def draw_class_all(self,float_res,C,vec_clas,vec_clas_neu):
		C.delete("all");
		int_tam=len(self.__patterns);
		int_tam_clas=len(vec_clas);
		int_tam_clas_neu=len(vec_clas_neu);
		for i in range(0,int_tam):
			coord = (self.__patterns[i][0])*float_res,(self.__patterns[i][1])*float_res,(self.__patterns[i][0]+0.2)*float_res,(self.__patterns[i][1]+0.2)*float_res;
			if(i<int_tam_clas):
				if(vec_clas[i]==1):
					C.create_oval(coord, fill="green",width=1);
				else:
					C.create_oval(coord, fill="yellow",width=1);
			else:
				C.create_oval(coord, fill="red",width=1);
		for i in range(0,self.__tamano):
			coord = (self.__neurons[i].get_vector()[0])*float_res,(self.__neurons[i].get_vector()[1])*float_res,(self.__neurons[i].get_vector()[0]+0.1)*float_res,(self.__neurons[i].get_vector()[1]+0.1)*float_res;
			if(i<int_tam_clas_neu):
				if(vec_clas_neu[i]==1):
					C.create_oval(coord, fill="green",width=1);
				else:
					C.create_oval(coord, fill="yellow",width=1);
			else:
				C.create_oval(coord, fill="blue",width=1);
			for j in range(0,self.__tamano):
				if(self.__vidas[i][j]>-1):
					C.create_line((self.__neurons[i].get_vector()[0])*float_res,(self.__neurons[i].get_vector()[1])*float_res,(self.__neurons[j].get_vector()[0])*float_res,(self.__neurons[j].get_vector()[1])*float_res,fill="blue");
		C.after(10);
		C.update();
						
	def draw_training(self,float_error,int_factor,float_res):
		top = Tkinter.Tk()
		C = Tkinter.Canvas(top, bg="white", height=600, width=600)
		C.pack();
		self.initialize();		
		int_tam_patterns=len(self.__patterns);	
		float_index_gm_ant=999999;
		int_cont=0;
		for i in range(0,self.__max_iterations):
			for j in range(0,int_tam_patterns):
				int_s,float_s,int_t,float_t=self.close_neurons(self.__patterns[j]);
				self.__activaciones[j]=int_s;
				if(self.are_connected(int_s,int_t)):
					self.set_live(int_s,int_t,0);
				else:
					self.connect_neurons(int_s,int_t);
				self.__neurons[int_s].update_error(self.__patterns[j]);
				self.__neurons[int_s].update_vector(self.__patterns[j],self.__epsilon_s);
				self.update_neighborhood(self.__patterns[j],self.__epsilon_t,int_s);
				self.increment_age_by_neighborhood(int_s);
				self.check_to_remove_connections();
				self.check_to_remove_neurons();
				if(((i*int_tam_patterns)+j)%self.__lambda==0):
					int_q=self.max_error_neuron();
					int_f=self.max_error_by_neighborhood(int_q);
					neuron_temp=neu.Neuron([],0);
					neuron_temp.become_middle(self.__neurons[int_q],self.__neurons[int_f],self.__alpha);
					self.add_neuron(neuron_temp,int_q,int_f);
				self.decrease_error_to_all(self.__delta);
				# ........................................................................ #
				C.delete("all");
				int_tam=len(self.__patterns);
				for i in range(0,int_tam):
					coord = (self.__patterns[i][0])*float_res,(self.__patterns[i][1])*float_res,(self.__patterns[i][0]+0.2)*float_res,(self.__patterns[i][1]+0.2)*float_res;
					C.create_oval(coord, fill="red",width=1);
				for i in range(0,self.__tamano):
					coord = (self.__neurons[i].get_vector()[0])*float_res,(self.__neurons[i].get_vector()[1])*float_res,(self.__neurons[i].get_vector()[0]+0.1)*float_res,(self.__neurons[i].get_vector()[1]+0.1)*float_res;
					C.create_oval(coord, fill="blue",width=1);
					for j in range(0,self.__tamano):
						if(self.__vidas[i][j]>-1):
							C.create_line((self.__neurons[i].get_vector()[0])*float_res,(self.__neurons[i].get_vector()[1])*float_res,(self.__neurons[j].get_vector()[0])*float_res,(self.__neurons[j].get_vector()[1])*float_res,fill="blue");
				C.after(10)
    			C.update()				
				# ........................................................................ #									
			self.regularize_activations();
			float_index_gm=self.index_graph_mapping();
			float_cur_error=abs(float_index_gm-float_index_gm_ant);
			#print float_cur_error, int_cont;
			if(float_cur_error<=abs(float_error)):
				int_cont=int_cont+1;
				if(int_cont>=int_factor):
					self.gm_connect_graph();
					raw_input("--Aprete una tecla para conectar el grafo--");
					self.draw_current(float_res,C);
					return True;
			else:
				int_cont=0;
			float_index_gm_ant=float_index_gm;
		top.mainloop();
		self.gm_connect_graph();
		raw_input("<<<<<<<<<<<<No Converge>>>>>>>>>>>>>>>");
		self.draw_current(float_res,C);
		return False;
	      
	def draw_training_fixed(self,float_error,int_factor,int_max_tam,float_res):
		top = Tkinter.Tk()
		C = Tkinter.Canvas(top, bg="white", height=600, width=600)
		C.pack();
		self.initialize();		
		int_tam_patterns=len(self.__patterns);	
		float_index_gm_ant=999999;
		int_cont=0;
		for i in range(0,self.__max_iterations):
			for j in range(0,int_tam_patterns):
				int_s,float_s,int_t,float_t=self.close_neurons(self.__patterns[j]);
				self.__activaciones[j]=int_s;
				if(self.are_connected(int_s,int_t)):
					self.set_live(int_s,int_t,0);
				else:
					self.connect_neurons(int_s,int_t);
				self.__neurons[int_s].update_error(self.__patterns[j]);
				self.__neurons[int_s].update_vector(self.__patterns[j],self.__epsilon_s);
				self.update_neighborhood(self.__patterns[j],self.__epsilon_t,int_s);
				self.increment_age_by_neighborhood(int_s);
				self.check_to_remove_connections();
				self.check_to_remove_neurons();
				if(((i*int_tam_patterns)+j)%self.__lambda==0 and self.__tamano<int_max_tam):
					int_q=self.max_error_neuron();
					int_f=self.max_error_by_neighborhood(int_q);
					neuron_temp=neu.Neuron([],0);
					neuron_temp.become_middle(self.__neurons[int_q],self.__neurons[int_f],self.__alpha);
					self.add_neuron(neuron_temp,int_q,int_f);
				self.decrease_error_to_all(self.__delta);
				# ........................................................................ #
				C.delete("all");
				int_tam=len(self.__patterns);
				for i in range(0,int_tam):
					coord = (self.__patterns[i][0])*float_res,(self.__patterns[i][1])*float_res,(self.__patterns[i][0]+0.2)*float_res,(self.__patterns[i][1]+0.2)*float_res;
					C.create_oval(coord, fill="red",width=1);
				for i in range(0,self.__tamano):
					coord = (self.__neurons[i].get_vector()[0])*float_res,(self.__neurons[i].get_vector()[1])*float_res,(self.__neurons[i].get_vector()[0]+0.1)*float_res,(self.__neurons[i].get_vector()[1]+0.1)*float_res;
					C.create_oval(coord, fill="blue",width=1);
					for j in range(0,self.__tamano):
						if(self.__vidas[i][j]>-1):
							C.create_line((self.__neurons[i].get_vector()[0])*float_res,(self.__neurons[i].get_vector()[1])*float_res,(self.__neurons[j].get_vector()[0])*float_res,(self.__neurons[j].get_vector()[1])*float_res,fill="blue");
				C.after(10);
				C.update();
				# ........................................................................ #									
			self.regularize_activations();
			float_index_gm=self.index_graph_mapping();
			float_cur_error=abs(float_index_gm-float_index_gm_ant);
			#print float_cur_error, int_cont;
			if(float_cur_error<=abs(float_error)):
				int_cont=int_cont+1;
				if(int_cont>=int_factor):
					self.gm_connect_graph();
					raw_input("--Aprete una tecla para conectar el grafo--");
					self.draw_current(float_res,C);
					return (C,True);
			else:
				int_cont=0;
			float_index_gm_ant=float_index_gm;
		top.mainloop();
		self.gm_connect_graph();
		raw_input("<<<<<<<<<<<<No Converge>>>>>>>>>>>>>>>");
		self.draw_current(float_res,C);
		return (C,False);
	      
#-------------------------------------------------------------------
#	PLOT FUNCTIONS
#-------------------------------------------------------------------

	def plot_data(self):
		pass;




#-----------------------------------------------------------------------
#	      END
#-----------------------------------------------------------------------




