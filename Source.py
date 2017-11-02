#!/usr/bin/python
import math
import random
import collections
import numpy as np
import time
from scipy import linalg as scipylinalg
from scipy import spatial

#--------------------------------------------------------------------#

# ESTANDARES:

# int_ integer
# float_ float
# vec_ python list (vector)
# tup_ tuple
# arr_ python list of python list (array, matriz)
# mat_ scipy array
# matx_ scipy matrix
# str_ string
# val_ structura, valor no identificado

# NOTAS:

# todos los reguladores reciben como labels solo los de los elementos etiquetados, es decir el vector de labels no puede tener entradas nulas que no representen ninguna etiqueta (ya sea positiva o negativa).

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
#                         General Functions                          #
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

def randomfloat(float_n1,float_n2): #float_n1<=numfloat<=float_n2
	int_intp1=random.randrange(np.floor(float_n1),np.floor(float_n2)+1,1);
	int_floatp2=random.randrange(0,101,1);
	float_floatp2=float(int_floatp2)/100;
	float_final=float(int_intp1)+float_floatp2;
	if(float_final>float_n2):
		float_final=float_n2-(float_final-float_n2);
	if(float_final<float_n1):
		float_final=float_n1+(float_n1-float_final);
	return float_final;

def distance(v1,v2,dt='e'):
	ss=0;
	if len(v1)!=len(v2):
		return -1;
	if dt=='e':
		i=0;
		for d1 in v1:
			ss=ss+pow((d1-v2[i]),2);
			i=i+1;
		ss=math.sqrt(ss);
	return ss;

def weightFunc(dist,tpe='e',hp1=0.5,hp2=1): # ---------------------------------------------> Tener en cuenta el parametro hp1 antes de hacer algo.
	if tpe=='e':
		return math.exp( (-pow(dist,2)) / (2*pow(hp1,2)) )
		
def makeSymetric(M):
	rows=len(M)
	for i in range(0,rows):
		cols=len(M[i]);
		for j in range(0,cols):
			if M[i][j]>M[j][i]:
				M[j][i]=M[i][j];
			else:
				M[i][j]=M[j][i];
	return M;
	
def affinityMatrix(data,sigma=20,dt='e',wt='e'):
	graph=[];
	for vec1 in data:
		edges=[];
		for vec2 in data:
			dst=distance(vec1,vec2,dt);
			wij=0;
			wij=weightFunc(dst,wt,sigma);
			edges.append(wij);
		graph.append(edges);
	return graph;
	
def buildFullGraph(data,dt='e',wt='e'):
	graph=[];
	for vec1 in data:
		edges=[];
		for vec2 in data:
			#dst=distance(vec1,vec2,dt);
			dst=np.linalg.norm(np.array(vec1)-np.array(vec2));
			wij=0;
			if dst>0:
				wij=weightFunc(dst,wt);
			edges.append(wij);
		graph.append(edges);
	return graph;

def buildEGraph(data,epsilon,dt='e',wt='e'):
	graph=[];
	for vec1 in data:
		edges=[];
		for vec2 in data:
			dst=distance(vec1,vec2,dt);
			wij=0;
			if(dst>0 and dst<=epsilon):
				wij=weightFunc(dst,wt);
			edges.append(wij);
		graph.append(edges);
	return graph;

def buildKGraph(data,K,dt='e',wt='e'):
	graph=[];
	for vec1 in data:
		edges=[];
		for vec2 in data:
			dst=distance(vec1,vec2,dt);
			wij=0;
			if dst>0:
				wij=weightFunc(dst,wt);
			edges.append(wij);
		tam=len(edges)
		edges2=[0]*tam;
		for i in range(0,K):
			maxim=max(edges);
			indice=edges.index(maxim);
			edges2[indice]=maxim;
			edges[indice]=-1;
		graph.append(edges2);
	graph=makeSymetric(graph);
	return graph;
	
def getDMatrix(graph):
	D=[];
	tam=len(graph);
	for i in range(0,tam):
		vec=[0]*tam;
		vec[i]=sum(graph[i]);
		D.append(vec);
	return D;
	
def getLaplaciano(graph):
	L=[];
	D=getDMatrix(graph);
	tam=len(graph);
	for i in range(0,tam):
		vec=[];
		for j in range(0,tam):
			val=D[i][j]-graph[i][j];
			vec.append(val);
		L.append(vec);
	return L;
	
def normalizeLaplaciano(arr_L,arr_graph):
	arr_D=getDMatrix(arr_graph);
	mat_D=np.array(arr_D);
	mat_L=np.array(arr_L);
	mat_sqrt_D=np.sqrt(mat_D);
	mat_inv_sqrt_D=np.linalg.pinv(mat_sqrt_D);
	#mat_inv_D=np.linalg.inv(mat_D);
	mat_norm_lap=np.dot(np.dot(mat_inv_sqrt_D,mat_L),mat_inv_sqrt_D);
	#print mat_norm_lap.tolist();
	return mat_norm_lap;
	
def configureLabels(vec_labels,int_fv,int_sv):
	int_tam=len(vec_labels);
	for i in range(0,int_tam):
		if(vec_labels[i]==int_fv):
			vec_labels[i]=int_sv;
	
def reOrganizeVector(V,yl):
	#yl.sort();
	tam=len(V);
	tam2=len(yl);
	RV=[0]*tam;
	flag=[0]*tam;
	for val in yl:
		flag[val]=1;
	h=0;
	i=0;
	j=tam2;
	for fl in flag:
		if fl==1:
			RV[i]=V[h];
			i=i+1;
		else:
			if fl==0:
				RV[j]=V[h];
				j=j+1;
		h=h+1;
	return RV;
	
def reOrganizeVectorRobust(vec_V,vec_ind):
	int_tam1=len(vec_V);
	int_tam2=len(vec_ind);
	if (int_tam1<int_tam2):
		raise "ERROR (Source.reOrganizeVectorRobust): wrong vector sizes", int_tam1, int_tam2;
	vec_RV=[0]*int_tam1;
	vec_flag=[0]*int_tam1;
	for i in range(0,int_tam2):
		vec_RV[i]=vec_V[vec_ind[i]];
		vec_flag[vec_ind[i]]=1;
	int_i=int_tam2;
	for i in range(0,int_tam1):
		if(vec_flag[i]==0):
			vec_RV[int_i]=vec_V[i];
			int_i=int_i+1;
	return vec_RV;
 
def reOrganizeMatrix(M,yl):
	tam=len(M);
	nyl=range(0,tam);
	nyl=reOrganizeVector(nyl,yl);
	MR=[];
	for i in range(0,tam):
		vec=[]
		for j in range(0,tam):
			vec.append(M[nyl[i]][nyl[j]]);
		MR.append(vec);
	return MR;

def indtosub(rows,n):
	n=n+1;
	n=float(n);
	j=np.ceil(n/rows);
	i=n-((j*rows)-rows);
	i=i-1;
	j=j-1;
	i=int(i);
	j=int(j);
	return (i,j);

def percentil(X,low=0.025,upp=0.975):
	#..Ordenando..#
	Xtam=len(X);
	Xnu=np.reshape(X,(1,Xtam));
	Xindex=np.argsort(Xnu);
	Xsort=np.sort(Xnu)
	#..Preproc esando parametros..#
	Y=np.ones(Xtam);
	#..Indices..#
	NS=np.sum(Y);
	MI=(NS-1)/2;
	PLI=low*(NS-1);
	PUI=upp*(NS-1);
	#..Suma Acumulada..#
	CF=np.cumsum(Y);
	CF=CF-1;
	#..calculo de contenedores..#
	MBF=0.0;fl1=False;
	MBC=0.0;fl2=False;
	PLBF=0.0;fl3=False;
	PLBC=0.0;fl4=False;
	PUBF=0.0;fl5=False;
	PUBC=0.0;fl6=False;
	for i in CF:
		if (i>=np.floor(MI))&(fl1==False):
			MBF=i;
			fl1=True;
		if (i>=np.ceil(MI))&(fl2==False):
			MBC=i;
			fl2=True;
		if (i>=np.floor(PLI))&(fl3==False):
			PLBF=i;
			fl3=True;
		if (i>=np.ceil(PLI))&(fl4==False):
			PLBC=i;
			fl4=True;
		if (i>=np.floor(PUI))&(fl5==False):
			PUBF=i;
			fl5=True;
		if (i>=np.ceil(PUI))&(fl6==False):
			PUBC=i;
			fl6=True;
	if fl6==False:
		PUBC=PUBF;
	M=(Xsort[0,MBF]+Xsort[0,MBC])/2;
	#..percentiles finales..#
	PL=(Xsort[0,PLBF]+Xsort[0,PLBC])/2;
	PU=(Xsort[0,PUBF]+Xsort[0,PUBC])/2;
	#..salida: PL y PE, Percentil superior e inferior segn los grados de ingreso
	return (PL,PU);
	
def copy_list(vec_l2):
    vec_l1=[];
    if isinstance(vec_l2,list):
        for val_cur in vec_l2:
            vec_l1.append(copy_list(val_cur));
    else:
        vec_l1=vec_l2;
    return vec_l1;

def int_list_normalization(vec_int):
	vec_ret=[];
	if isinstance(vec_int,list):
		for val_cur in vec_int:
			vec_ret.append(int_list_normalization(val_cur));
	else:
		vec_ret=int(vec_int);
	return vec_ret;

def float_list_normalization(vec_float):
	vec_ret=[];
	if isinstance(vec_float,list):
		for val_cur in vec_float:
			vec_ret.append(float_list_normalization(val_cur));
	else:
		vec_ret=float(vec_float);
	return vec_ret;

def pre_process_list(mat_pseudo_list):
	vec_ready=mat_pseudo_list.tolist();
	return vec_ready;
	
def pre_process_list_of_list(mat_pseudo_list):
	vec_ready=[];
	int_tam=len(mat_pseudo_list);
	for i in range(0,int_tam):
		vec_temp=mat_pseudo_list[i].tolist();
		vec_ready.append(vec_temp);
	return vec_ready;

def convert_to_list(vec_pseudo):
	# Validating Input
	vec_ret=[];
	if(isinstance(vec_pseudo,list)):
		vec_ret=vec_pseudo;
	elif(isinstance(vec_pseudo,np.matrix)):
		vec_ret=vec_pseudo.flatten().tolist()[0];
	elif(isinstance(vec_pseudo,np.ndarray)):
		vec_ret=vec_pseudo.tolist();
	else:
		print "ERROR (GNG.ssl_recover_all): input parameter is not a list -> ", type(vec_class);
		return;
	return vec_ret;

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
#                           Harmonic Manifold                        #
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
	
def HarmonicRegularizer(Graph,Yl): 
	tam=len(Yl);
	tot=len(Graph);
	L=getLaplaciano(Graph);
	L=np.matrix(L);
	Yl=np.matrix(Yl);
	Lll=L[0:tam,0:tam];
	Llu=L[0:tam,tam:tot];
	Lul=L[tam:tot,0:tam];
	Luu=L[tam:tot,tam:tot];
	fl=Yl;
	fl=np.reshape(fl,(tam,1))
	#fu1=np.linalg.inv(Luu);
	#fu2=-fu1;
	#fu3=fu2*Lul;
	#fu=fu3*np.transpose(Yl);
	fu=-np.linalg.pinv(Luu)*Lul*np.transpose(Yl);
	fr=np.concatenate((fl,fu),0);
	return fr;

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
#                           Krylov with MINRES                       #
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

def KrylovRegularizer(Graph,Yl,it):
	tam=len(Yl);
	tot=len(Graph);
	u=tot-tam;
	L=getLaplaciano(Graph);
	L=np.matrix(L);
	Luu=L[tam:tot,tam:tot];
	Yl=np.matrix(Yl);
	W=np.matrix(Graph);
	Wul=W[tam:tot,0:tam];
	Yl=np.reshape(Yl,(tam,1));
	b=Wul*Yl;
	beta_prev=0;
	beta=0;
	alpha=0;
	q_prev=[0]*(u);
	q_prev=np.matrix(q_prev);
	q_prev=np.reshape(q_prev,(u,1));
	b_mag=np.linalg.norm(b);
	#b_mag=np.sqrt(np.sum(np.square(b)));
	q=b/b_mag;
	H=[];
	Q=q;
	Q=np.matrix(Q);
	Yu=[];
	for t in range(1,it):
		v=Luu*q; # actualizando v
		alpha=np.transpose(q)*v;
		v=v-(beta_prev*q_prev)-(alpha[0,0]*q);
		
		beta=np.sqrt(np.sum(np.square(v))); # igualando parametros alpha y beta y actualizando matriz H
		vec_temp=[];
		if(t==1):
			vec_temp=[[alpha[0,0]],[beta]];
			H=vec_temp;
			H=np.matrix(H);
		else:
			vec_temp=[[beta_prev],[alpha],[beta]];
			vec_temp=np.matrix(vec_temp);
			tam_H=np.shape(H);
			rows=tam_H[0];
			cols=tam_H[1];
			hor_zeros=[0]*cols;
			hor_zeros=np.matrix(hor_zeros);
			H=np.concatenate((H,hor_zeros),0);
			if (rows-2<1):
				H=np.concatenate((H,vec_temp),1);
			else:
				ver_zeros=[0]*(rows-2);
				ver_zeros=np.matrix(ver_zeros);
				ver_zeros=np.reshape(ver_zeros,(rows-2,1));
				ver_nonzeros=np.concatenate((ver_zeros,vec_temp),0);
				H=np.concatenate((H,ver_nonzeros),1);
		beta_prev=beta;
		
		#OPTIMIZAR POR LEAST SQUARES con H c b_mag y e1
		#----------------------------------------
		e1=[0]*(t+1);
		e1[0]=1;
		e1=np.matrix(e1);
		e1=np.reshape(e1,(t+1,1));
		at_b=((-1)*b_mag)*e1;
		at_a=(-1)*H;
		c,par1,par2,par3=np.linalg.lstsq(at_a,at_b);
		#---------------------------------------
		
		q_prev=q; # actualizando vectores q
		if(beta!=0):
			q=v/beta;
		else:
			q=v/(beta+1);
		
		#APROXIMACION DE YU CON Q Y c.
		#----------------------------------------
		Yu=Q*c;
		#----------------------------------------
		
		Q=np.concatenate((Q,q),1);
	Y=np.concatenate((Yl,Yu),0);
	return Y;

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
#                           Fast Gauss Transform                     #
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

def FastGaussTransform(arr_data):
	pass;

def KCentersClustering(arr_data,int_K):
	int_tam=len(arr_data);
	print int_tam;
	int_ind=random.randrange(0,int_tam,2);
	print int_ind;
	vec_K_centers=[];
	arr_distances=[];
	flo_dist_max=-1;
	int_ind_max=-1;
	for i in range(0,int_K):	
		vec_K_centers.append(int_ind);
		vec_temp=[];
		for j in range(0,int_tam):
			flo_act_dist=distance(arr_data[j],arr_data[int_ind]);
			vec_temp.append(flo_act_dist);
			if flo_act_dist>flo_dist_max:
				int_count=vec_K_centers.count(j);
				if int_count==0:
					flo_dist_max=flo_act_dist;
					int_ind_max=j;
		arr_distances.append(vec_temp);
		int_ind=int_ind_max;
	print vec_K_centers;
	print np.array(arr_distances);	
	
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
#             EigenVectors of the Smooth Operator                    #
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#	
	
def SmoothOperatorRegularizer(arr_graph,vec_label,int_eigenvectors):
	# Setting the label vector
	vec_lab=vec_label[:];
	configureLabels(vec_lab,0,-1);
	int_tam_pat=len(arr_graph);
	int_tam_lab=len(vec_lab);
	int_tam_rest=int_tam_pat-int_tam_lab;
	for i in range(0,int_tam_rest):
		vec_lab.append(0);
	mat_lab=np.array(vec_lab);
	# Setting the Lambda matrix;
	vec_lambda=[];
	for i in range(0,int_tam_pat):
		if(vec_lab[i]!=0):
			vec_lambda.append(1000);
		else:
			vec_lambda.append(0);
	mat_lambda=np.diag(np.array(vec_lambda));
	# Setting Graph Laplacian			
	arr_L=getLaplaciano(arr_graph);
	mat_LN=normalizeLaplaciano(arr_L,arr_graph);
	# Getting Eigenvalues and eigenvectors
	mat_U,mat_S,mat_V=np.linalg.svd(mat_LN);
	mat_rev_U=np.fliplr(mat_U);
	mat_sel_U=mat_rev_U[:,1:int_eigenvectors+1]; # eigenvectors with smallest eigenvalue.
	mat_sel_U_t=np.transpose(mat_sel_U);
	mat_left1=np.dot(np.dot(mat_sel_U_t,mat_LN),mat_sel_U);
	mat_left2=np.dot(np.dot(mat_sel_U_t,mat_lambda),mat_sel_U);
	mat_left=mat_left1+mat_left2;
	#mat_lab_t=np.transpose(mat_lab);
	mat_right=np.dot(np.dot(mat_sel_U_t,mat_lambda),mat_lab);
	mat_left_inv=np.linalg.pinv(mat_left);
 	mat_alpha=np.dot(mat_left_inv,mat_right);
	mat_f=np.dot(mat_sel_U,mat_alpha);
	return mat_f;

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
#                           Eigen Functions                          #
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

def CalculateEigenFunction(data1D,sigma,epsilon,num_bins):
	data1D=np.matrix(data1D);
	#..Calculo de numero de beans del histograma..#
	tam_data=len(data1D);
	tam_data=tam_data+1.1-1.1;
	nBins=tam_data/5;
	nBins=np.ceil(min(nBins,num_bins));
	
	#..Calcular histograma..#
	pp,bins=np.histogram(data1D,nBins); # hallando histograma.
	bins_temp=[]; # recalculando bins
	for i in range(0,len(bins)-1):
		temp= (bins[i]+bins[i+1])/2
		bins_temp.append(temp);
	bins=bins_temp;
	bins=np.reshape(bins,(len(bins),1));
	pp=pp+1.1-1.1;
	pp=np.reshape(pp,(len(pp),1));
	pp=pp/np.sum(pp); # normalizando histograma
	pOld=pp # guardando anterior histograma
	pp=pp+epsilon; # sumando un minimo para evitar 0s
	pp=pp/np.sum(pp); # normalizando el nuevo histograma que evita 0s
	
	#..Calculando W chapeu..#	
	W=affinityMatrix(bins);
	W=np.matrix(W);
	
	#..Calculando P..#
	pptemp=np.reshape(pp,(1,len(pp)));
	P=np.diag(pptemp[0]);
	
	#..Calculando D sombrero..#
	PW=P*W;
	Ds_temp=[];
	for j in (PW):
		 Ds_temp.append(np.sum(j));
	Ds=np.diag(Ds_temp);
	
	#..Calculado D chapeu..#
	PWP=P*W*P;
	Dch_temp=[];
	for j in (PWP):
		 Dch_temp.append(np.sum(j));
	Dch=np.diag(Dch_temp);
	
	#..Resolviendo la ecuacion (D~-PW~P)g=oPD^g, hallando autofunciones g..#
	nL=Dch-PWP;
	PDs=P*Ds;
	IP=np.linalg.inv(np.sqrt(PDs))
	LL=IP*nL*IP
	UU,SS,VV=np.linalg.svd(LL);
	g=IP*UU; # g es al aproximacion numerica de las autofunciones de data1D;
	
	#..hallando autovalores de las autofunciones..#
	gT=np.transpose(g);
	M_temp=gT*nL*g;
	lambdas=np.diag(M_temp);
	M2_temp=gT*P*Ds*g;
	lambdas=lambdas/np.diag(M2_temp);
	lambdas=np.reshape(lambdas,(len(lambdas),1));
	
	#..Salidas: bins del histograma, autofunciones, autovalores, histograma.
	return (bins,g,lambdas,pp);
	
def AutoFunctionAproximator(Data,sigma,num_evecs,epsilon):
	#..Preprocesamiento de parametros..#
	CLIP_MARGIN=2.5;
	nPoints,nDims=np.shape(Data);
	Data_matrix=np.matrix(Data);
	
	#..Calculo de Autofunciones y Autovalores..#
	Bins=[];
	Gs=[];
	lambdas=[];
	for i in range(0,nDims):
		temp_bins,temp_gs,temp_lambdas,pp=CalculateEigenFunction(Data_matrix[:,i],sigma,epsilon,10);
		Bins.append(temp_bins);
		Gs.append(temp_gs);
		lambdas.append(temp_lambdas);
		
	#..Escogemos los num_evecs menores autofunciones sobre todas las dimensiones..#
	lambdas=np.matrix(np.transpose(lambdas));# formateando lambdas
	lambdas_dim=np.shape(lambdas);
	mthv=1*(math.pow(10,-10));# minimo permitido para autovalores
	mnv=1*(math.pow(10,10));
	#--------inicio(RIC1) esta parte del codigo puede cambiarse por (ver RIC2)
	lambdas_tam = np.shape(lambdas);
	for il in range(0,lambdas_tam[0]):
		for jl in range(0,lambdas_tam[1]):
			if (lambdas[il,jl]<mthv):
				lambdas[il,jl]=mnv;
	#--------fin(RIC1)
	#--------inicio(RIC2) esta parte del codigo puede cambiarse por (ver RIC1)
	#it=np.nditer(lambdas,[],['readwrite']);# eliminar los autovalores muy pequenos
	#while not it.finished:
	#	if (it[0]<mthv):
	#		it[0]=mnv;
	#	it.iternext()
	#--------fin(RIC2)
	
	vec_lambdas=np.reshape(lambdas,(1,lambdas_dim[0]*lambdas_dim[1]),order='F');
	sorted_lambdas=np.sort(vec_lambdas);# ordenar los autovalores.
	index_lambdas=np.argsort(vec_lambdas);# indices ordenados del vector de lambdas
	index_lambdas=index_lambdas[0,0:num_evecs];# escoger los num_evecs mas pequenos.
	out_lambda=sorted_lambdas[0,0:num_evecs]; 
	index_lambdas=np.transpose(index_lambdas);
	out_lambda=np.transpose(out_lambda);
	Lambdas_diag=np.diagflat(out_lambda);# SALIDA,matriz diagonal de autovalores
	iijj=[];# identificando los indices de los autovalores en la matriz de lambdas modificada
	rows_lambdas=np.shape(lambdas)[0];
	for i in range(0,num_evecs):
		ij_temp=indtosub(rows_lambdas,index_lambdas[i]);
		iijj.append(ij_temp);# estan intercambiados los indices i y j de lambdas porque la matriz fue transpuesta
	
	#..Acotar data por arriba y abajo..#
	low3=CLIP_MARGIN/100;
	upp3=1-low3;
	for a in range(0,nDims):
		cl3,cu3=percentil(Data_matrix[:,a],low3,upp3);
		#--------inicio(RIC3) esta parte del codigo puede cambiarse por (ver RIC4)
		Data_matrix_tam=np.shape(Data_matrix);
		for dmi in range(0,Data_matrix_tam[0]):
			for dmj in range(0,Data_matrix_tam[1]):
				if (Data_matrix[dmi,dmj]<cl3):
					Data_matrix[dmi,dmj]=cl3;
				else:
					if(Data_matrix[dmi,dmj]>cu3):
						Data_matrix[dmi,dmj]=cu3;
		#--------fin(RIC3)
		#--------inicio(RIC4) esta parte del codigo puede cambiarse por (ver RIC3)
		#it=np.nditer(Data_matrix[:,a],[],['readwrite']);
		#while not it.finished:
		#	if (it[0]<cl3):
		#		it[0]=cl3;
		#	else:
		#		if(it[0]>cu3):
		#			it[0]=cu3;
		#	it.iternext();
		#--------fin(RIC4)
			
	#..Interpolar la data en los num_evecs mas pequenas autofunciones (es decir que tengan los autovalores mas pequenos)
	UU2=[];
	Bins=np.matrix(np.transpose(Bins));# formateando lambdas
	Bins_dim=np.shape(Bins);
	Gs_dim=np.shape(Gs);
	for a in range(0,num_evecs):
		bins_out=Bins[:,iijj[a][1]]; # sacando los bins del histograma.
		uu=Gs[iijj[a][1]][:,iijj[a][0]]; # sacando las autofunciones.
		li_bins_out=bins_out.flatten().tolist()[0];
		li_uu=uu.flatten().tolist()[0];
		so_bins_out=np.sort(li_bins_out);
		in_bins_out=np.argsort(li_bins_out);
		so_uu=[];
		for eve in in_bins_out:
			so_uu.append(li_uu[eve]);
		data_temp=Data_matrix[:,iijj[a][1]];
		data_temp=data_temp.flatten().tolist()[0];
		temp_uu2=np.interp(data_temp,so_bins_out,so_uu); # interpolamos los bins del histograma con sus valores de las autofunciones, para hallar los valores de los autovectores aproximados de los datos en la dimension del histograma.
		UU2.append(temp_uu2);
	UU2=np.transpose(UU2);
	PUU2=np.power(UU2,2);
	NUU2=[];
	for q in range(0,num_evecs):
		Nuu2_temp=UU2[:,q]/(np.sqrt(np.sum(PUU2[:,q])));
		NUU2.append(Nuu2_temp);
	NUU2=np.transpose(NUU2); # SALIDA, matriz de autovectores aproximados.
	#salidas: matriz diagonal de autovalores(NExNE) y matriz de autovectores aproximados(NpontosxNE)
	return (Lambdas_diag,NUU2);

def EigenFunctionRegularizer(Data,Yl,sigma,num_evecs,epsilon):
	DD,UU=AutoFunctionAproximator(Data,sigma,num_evecs,epsilon);
	data_tam=len(Data);
	YY=Yl;
	YY_tam=len(YY);
	for i in range(0,YY_tam):
		if YY[i]==0:
			YY[i]=-1;
	YY_con=[50]*YY_tam;
	YY=np.transpose(np.matrix(YY));
	YY_con=np.transpose(np.matrix(YY_con));
	vec_temp=np.zeros((data_tam-YY_tam,1));
	vec_temp=np.matrix(vec_temp);
	Lambda=np.diagflat(np.concatenate((YY_con,vec_temp),0));
	YY=np.concatenate((YY,vec_temp),0);
	UUT=np.transpose(UU);
	alpha=np.linalg.inv(DD+(UUT*Lambda*UU))*(UUT*Lambda*YY);
	Yf=UU*alpha;
	return Yf;
   
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
#                           Nystrom Isomap 1NN                       #
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

def validar(L): # Si el vector tiene por lo menos un valor distinto de -1 , -1 representa el estado vacio.
	int_tam=len(L);
	for i in range(0,int_tam):
		if(L[i]!=-1):
			return True;
	return False;

def minimumPositive(L):
	int_min=9999999;
	int_est=-1;
	int_ind=-1;
	int_tam=len(L);
	for i in range(0,int_tam):
		if(L[i]>-1):
			if(L[i]<int_min):
				int_min=L[i];
				int_ind=i;
				int_est=1;
			if(L[i]==int_min):
				int_ind=i;
	return (int_ind,int_est);

def markGeodesicDistance(arr_gd,vec_flags,int_node,pi,d):
	if vec_flags[int_node]==False:
		int_tam=len(arr_gd);
		vec_temp=[-1]*int_tam;
		for j in range(0,int_tam):
			if(int_node==j):
				vec_temp[j]=0;
			elif(pi[j]!=-1):
				vec_temp[j]=d[j];
			elif(pi[j]==-1):
				vec_temp[j]=9999999; # revisar, posible problema.
		arr_gd[int_node]=vec_temp;
		vec_flags[int_node]=True;

def multiDijkstra(arr_G,int_s,arr_gd,vec_flags):
	d=[];
	pi=[];
	Q=[];
	int_tam_nodes=len(arr_G);
	d=[9999999]*int_tam_nodes; # revisar, posible problema.
	Q=[9999999]*int_tam_nodes; # revisar, posible problema.
	pi=[-1]*int_tam_nodes; 
	d[int_s]=0;
	Q[int_s]=0;
	while(validar(Q)):
		tup_min=minimumPositive(Q); 
		u=tup_min[0]# u es un indice
		int_est=tup_min[1]
		if(int_est==-1):
			markGeodesicDistance(arr_gd,vec_flags,int_s,pi,d);
			d=[9999999]*int_tam_nodes; # revisar, posible problema.
			pi=[-1]*int_tam_nodes; 
			int_s=u;
			d[u]=0;
		Q[u]=-1;
		int_j=0;
		for v in arr_G[u]: # v es el valor ( peso )
			if(v>0):
				if(d[int_j]>d[u]+v):
					d[int_j]=d[u]+v;
					pi[int_j]=u;
					Q[int_j]=d[int_j];
			int_j=int_j+1;
	markGeodesicDistance(arr_gd,vec_flags,int_s,pi,d);
	return pi;

def randomSampling(arr_sgd,l):
	int_tam=len(arr_sgd);
	int_raz=math.floor(int_tam/l);
	int_pos=0;
	vec_pos=[];
	for i in range(0,l):
		#int_fin=int_pos+int_raz+1;
		int_fin=int_pos+int_raz;
		if int_fin>int_tam:
			int_fin=int_tam;
		#if int_fin==int_pos:
		#	int_fin=int_pos+1;
		int_sel=random.randrange(int_pos,int_fin,2);
		vec_pos.append(int_sel);
		int_pos=int_fin;
	vec_nopos=[];
	int_ind_1=0;
	for i in range(0,int_tam):
		if(i==vec_pos[int_ind_1]):
				if(int_ind_1+1<l):
					int_ind_1=int_ind_1+1;
		else:
			vec_nopos.append(i);
	arr_W=[];
	for i in range(0,l):
		vec_temp=[];
		for j in range(0,l):
			vec_temp.append(arr_sgd[vec_pos[i]][vec_pos[j]]);
		arr_W.append(vec_temp);
	arr_G21=[];
	int_tam_nopos=len(vec_nopos);
	for i in range(0,int_tam_nopos):
		vec_temp_2=[];
		for j in range(0,l):
			vec_temp_2.append(arr_sgd[vec_nopos[i]][vec_pos[j]]);
		arr_G21.append(vec_temp_2);
	return (arr_W,arr_G21);

def nystrom(mat_sgd,l):  #nota en la medida de lo posible siempre usar la clase array de numpy#
	int_n=len(mat_sgd);
	arr_W,arr_G21=randomSampling(mat_sgd,l);
	mat_W=np.array(arr_W);
	mat_G21=np.array(arr_G21);
	mat_C=np.concatenate((mat_W,mat_G21),axis=0);
	vec_Zw,mat_Uw=np.linalg.eig(mat_W); #los autovectores son las columnas de la matriz de autovectores.
	mat_Zw=np.diag(vec_Zw);
	mat_ZwP=np.linalg.pinv(mat_Zw);
	flo_l=float(l);
	flo_n=float(int_n);
	flo_fac1=flo_n/flo_l;
	flo_fac2=math.sqrt(flo_l/flo_n);
	mat_Zaprox=flo_fac1*mat_Zw;
	mat_Uaprox=flo_fac2*np.dot(np.dot(mat_C,mat_Uw),mat_ZwP);
	return (mat_Zaprox,mat_Uaprox);

def NI_1_buildG(data,t,dt='e',wt='e'):
	graph=[];
	for vec1 in data:
		edges=[];
		for vec2 in data:
			dst=distance(vec1,vec2,dt);
			if(dst==0) :
				dst=9999999; # revisar, posible problema!
			edges.append(dst);
		tam=len(edges)
		edges2=[0]*tam;
		for i in range(0,t):
			minim=min(edges);
			indice=edges.index(minim);
			edges2[indice]=minim;
			edges[indice]=9999999;  # revisar, posible problema!
		graph.append(edges2);
	graph=makeSymetric(graph);
	return graph;

def NI_2_geodesic(G):
	int_tamG=len(G);
	arr_gd_temp=[-1]*int_tamG;
	arr_gd=[arr_gd_temp]*int_tamG;
	vec_flags=[False]*int_tamG;
	for i in range(0,int_tamG):
		if(vec_flags[i]==False):
			multiDijkstra(G,i,arr_gd,vec_flags);
	arr_S=[];
	for i in range(0,int_tamG):
		vec_temp=[];
		for j in range(0,int_tamG):
			val_cur=arr_gd[i][j];
			val_sq=val_cur;
			val_s=(val_sq*val_sq);
			vec_temp.append(val_s);
		arr_S.append(vec_temp);
	mat_dI=np.identity(int_tamG);
	flo_cf_01=1/float(int_tamG);
	mat_H=mat_dI-flo_cf_01;
	mat_S=np.array(arr_S);
	mat_ro=-np.dot(np.dot(mat_H,mat_S),mat_H)/2;
	return mat_ro;

def sortEigen(mat_Z,mat_U):#matriz diagonal de autovalore, matrias de vectores columna de autovectores.
	vec_Z=mat_Z.diagonal();
	dic_temp={};
	mat_UT=np.transpose(mat_U);
	int_tam=len(vec_Z);
	for i in range(0,int_tam):
		dic_temp[vec_Z[i]]=mat_UT[i];
	vec_Z_sorted=np.array(sorted(vec_Z));
	arr_UT_sorted=[];
	vec_Z_reversed=[];
	for i in reversed(range(0,int_tam)):
		vec_Z_reversed.append(vec_Z_sorted[i]);
		arr_UT_sorted.append(dic_temp[vec_Z_sorted[i]]);
	mat_Z_sorted=np.diag(vec_Z_reversed);
	mat_UT_sorted=np.array(arr_UT_sorted);
	mat_U_sorted=np.transpose(mat_UT_sorted);
	return (mat_Z_sorted,mat_U_sorted);
	
def NI_3_spectral(arr_geo,l):
	mat_Zk,mat_Uk=nystrom(arr_geo,l);
	mat_Zo,mat_Uo=sortEigen(mat_Zk,mat_Uk);
	mat_UoT=np.transpose(mat_Uo);
	vec_Zo=mat_Zo.diagonal();
	Y=[];
	int_N=len(arr_geo);
	int_p=len(vec_Zo);
	for i in range(0,int_N):
		vec_yi=[];
		for j in range(0,int_p):
			#flo_val=math.sqrt(vec_Zo[j])*mat_UoT[j,i];
			flo_val=np.lib.scimath.sqrt(vec_Zo[j])*mat_UoT[j,i];
			vec_yi.append(flo_val);
		Y.append(vec_yi);
	mat_Y=np.array(Y);
	return mat_Y;

def NystromIsomap(arr_data,int_k,int_l):
	arr_G=NI_1_buildG(arr_data,int_k);
	arr_SGD=NI_2_geodesic(arr_G);
	mat_Y=NI_3_spectral(arr_SGD,int_l);
	return mat_Y;

def KNN_classifier_build(mat_data): #matriz de datos.
	tree_mam = spatial.KDTree(mat_data);
	#print tree_mam.data;
	return tree_mam;

def KNN_classifier_predict(tree_data,mat_points,mat_labels,int_K): # tree,numpy.array,numpy.array,int
	int_data_tam=len(tree_data.data)
	if(int_K>int_data_tam):
		int_K=int_data_tam;
	val_query=tree_data.query(mat_points,int_K); # get the query in the KDTree
	val_query_ind=val_query[1]; # get the knn indices
	int_tam=len(mat_points); # get the size of the output, or points that were queried
	vec_resp=[]; # vector (list) of answers
	for i in range(0,int_tam): # iterating over the knn of each point that were queried
		vec_labels_temp=mat_labels[val_query_ind[i]]; # get the labels of the knn of each queried point.
		if(int_K==1): # check if it is 1-nn (k=1) and convert the current label into a list for integrity
			vec_labels_temp=[vec_labels_temp]; # convert the current label into a list for integrity
		vec_int_counts=np.bincount(vec_labels_temp); # count the labels.
		int_label_final_cur=np.argmax(vec_int_counts); # get the most common label.
		vec_resp.append(int_label_final_cur); # append to the reponse list the final label for the current queried point.
	return vec_resp;

def KNN_classifier_percent(tree_data,mat_points,mat_labels,int_K): # tree,numpy.array,numpy.array,int
	int_data_tam=len(tree_data.data)
	if(int_K>int_data_tam):
		int_K=int_data_tam;
	val_query=tree_data.query(mat_points,int_K); # get the query in the KDTree
	val_query_ind=val_query[1]; # get the knn indices
	int_tam=len(mat_points); # get the size of the output, or points that were queried
	vec_resp=[]; # vector (list) of answers
	for i in range(0,int_tam): # iterating over the knn of each point that were queried
		vec_labels_temp=mat_labels[val_query_ind[i]]; # get the labels of the knn of each queried point.
		if(int_K==1): # check if it is 1-nn (k=1) and convert the current label into a list for integrity
			vec_labels_temp=[vec_labels_temp]; # convert the current label into a list for integrity
		int_vec_labels_temp_tam=len(vec_labels_temp); # cast operation over the vector of labels;
		vec_labels_temp_casted=[]
		for j in range(0,int_vec_labels_temp_tam):
			vec_labels_temp_casted.append(int(vec_labels_temp[j]));
		vec_int_counts=np.bincount(vec_labels_temp_casted); # count the labels.
		float_percent=float(1)-(float(vec_int_counts[0])/int_K);	
		#int_label_final_cur=np.argmax(vec_int_counts); # get the most common label.
		vec_resp.append(float_percent); # append to the reponse list the final label for the current queried point.
	return vec_resp;
      
def NystromIsomapKNN(vec_data,vec_labels,int_k_graph,int_sampling,int_K):
	mat_reduced_data=NystromIsomap(vec_data,int_k_graph,int_sampling);
	int_tam_data=len(vec_data);
	int_tam_labels=len(vec_labels);
	mat_labels=np.array(vec_labels);
	tree_val=KNN_classifier_build(mat_reduced_data[0:int_tam_labels]);
	#vec_resp=KNN_classifier_predict(tree_val,mat_reduced_data[int_tam_labels:int_tam_data],mat_labels,int_K);
	vec_resp=KNN_classifier_percent(tree_val,mat_reduced_data[int_tam_labels:int_tam_data],mat_labels,int_K);
	#mat2_resp=np.matrix(vec_resp);
	mat_resp=np.array(vec_resp);
	mat_resp=np.concatenate((mat_labels,mat_resp),axis=0);
	return mat_resp;

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
#                         Decision Functions                         #
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

def alphaCut(Mat,alpha):
	ind=0;
	for i in Mat:
		if (i-math.floor(i))<alpha:
			Mat[ind,0]=math.floor(i);
		else:
			Mat[ind,0]=math.ceil(i);
		ind=ind+1;
	return Mat;
	
def alphaCutV02(Mat,alpha):
	ind=0;
	Mat2=np.matrix(Mat);
	for i in Mat:
		if (i-math.floor(i))<alpha:
			Mat2[0,ind]=math.floor(i);
		else:
			Mat2[0,ind]=math.ceil(i);
		ind=ind+1;
	return Mat2;
	
def alphaZeroCut(Mat):
	ind=0;
	for i in Mat:
		if (i<0):
			Mat[ind,0]=0;
		else:
			Mat[ind,0]=1;
		ind=ind+1;
	return Mat;
	
def generalizeAlphaCut(mat_arg,float_cut,int_neg,int_pos):
	int_tam=len(mat_arg);
	vec_resp=[];
	for i in range(0,int_tam):
		if(mat_arg[i]<float_cut):
			vec_resp.append(int_neg);
		else:
			vec_resp.append(int_pos);
	return vec_resp;

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
#                         Testing Functions                          #
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

def filterData(data,label,eti,cantpos,cantneg,totpos,totneg):
	tot=np.shape(data)[0];
	out_data=[]; # data lista
	out_label=[]; # labels listos
	out_label_final=[]; # labels para comprobar
	label_index=[]; # indices de labels listos
	cont_pos=0;
	cont_neg=0;
	for i in range(0,tot): # iterando toda la data
		temp_data=data[i];
		temp_label=0;
		temp_index=i;
		if label[i]==eti:
			cont_pos=cont_pos+1;
			temp_label=label[i];
			if cont_pos<=totpos:
				out_data.append(temp_data);
				out_label_final.append(temp_label);
		else:
			cont_neg=cont_neg+1;
			temp_label=0;
			if cont_neg<=totneg:
				out_data.append(temp_data);
				out_label_final.append(temp_label);
		if (cont_neg>totneg) & (cont_pos>totpos):
			break;
	cont_pos=0;
	cont_neg=0;
	for j in range(0,totpos+totneg): # iterando la data seleccionada (out_data,out_label_final)
		temp_data=out_data[j];
		temp_label=0;
		temp_index=j;
		if out_label_final[j]==eti:
			cont_pos=cont_pos+1;
			temp_label=out_label_final[j];
			if cont_pos<=cantpos:
				out_label.append(temp_label);
				label_index.append(temp_index);
		else:
			cont_neg=cont_neg+1;
			temp_label=0;
			if cont_neg<=cantneg:
				out_label.append(temp_label);
				label_index.append(temp_index);
		if (cont_neg>cantneg) & (cont_pos>cantpos):
			break;
	out_data=reOrganizeVector(out_data,label_index);
	out_label_final=reOrganizeVector(out_label_final,label_index);
	#print np.shape(out_data);
	#print np.shape(out_label_final);
	#print len(label_index);
	#print label_index;
	#print np.shape(out_label);
	#print out_label;
	#print out_label_final;
	return (out_data,out_label_final,out_label); #  out_label esta con 0s y etis

def filterDataRandomly(data,label,eti,cantpos,cantneg,totpos,totneg):
	tot=np.shape(data)[0];
	out_data=[]; # data lista
	out_label=[]; # labels listos
	out_label_final=[]; # labels para comprobar
	label_index=[]; # indices de labels listos
	cont_pos=0;
	cont_neg=0;
	for i in range(0,tot): # iterando toda la data
		temp_data=data[i];
		temp_label=0;
		temp_index=i;
		if label[i]==eti:
			cont_pos=cont_pos+1;
			temp_label=label[i];
			if cont_pos<=totpos:
				out_data.append(temp_data);
				out_label_final.append(temp_label);
		else:
			cont_neg=cont_neg+1;
			temp_label=0;
			if cont_neg<=totneg:
				out_data.append(temp_data);
				out_label_final.append(temp_label);
		if (cont_neg>totneg) & (cont_pos>totpos):
			break;
	cont_pos=0;
	cont_neg=0;
	int_random=random.randrange(0,totpos+totneg,2);
	vec_pos_selected_data=range(0,totpos+totneg);
	vec_pos_randomized_data=vec_pos_selected_data[int_random:]+vec_pos_selected_data[0:int_random];
	for j in vec_pos_randomized_data: # iterando la data seleccionada (out_data,out_label_final)
		temp_data=out_data[j];
		temp_label=0;
		temp_index=j;
		if out_label_final[j]==eti:
			cont_pos=cont_pos+1;
			temp_label=out_label_final[j];
			if cont_pos<=cantpos:
				out_label.append(temp_label);
				label_index.append(temp_index);
		else:
			cont_neg=cont_neg+1;
			temp_label=0;
			if cont_neg<=cantneg:
				out_label.append(temp_label);
				label_index.append(temp_index);
		if (cont_neg>cantneg) & (cont_pos>cantpos):
			break;
	out_data=reOrganizeVectorRobust(out_data,label_index);
	out_label_final=reOrganizeVectorRobust(out_label_final,label_index);
	return (out_data,out_label_final,out_label); #  out_label esta con 0s y etis

def accuracyTest(v1,v2):
	tam1=len(v1);
	tam2=len(v2);
	if tam1==tam2:
		cont=0;
		for i in range(0,tam1):
			v1[i]=int(v1[i]);
			v2[i]=int(v2[i]);
			if (v1[i]==v2[i]):
				cont=cont+1;
		return (float(cont)/tam1);
	else:
		print "ERROR (accuracyTest): vectores de distintos tamanos";
		return 0;
		
def confussionMatrix(v1,v2):#v1 resultados predecidos, v2 resultados verdaderos(reales)
	tam1=len(v1);
	tam2=len(v2);
	if tam1==tam2:
		TP=0;
		TN=0;
		FP=0;
		FN=0;
		for i in range(0,tam1):
			v1[i]=int(v1[i]);
			v2[i]=int(v2[i]);
			if (v1[i]==v2[i]):
				if v1[i]==1:
					TP=TP+1;
				else:
					TN=TN+1;
			else:
				if v1[i]==1:
					FP=FP+1;
				else:
					FN=FN+1;
		return (TP,TN,FP,FN);
	else:
		print "ERROR (confussionMatrix): vectores de distintos tamanos";
		return 0;

