#! /usr/bin/env python
# -*- coding: iso-8859-1 -*-

import numpy as np
import scipy as sp
import Source as src
import Neuron as neu
import GNG as gng
import SSLFilter as ssl
import Test as ts

print "Hello"

data=np.loadtxt("Xt.txt");
label=np.loadtxt("Yt.txt");

dt,lb1,lb2=ts.GetData(data,label,20,20)

#print dt,type(dt),len(dt)
#print lb1,type(lb1),len(lb1)
#print lb2,type(lb2),len(lb2)

#gr=ts.BuildKGraph(dt,5)

#gr=ts.BuildFullGraph(dt)
#print gr,type(gr),type(gr[0]),len(gr),len(gr[0])

"""print type(gr),type(gr[0])
print len(gr),len(gr[0])"""

#ts.FIR_Regularizer(gr,lb2,1,5,0.5,"FIRtest(1).txt",lb1)

"""
ts.FIR_Regularizer(gr,lb2,1,5,0.5,"FIRtest(1).txt",lb1)
ts.FIR_Regularizer(gr,lb2,2,5,0.5,"FIRtest(2).txt",lb1)
ts.FIR_Regularizer(gr,lb2,3,5,0.5,"FIRtest(3).txt",lb1)
ts.FIR_Regularizer(gr,lb2,4,5,0.5,"FIRtest(4).txt",lb1)
ts.FIR_Regularizer(gr,lb2,5,5,0.5,"FIRtest(5).txt",lb1)
ts.FIR_Regularizer(gr,lb2,6,5,0.5,"FIRtest(6).txt",lb1)
ts.FIR_Regularizer(gr,lb2,7,5,0.5,"FIRtest(7).txt",lb1)
ts.FIR_Regularizer(gr,lb2,8,5,0.5,"FIRtest(8).txt",lb1)
ts.FIR_Regularizer(gr,lb2,9,5,0.5,"FIRtest(9).txt",lb1)
ts.FIR_Regularizer(gr,lb2,10,5,0.5,"FIRtest(10).txt",lb1)
"""

"""ts.FIR_Regularizer(gr,lb2,3,5,0.5,"FIRtest(3).txt",lb1)
ts.FIR_Regularizer(gr,lb2,4,5,0.5,"FIRtest(4).txt",lb1)
ts.FIR_Regularizer(gr,lb2,5,5,0.5,"FIRtest(5).txt",lb1)
ts.FIR_Regularizer(gr,lb2,10,5,0.5,"FIRtest(10).txt",lb1)
ts.FIR_Regularizer(gr,lb2,15,5,0.5,"FIRtest(15).txt",lb1)
ts.FIR_Regularizer(gr,lb2,20,5,0.5,"FIRtest(20).txt",lb1)
ts.FIR_Regularizer(gr,lb2,25,5,0.5,"FIRtest(25).txt",lb1)
ts.FIR_Regularizer(gr,lb2,30,5,0.5,"FIRtest(30).txt",lb1)"""
#ts.HAR_Regularizer(gr,lb2,"HARtest.txt",lb1)
#ts.KRY_Regularizer(gr,lb2,10,"KRYtest.txt",lb1)
#ts.SMO_Regularizer(gr,lb2,0.95,"SMOtest.txt",lb1)
#ts.EIF_Regularizer(dt,lb2,0.2,8,0.1,"EIFtest.txt",lb1)
#ts.NYS_Regularizer(dt,lb2,5,15,3,"NYStest.txt",lb1)


#print "Finish"


#ssl_class = ssl.SSLFilter(3,[1,0,-1,1,0,0,0,0,0],[],0.5)
#print ssl_class




##--------------------------------------------------------------------------------------##


print "---probandoooooo---";
P=[[2,1],[5,3],[2,2],[2,3],[2,4],[2,5],[2,6],[2,7],[2,8],[2,9],[2,10],[3,10],[4,10],[5,10],[6,10],[7,10],[8,10],[9,10],[10,10],[10,9],[10,8],[10,7],[10,6],[10,5],[10,4],[10,3],[10,2],[5,4],[6,4],[6,3],[7,3],[7,4],[5,3.3],[6,3.5],[5,1],[5.5,1.5]];
#P=[[2,2],[5,4],[2,4],[2,5],[3,2],[4,2],[5,2],[6,2],[7,2],[2,3],[5,5],[5,6],[5,7],[5,8],[5,9],[4,9],[3,9],[2,9],[1,9],[11,11]];
#P=[[2,3],[7,7],[3,5],[1,2],[6,7],[8,6]];
#P = [[1,2],[15,16],[5,4],[16,17],[18,15],[5,3],[14,16],[1,1],[3,2],[14,14]];
Y=[1,0];
print Y;
#src.configureLabels(Y,0,-1);
print P;
#print len(P);
print "=================================================";
G=gng.GNG();
G.set_patterns(P);
int_tam_Gdata=len(G.get_patterns());
INT_RESOLUCION=35;
print int_tam_Gdata;
G.set_hyperparameters(0.5,0.2,int(int_tam_Gdata*0.5),0.2,int(int_tam_Gdata*0.5),0.05);
print ".................................................";
#G.train_wsc_fixed(0.2,5,int(int_tam_Gdata*0.7));
Cc,Bf=G.draw_training_fixed(0.2,5,int(int_tam_Gdata*0.7),INT_RESOLUCION);
print ".................................................";
G.print_GNG();
#G.print_GNG_data();
print ".................................................";
print G.index_graph_mapping();
print ".................................................";
#G.gm_connect_graph();
print ".................................................";
raw_input("--pausa-- ");
GR_GNG,Y_GNG,ind_GNG=G.ssl_get_all(Y);
#print GR_GNG;
print Y_GNG;
#print ind_GNG;
#G.draw_class_data(INT_RESOLUCION,Cc,Y);
#raw_input("--pausa-- ");
#G.draw_class_all(INT_RESOLUCION,Cc,Y,Y_GNG);
print ".................................................";
raw_input("--Aprete una tecla para continuar-- ");
print ".................................................";
print "Harmonic: ";
respH=G.ssl_wrap_regularizer(src.HarmonicRegularizer,GR_GNG,Y_GNG);
#print respH;
repH=src.alphaCut(respH,0.5);
repH=repH.flatten().tolist()[0];
print repH;
dataH=src.int_list_normalization(G.ssl_recover_all(repH,ind_GNG));
print dataH;
#G.draw_class_neuron(INT_RESOLUCION,Cc,repH);
#raw_input("--pausa-- ");
#G.draw_class_all(INT_RESOLUCION,Cc,dataH,repH);
#raw_input("--Aprete una tecla para continuar-- ");
print ".................................................";
print "Smooth Operator: ",int(len(GR_GNG)*0.5);
respS=G.ssl_wrap_regularizer(src.SmoothOperatorRegularizer,GR_GNG,Y_GNG,int((len(GR_GNG)*0.5)));
#print respS;
repS=src.generalizeAlphaCut(respS,0,0,1);
print repS;
dataS=src.int_list_normalization(G.ssl_recover_all(repS,ind_GNG));
print dataS;
#G.draw_class_neuron(INT_RESOLUCION,Cc,repS);
#raw_input("--pausa-- ");
#G.draw_class_all(INT_RESOLUCION,Cc,dataS,repS);
raw_input("--Aprete una tecla para continuar-- ");
print ".................................................";
print "Krylov: ";
respK=G.ssl_wrap_regularizer(src.KrylovRegularizer,GR_GNG,Y_GNG,100);
#print respK;
repK=src.alphaCut(respK,0.5);
repK=repK.flatten().tolist()[0];
print repK;
dataK=src.int_list_normalization(G.ssl_recover_all(repK,ind_GNG));
print dataK;
#G.draw_class_neuron(INT_RESOLUCION,Cc,repK);
#raw_input("--pausa-- ");
#G.draw_class_all(INT_RESOLUCION,Cc,dataK,repK);
raw_input("--Aprete una tecla para continuar-- ");
print ".................................................";
print "Reduccion: ", int_tam_Gdata, " a ", G.get_tamano();
raw_input("--Aprete una tecla para terminar-- ");


"""
G.initialize();
G.print_GNG();
pr=neu.Neuron([],0.1);
pr.become_random(0,5.5,3);
G.add_neuron(pr,0,1);
pr2=neu.Neuron([],0.2);
pr2.become_random(0,5.5,3);
G.add_neuron(pr2,0,2);
G.print_GNG();
#G.delete_neuron(1);
#G.print_GNG();
G.connect_neurons(0,1);
G.print_GNG();
G.disconnect_neurons(0,2);
G.print_GNG();
G.print_GNG_data();
print range(2,2);
print G.close_neurons(G.get_patterns()[0]);
G.increment_age();
G.increment_age();
#G.increment_age_by_neighborhood(0);
G.increment_age();
G.increment_age();
G.increment_age();
G.connect_neurons(0,1);
G.increment_age();
G.increment_age();
G.increment_age();
G.increment_age();
G.increment_age();
G.increment_age();
#G.print_GNG();
#int_pr=G.max_error_neuron();
#print int_pr;
#print G.max_error_by_neighborhood(int_pr);
#print G.are_connected(0,1);
#G.set_live(0,1,5);
#G.print_GNG();
#G.update_neighborhood([0,1,1],0.2,2);
#G.print_GNG();
#G.decrease_error_to_all(0.05);
G.print_GNG();
G.check_to_remove_connections();
G.print_GNG();
G.check_to_remove_neurons();
G.print_GNG();
"""

#print "=================================================";

"""
data = [[1,2],\
  [15,16],\
  [5,4],\
  [16,17],\
  [18,15],\
  [5,3],\
  [14,16]\
  ];

labeled=[0,1];
labels=[1,0];

g=src.reOrganizeVector(data,labeled);

#print g;
gg=src.buildFullGraph(g);

print "----------------------";

#data_matrix=np.matrix(data);

#respuesta=EigenFunctionRegularizer(g,labels,0.2,2,0.1);
#respuesta=KrylovRegularizer(gg,labels,4);
respuesta=src.HarmonicRegularizer(gg,labels);  
#respuesta=NystromIsomapKNN(g,labels,3,2,3); # vec_data,vec_labels,int_k_graph,int_sampling,int_K

#
#rep=alphaZeroCut(respuesta);
print "----------------------";
print respuesta;
rep=src.alphaCut(respuesta,0.5);
print rep;
rep=rep.flatten().tolist()[0];
print rep;
print 0==0.0;
print 0==0;
print 0==-0;
print 0==-0.0;
#print respuesta;
print "......................";
"""

#----------------------------------------------------------------------------------------------------#

#time.sleep(7);

#W=src.buildKGraph(P,2);
#yu=src.SmoothOperatorRegularizer(W,Y,2);
#print yu;
#rep=src.generalizeAlphaCut(yu,0,0,1);
#print rep;
#print W;
#print isinstance(P,list);
#print max(max(P));
#print min(min(P));


#G.draw_all(50);

#a=Neuron([1,1,1],1);
#print a;
#a.decrease_error(0.25);
#print a;
#a.update_error([2,1,1]);
#print a;
#a.update_vector([1,1,2],0.25)
#print a;
#b=Neuron([],0.1);
#print b;
#b.become_random(0,5.5,3);
#print b;
#print "-";
#c=Neuron([],0);
#c.become_middle(a,b,0.1);
#print c;
#print "-";
#print a;
#print b;
#int_sel=random.randrange(0,5,1);
#print int_sel;
#float_pr=src.randomfloat(0.9,5.2);
#print float_pr;



#----------------------------------------------------------------------------------------#
# 					Tests Section					 #
#----------------------------------------------------------------------------------------#
"""

data = [[1,2,3],\
  [5,4,4],\
  [25,36,30],\
  [35,30,30],\
  [29,35,30],\
  [5,3,4],\
  [32,36,25]\
  ];

labeled=[1,6,0,4];
labels=[1,0,1,0];

g=reOrganizeVector(data,labeled);

#print g;
#gg=buildFullGraph(g);

print "----------------------";

#data_matrix=np.matrix(data);

#respuesta=EigenFunctionRegularizer(g,labels,0.2,2,0.1);
#respuesta=KrylovRegularizer(gg,labels,4);
#respuesta=HarmonicRegularizer(gg,labels);  
respuesta=NystromIsomapKNN(g,labels,3,2,3); # vec_data,vec_labels,int_k_graph,int_sampling,int_K

#rep=alphaCut(respuesta,0);
#rep=alphaZeroCut(respuesta);
print "----------------------";
print respuesta;
#print respuesta;
print "......................";
#print respuesta[0:6];
print "----------------------";
#KCentersClustering(data,2);
print "----------------------";

"""

#----------------------------------------------------------------------------------------#
  
#data2 = [ ['a','b','c','d'],\
#  ['x','y','z','w'],\
#  ['1','2','3','4'],\
#  ['o','p','q','r'],\
# ]; 
#data3 = [[1,2],\
#  [5,4],\
#  [2,3]\
#  ];
#data4=np.matrix(data3); 
#data4T=np.transpose(data4);
#data5 = [[1,2,1],\
#  [2,1,1],\
#  [2,2,1]\
#  ];
#data6=np.matrix(data5); 
#pr=[[0]*4]*4;
#pr=range(0,4)
#data4inv=np.linalg.inv(data4);
#data7=data6[0:2,0:3];
#data8=np.concatenate((data7[0],data6),0);
#
#print data7;
#print data7[0];
#print data6;
#print data8;
#
#AutoFunctionAproximator(data,0.2,2,0.1)
#CalculateEigenFunction(data_matrix[:,1],0.2,0.1,10);
#percentil(data_matrix[:,1]);
#print indtosub(2,2);
#print g;
