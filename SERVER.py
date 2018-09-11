#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 21:24:26 2018

@author: joel
"""

from flask import Flask, request, jsonify

from sklearn.externals import joblib
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
#from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import GridSearchCV



import numpy as np
from pyDOE import lhs

import json

app = Flask(__name__)

#variaveis globais


"""
######################## INICIALIZA OS CLASSIFICADORES ##################################################################
"""
#classifierSVM = SVR(kernel='rbf', C=1e3, gamma=0.1, tol=0.01)
#classifierMLP = MLPRegressor(hidden_layer_sizes=(10,),max_iter=1000)
# #                                      activation='relu',
# #                                      solver='adam',
#  #                                     learning_rate='adaptive',
#   #                                    max_iter=10000,
#    #                                   learning_rate_init=0.01,
#     #                                  alpha=0.01)
#classifierRF = RandomForestRegressor(n_estimators=1000, max_depth=None,min_samples_split=2, random_state=0,criterion='mse');
classifierTree = DecisionTreeRegressor(max_depth=500,min_samples_split=2, random_state=0,criterion='mse')


"""
########################## VARIAVEIS GLOBAIS ############################################################################
"""
execult = [];
tamanho = 0;
txErro = [];
real = [];
classifier = classifierTree;


"""
######################################## FUNÇÕES PARA SALVAR OS VALORES ##################################################
"""

def Leitura(name):
	arquivo = open(name,'r')
	linhas = arquivo.readlines()
	retorno = []

	for linha in linhas:
		retorno.append(eval(linha.replace("[","").replace("]","").replce(" ",",")))
	arquivo.close()

	return retorno


def Save(lista, name):
    arquivo = open(name,'w')
    i = 0
    while i < len(lista):    
        arquivo.writelines((lista[i]))
        i += 1
    arquivo.close()

def EscreveArquivo(indice, lista, name):
    arquivo = open(name,'w')
    i = 0
    while i < len(lista):    
        arquivo.write(str(indice[i]) +" "+ str(lista[i])+'\n')
        i += 1
    arquivo.close()


def EscreveArquivoC(indice, compara ,lista, name):
    arquivo = open(name,'w')
    i = 0
    arquivo.write("Indice" +" "+ "Predição"+" "+"Rotulo"+" "+"Erro"+'\n')
    while i < len(lista):
        arquivo.write(str(indice[i]) +" "+str(compara[i])+" "+ str(lista[i])+'\n')
        i += 1
    arquivo.close()



"""
##################################################### TREINAMENTO ########################################################
"""
@app.route("/treinamento", methods=['GET', 'POST'])
def treino():
    global classifier
    lista = []
    message = request.get_json(silent=True)
    nSolution = np.asarray(message["solucoes"])
    nObj = np.asarray(message["objetivos"])
    #processar = message["processar"]
    #erro =message["erro"]
    
    
#    if processar == "treino":
#        tamanho = len(nSolution)
#        print(f'Tamanho: {tamanho}')
#        classifier = classifier.fit(nSolution, nObj);		
#    else:
#        ypredict = classifier.predict(nSolution)
#        lista = ypredict.tolist();
    classifier = classifier.fit(nSolution, nObj);	
    
    
    
    return json.dumps({"retorno": lista})




"""
######################################################### SALVAR #########################################################
"""
@app.route("/save", methods=['GET', 'POST'])
def save():
    global classifier
    global txErro
    
    message = request.get_json(silent=True)
    
    #salva o erro e o classificador
    joblib.dump(classifier, "TREE.pkl")
    EscreveArquivo(execult, real,'Real1k.txt')
    EscreveArquivo(execult, txErro,'Predito1k.txt')
    return json.dumps({"retorno": []})

"""
#################################### CLASSIFICA AS SOLUCOES ###############################################################
"""

@app.route("/classifica", methods=['GET', 'POST'])
def classifica():
    global classifier
    global txErro
    global real 
    global tamanho
    
    message = request.get_json(silent=True)
    nSolution = np.asarray(message["solucoes"]) #passa o numero de exemplos na posicao 0
    nObj = np.asarray(message["objetivos"]) #passa o numero de variaveis na posicao 0
    
    #classifier = joblib.load("TREE30k.pkl")    
    y_predict = classifier.predict(nSolution)
    
    i = 0
    
    for valor in y_predict:
        real.append(nObj[i])
        txErro.append(y_predict[i])
        execult.append(tamanho)
        tamanho += 1
        i+=1


    return json.dumps({"retorno": y_predict.tolist()})


"""
############################## INICIALIZA AS VARIAVEIS COM A FUNCAO LHS ####################################################
"""
@app.route("/inicializa", methods=['GET', 'POST'])
def Inicializa():
    
    message = request.get_json(silent=True)
    
    nSolution = np.asarray(message["solucoes"]) #passa o numero de exemplos na posicao 0
    nObj = np.asarray(message["objetivos"]) #passa o numero de variaveis na posicao 0
    
    retorno = lhs(nObj[0], samples=nSolution[0])
    retorno = np.asarray(retorno)


    return json.dumps({"retorno": retorno.tolist()})