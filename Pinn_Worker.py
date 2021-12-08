# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 00:13:30 2021

@author: Felix
"""

from hpbandster.core.worker import Worker
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

import numpy as np

import sciann as sn

from scipy.interpolate import griddata
import AuxFuncts as AF
import System as SEQ

import mlflow

import time

class PINN_Worker(Worker):
    
    def __init__(self,valData,test_gridObj,PDESystem,configspecs):
        
        #Load validation data
        #Validation files should be in the same order as funcNames in System
        valuesSet = []
        for dataFile in valData[0]:
            points, values = AF.preparadatos(dataFile,valData[1])
            valuesSet.append(values)
        #Generar mallado y datos de test
        self.validation_preds = []
        for values in valuesSet:
            self.validation_preds.append(griddata(points, values, test_gridObj.grid , method='nearest'))
        self.test_gridObj = test_gridObj
        #Guardar objeto System
        self.PDESystem = PDESystem
        self.configspecs = configspecs
        
        
    def compute(self,config,budget,**kwargs):
        with mlflow.start_run():
            
            
            
            train_gridObj = SEQ.Grid(self.test_gridObj.ndim, 
                                     self.test_gridObj.dimnames,
                                     self.test_gridObj.cubes, config['denspt'])
            
            
            variables = self.PDESystem.getVariables()
            Functionals, PDEs = self.PDESystem.getPDEs(config['numNeurons'],config['numLayers'],config['activator'])
            m = sn.SciModel(variables,PDEs,config['loss'],config['optimizer'])
            
            dimlist, data = self.PDESystem.evalSystem(train_gridObj)
            
            mlflow.log_param('Density',config['denspt'])
            mlflow.log_param('Num Neurons',config['numNeurons'])
            mlflow.log_param('Num Layers',config['numLayers'])
            mlflow.log_param('Activation Function',config['activator'])
            mlflow.log_param('Loss Function',config['loss'])
            mlflow.log_param('Optimizer',config['optimizer'])
            mlflow.log_param('Num Points',train_gridObj.shape[0])
            
            
            start = time.process_time()
            history = m.train(dimlist,data, epochs = budget,verbose=0,
                                    batch_size=config['batch_size'])        
            TrainTime = time.process_time()-start
            
            preds = []
            start2 = time.process_time()
            for functional in Functionals:
                pred = functional.eval(m,[self.test_gridObj.grid[:,i] for i in range(self.test_gridObj.grid.shape[1])])
                preds.append(pred)
            TestTime = time.process_time() - start2
            
            predErrors = []
            for pred,val in zip(preds,self.validation_preds) :
                predErrors.append(pred-val)
            
            L1 = self.test_gridObj.volume * np.sum([np.mean(np.abs(error)) for error in predErrors])
            
            L2 = self.test_gridObj.volume * np.sum([np.mean(error**2)**1/2 for error in predErrors])
        
            MAX = np.max([np.abs(error).max() for error in predErrors])
            
            mlflow.log_metric('L1',L1)
            mlflow.log_metric('L2',L2)
            mlflow.log_metric('MAX',MAX)
            mlflow.log_metric('TrainTime',TrainTime)
            mlflow.log_metric('TestTime',TestTime)
            
            np.save('Predictions',np.array(preds),allow_pickle=True)
            mlflow.log_artifact('Predictions.npy')
            
            np.save('History',history.history,allow_pickle=True)
            mlflow.log_artifact('History.npy')
            
            

    def get_configspace(self):
        
        config_space = CS.ConfigurationSpace()
        
        denspt = CSH.UniformIntegerHyperparameter(name = 'denspt', 
                                                  lower=self.configspecs['denspt'][0],
                                                  upper=self.configspecs['denspt'][1])
        
        numNeurons = CSH.UniformIntegerHyperparameter(name = 'numNeurons', 
                                                  lower=self.configspecs['numNeurons'][0],
                                                  upper=self.configspecs['numNeurons'][1])
        
        numLayers = CSH.UniformIntegerHyperparameter(name = 'numLayers', 
                                                  lower=self.configspecs['numLayers'][0],
                                                  upper=self.configspecs['numLayers'][1])
        
        config_space.add_hyperparameter([denspt,numNeurons,numLayers])

        
        activator = CSH.CategoricalHyperparameter(name = 'activator',
                                                  choices = self.configspecs['activator'])

        loss = CSH.CategoricalHyperparameter(name = 'loss',
                                                  choices = self.configspecs['loss'])

        optimizer = CSH.CategoricalHyperparameter(name = 'optimizer',
                                                  choices = self.configspecs['optimizer'])
        
        batch_size = CSH.CategoricalHyperparameter(name = 'batch_size',
                                                  choices = self.configspecs['batch_size'])
        
        config_space.add_hyperparameter([activator,loss,optimizer,batch_size])
        
        return(config_space)
    
    