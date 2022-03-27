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
import tensorflow as tf
import tensorflow_addons as tfa

from scipy.interpolate import griddata
import AuxFuncts as AF
import System as SEQ

import mlflow

import time

class PINN_Worker(Worker):
    
    def __init__(self,valData,test_gridObj,PDESystem,configspecs,valFromFEM=True,valParams={},
                 experiment_name='DEFAULT',eval_mode='functional',
                 *args,**kwargs):
        
        #Worker elements
        super().__init__(*args, **kwargs)
        print('Creating client')
        self.client = mlflow.tracking.MlflowClient()
        self.mlflow_id = kwargs['id']
        self.eval_mode = eval_mode
        
        #Check if experiment exists and get id. If not, create experiment and save id
        if experiment_name in [experiment.name for experiment in mlflow.list_experiments()]:
            print('Existing experiment. Recovering id.')
            experiment = mlflow.get_experiment_by_name(experiment_name)
            self.experiment_id = experiment.experiment_id
        else:
            print('New experiment. Creating instance.')
            self.experiment_id = mlflow.create_experiment(experiment_name)
        
        
        #Load validation data
        #Validation files should be in the same order as funcNames in System
        valuesSet = []
        pointsSet = []
        if valFromFEM:
            print('Parsing validation data')
            for dataFile in valData[0]:
                if type(dataFile) == str:
                    points, values = AF.preparadatos(dataFile,valData[1],test_gridObj.ndim)
                else:
                    print('Data is not string. Assuming data as 2-D numpy array')
                    points = dataFile[:,:-1]
                    values = dataFile[:,-1]
                valuesSet.append(values)
                pointsSet.append(points)
        else:
            for i in range(test_gridObj.ndim,valData.shape[1]):
                pointsSet.append(valData[:,:test_gridObj.ndim])
                valuesSet.append(valData[:,i])
        #Generar mallado y datos de test
        self.validation_preds = []
        print('Applying validation data to test grid')
        for values,points in zip(valuesSet,pointsSet):
            self.validation_preds.append(griddata(points, values, test_gridObj.grid , method='nearest'))
        self.test_gridObj = test_gridObj
        #Guardar objeto System
        print('Saving PDE and config')
        self.PDESystem = PDESystem
        self.configspecs = configspecs
        #
        if eval_mode == 'parameter':
            if len(valParams<1):
                print('WARNING: Param mode was chosen, but no param values were given')
            self.valParams = valParams
        
        print('Worker ready')
        
    def compute(self,config,budget,**kwargs):
        
        #Adding an implementation for multiple experiments could be useful
        run = self.client.create_run(self.experiment_id)
            
        train_gridObj = SEQ.Grid(self.test_gridObj.ndim, 
                                 self.test_gridObj.dimnames,
                                 self.test_gridObj.cubes, config['denspt'])
        
        activator_obj = tf.keras.activations.deserialize(config['activator'])
        Functionals, PDEs, variables, params = self.PDESystem.getPDEs(config['numNeurons'],config['numLayers'],activator_obj)
        m = sn.SciModel(list(variables.values()),PDEs,config['loss'],config['optimizer'])
        
        dimlist, data = self.PDESystem.evalSystem(train_gridObj)
        
        self.client.log_param(run_id=run.info.run_id,key='Density',value=config['denspt'])
        self.client.log_param(run_id=run.info.run_id,key='Num Neurons',value=config['numNeurons'])
        self.client.log_param(run_id=run.info.run_id,key='Num Layers',value=config['numLayers'])
        self.client.log_param(run_id=run.info.run_id,key='Activation Function',value=config['activator'])
        self.client.log_param(run_id=run.info.run_id,key='Loss Function',value=config['loss'])
        self.client.log_param(run_id=run.info.run_id,key='Optimizer',value=config['optimizer'])
        self.client.log_param(run_id=run.info.run_id,key='Initial LR',value=config['initial_lr'])
        self.client.log_param(run_id=run.info.run_id,key='Batch size',value=config['batch_size'])

        self.client.log_param(run_id=run.info.run_id,key='Num Points',value=train_gridObj.grid.shape[0])
        self.client.log_param(run_id=run.info.run_id,key='Epochs',value=int(budget))
        
        start = time.process_time()
        history = m.train(dimlist,data, epochs = int(budget),verbose=0,
                                batch_size=config['batch_size'],learning_rate=config['initial_lr'])        
        TrainTime = time.process_time()-start
        
        preds = []
        start2 = time.process_time()
        for functional in Functionals.values():
            pred = functional.eval(m,[self.test_gridObj.grid[:,i] for i in range(self.test_gridObj.grid.shape[1])])
            preds.append(pred)
        TestTime = time.process_time() - start2
        
        predErrors = []
        for pred,val in zip(preds,self.validation_preds) :
            predErrors.append(pred-val)
        
        L1 = self.test_gridObj.volume * np.sum([np.mean(np.abs(error)) for error in predErrors])
        
        L2 = self.test_gridObj.volume * np.sum([np.mean(error**2) for error in predErrors])
    
        MAX = np.max([np.abs(error).max() for error in predErrors])
        
        self.client.log_metric(run_id=run.info.run_id,key='L1',value=L1)
        self.client.log_metric(run_id=run.info.run_id,key='L2',value=L2)
        self.client.log_metric(run_id=run.info.run_id,key='MAX',value=MAX)
        self.client.log_metric(run_id=run.info.run_id,key='TrainTime',value=TrainTime)
        self.client.log_metric(run_id=run.info.run_id,key='TestTime',value=TestTime)
        
        if self.eval_mode == 'parameter':
            paramError = np.mean([abs(params[key].value-self.valParams[key]) for key in self.valParams.keys()])
            self.client.log_metric(run_id=run.info.run_id,key='ParamError',value=paramError)

        
        np.save('Predictions'+str(self.mlflow_id),np.array(preds),allow_pickle=True)
        self.client.log_artifact(run.info.run_id,'Predictions' + str(self.mlflow_id) + '.npy')
        
        np.save('History'+str(self.mlflow_id),history.history,allow_pickle=True)
        self.client.log_artifact(run.info.run_id,'History' + str(self.mlflow_id) + '.npy')
        
        self.client.set_terminated(run.info.run_id)
        
        if self.eval_mode == 'parameter':
            return ({
                'loss': paramError, # remember: HpBandSter always minimizes!
                'info': {'L1': L1,
                        'L2': L2,
                        'MAX': MAX,
                        'ParamError':paramError,
                        'TrainTime': TrainTime,
                    }})
        
        
        
        else:
            
            if np.isnan(L1):
                return ({
                    'loss': np.finfo(np.float).max, # remember: HpBandSter always minimizes!
                    'info': {	'L1': L1,
                                'L2': L2,
                                'MAX': MAX,
                                'TrainTime': TrainTime,
                            }})                
            else:
                return ({
                    'loss': L1, # remember: HpBandSter always minimizes!
                    'info': {	'L1': L1,
                                'L2': L2,
                                'MAX': MAX,
                                'TrainTime': TrainTime,
                            }})
                    
    

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
        
        initial_lr = CSH.UniformFloatHyperparameter(name='initial_lr',
                                                   lower = self.configspecs['initial_lr'][0],
                                                   upper = self.configspecs['initial_lr'][1],)
        
        config_space.add_hyperparameters([denspt,numNeurons,numLayers,initial_lr])

        
        activator = CSH.CategoricalHyperparameter(name = 'activator',
                                                  choices = self.configspecs['activator'])

        loss = CSH.CategoricalHyperparameter(name = 'loss',
                                                  choices = self.configspecs['loss'])

        optimizer = CSH.CategoricalHyperparameter(name = 'optimizer',
                                                  choices = self.configspecs['optimizer'])
        
        batch_size = CSH.CategoricalHyperparameter(name = 'batch_size',
                                                  choices = self.configspecs['batch_size'])
        
        config_space.add_hyperparameters([activator,loss,optimizer,batch_size])
        
        return(config_space)

    
