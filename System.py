# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 19:17:09 2021

@author: Felix
"""

import numpy as np
import sciann as sn
from sciann import Functional
from sciann import Parameter
from sciann import Variable
from sciann.utils.math import diff, sign, sin, exp



class PDESystem:
    
    def __init__(self,variableNames,funcNames,paramSpecs,constants):
        self.variables = self.createVariables(variableNames)
        self.funcNames = funcNames
        self.equations = []
        self.functionals = {}
        self.constants = constants
        self.parameters = self.createParams(paramSpecs)
        
    def addEquation(self,equation):
        if isinstance(equation, Equation):
            self.equations.append(equation)
        else:
            print('This object is not an Equation')
     
    def createFunctionals(self,numNeurons,numLayers,activation):
        for funcName in self.funcNames.keys():
            self.functionals[funcName] = Functional(funcName,
                                                    list(self.variables.values()),
                                                    numLayers*[numNeurons],
                                                    activation)
            
        
    def evalSystem(self,gridObj):
        data = []
        for equation in self.equations:
            data.append(equation.getValues(gridObj,self.constants))
        
        dimList = []
        for i in range(gridObj.grid.shape[1]):
            dimList.append(gridObj.grid[:,i][:,None])
        return dimList,data
    
    def getPDEs(self,numNeurons,numLayers,activation):
        self.createFunctionals(numNeurons,numLayers,activation)
        elementDict = self.collapseItemDictionaries()
        PDEs = [equation.execPDE(elementDict) for equation in self.equations]
        functionals = self.functionals
        self.functionals = {}
        return functionals, PDEs
    
    def collapseItemDictionaries(self):
        elementDict = {**self.variables,**self.constants,**self.parameters,**self.functionals}
        return elementDict
    
    def createVariables(self,variableNames):
        variableDict = {varname:Variable(varname) for varname in variableNames}
        return variableDict
    
    def createParams(self,paramSpecs):
        params = {spec['name']:Parameter(**spec,inputs=self.variables) for spec in paramSpecs}
        return params
        
    def getVariables(self):
        return list(self.variables.values())

    

class Equation:
    
    def __init__(self,constraints,valuesFunc,PDE):
        self.constraints = constraints
        self.values = valuesFunc
        self.PDE = PDE
    
    #meter cttes
    def evalConstraints(self,gridObj,constants):
        currInd = np.arange(gridObj.grid.shape[0])
        for constraint in self.constraints:
            currInd = np.intersect1d(currInd,constraint.evalOverGrid(gridObj,constants))
        return currInd
    
    #meter cttes
    def getValues(self,gridObj,constants):
        indexes = self.evalConstraints(gridObj,constants)
        if type(self.values) == str:
            values = self.evalOverGrid(gridObj,indexes,constants)
            return((indexes[:,None],values[:,None]))
        else:
            return((indexes[:,None],self.values))
            
    #meter cttes #hecho
    def evalOverGrid(self,gridObj,indexes,constants):
        preparedFUN = self.prepareValuesFunc(gridObj.dimnames,constants)
        evaluationSentence = 'values = ' + preparedFUN
        exec(evaluationSentence) 
        return locals()['values']
        
    #meter cttes #hecho
    def prepareValuesFunc(self,dimnames,constants):
        splittedFUN = self.values.split('#')
        splittedFUN = [str(constants.get(k,k)) for k in splittedFUN]
        newDimnames = {k:'gridObj.grid[indexes,'+str(v)+']' for (k,v) in dimnames.items()}
        preparedFUN = ''.join([str(newDimnames.get(k,k)) for k in splittedFUN])
        return preparedFUN
    

    def execPDE(self,elementDict):
        preparedPDE = self.preparePDE(elementDict)
        pdeSentence = 'PDE = sn.constraints.Data(' + preparedPDE + ')'
        exec(pdeSentence)
        return locals()['PDE']
        
      
    def preparePDE(self,elementDict):
        splittedPDE = self.PDE.split('#')
        newElementNames = {k:'elementDict["'+k+'"]' for k in elementDict.keys()}
        preparedPDE = ''.join([str(newElementNames.get(k,k)) for k in splittedPDE])
        return preparedPDE
    
        
class Constraint:
    def __init__(self,FUN,comparator,CTT):
        self.FUN = FUN
        self.comparator = comparator
        self.CTT = CTT
        
    def evalOverGrid(self,gridObj,constants):
        preparedFUN = self.prepareFUN(gridObj.dimnames,constants)
        evaluationSentence = 'indexes = np.where(' +preparedFUN + self.comparator + str(self.CTT) + ')'
        exec(evaluationSentence) 
        return locals()['indexes'][0]
        
         
    def prepareFUN(self,dimnames,constants):
        splittedFUN = self.FUN.split('#')
        splittedFUN = [str(constants.get(k,k)) for k in splittedFUN]
        newDimnames = {k:'gridObj.grid[:,'+str(v)+']' for (k,v) in dimnames.items()}
        preparedFUN = ''.join([str(newDimnames.get(k,k)) for k in splittedFUN])
        return preparedFUN
    
        
        
class Grid:

    def __init__(self,ndim,dimnames,cubes,denspt):
        self.ndim = ndim
        self.dimnames = dimnames
        pointSet = []
        for cube in cubes:
            pointSet.append(self.buildCube(ndim,cube,denspt))
        self.grid = np.concatenate(pointSet)
        self.cubes = cubes
        self.volume = self.calculateVolume()
        
    def buildCube(self,ndim,lims,denspt):
        linpt = []
        for dim in range(ndim):
            span = lims[dim][1]-lims[dim][0]
            linpt.append(np.linspace(lims[dim][0],lims[dim][1],int(span*denspt)))
        grid = np.meshgrid(*linpt)
        grid = self.flattenGrid(grid)
        return grid
            
    def flattenGrid(self,grid):
        flattenedGrid = []
        for dim in grid:
            flattenedGrid.append(dim.flatten())
            
        flattenedGrid = np.array(flattenedGrid).transpose()
        return flattenedGrid
    
    def calculateVolume(self):
        volume = 0
        for cube in self.cubes:
            spans = np.array([dimspan[1]-dimspan[0] for dimspan in cube])
            volume += np.product(spans)
        return volume