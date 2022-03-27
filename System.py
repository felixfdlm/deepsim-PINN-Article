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
import re


class PDESystem:
    
    def __init__(self,variableNames,funcNames,paramSpecs,constants):
        self.variableNames = variableNames
        self.funcNames = funcNames
        self.equations = []
        self.functionals = {}
        self.constants = constants
        self.parameterSpecs = paramSpecs
        
    def addEquation(self,equation):
        if isinstance(equation, Equation):
            self.equations.append(equation)
        else:
            print('This object is not an Equation')
     
    def createFunctionals(self,numNeurons,numLayers,activation,variables,parameters):
        #Initialize empty functional dictionary
        functionals = {}
        #First create base functionals (the ones defined by numbers).
        for funcName in [key for key in self.funcNames.keys() if type(self.funcNames[key])==int]:
            functionals[funcName] = Functional(funcName,
                                                    list(variables.values()),
                                                    numLayers*[numNeurons],
                                                    activation)
            
        #Then create the functionals derived from base ones
        for funcName in [key for key in self.funcNames.keys() if type(self.funcNames[key])==str]:
            sentence = self.funcNames[funcName]
            elementDict = self.collapseItemDictionaries(variables,functionals,parameters)
            preparedFunctionalSentence = self.prepareFunctionalSentence(sentence,elementDict)
            
            exec(funcName + '=' + preparedFunctionalSentence)
            functionals[funcName] = locals()[funcName]
        return functionals
            
        
    def evalSystem(self,gridObj):
        data = []
        for equation in self.equations:
            data.append(equation.getValues(gridObj,self.constants))
        
        dimList = []
        for i in range(gridObj.grid.shape[1]):
            dimList.append(gridObj.grid[:,i][:,None])
        return dimList,data
    
    def getPDEs(self,numNeurons,numLayers,activation):
        variables = self.createVariables(self.variableNames)
        parameters = self.createParams(variables)
        functionals = self.createFunctionals(numNeurons,numLayers,activation,variables,parameters)
        elementDict = self.collapseItemDictionaries(variables,functionals,parameters)
        PDEs = [equation.execPDE(elementDict) for equation in self.equations]
        return functionals, PDEs, variables, parameters
    
    def collapseItemDictionaries(self,variables,functionals,parameters):
        elementDict = {**variables,**self.constants,**parameters,**functionals}
        return elementDict
    
    def createVariables(self,variableNames):
        variableDict = {varname:Variable(varname) for varname in variableNames}
        return variableDict
    
    def createParams(self,variables):
        params = {paramName:Parameter(inputs=list(variables.values()),**specs) for paramName,specs in self.parameterSpecs.items()}
        return params


    def prepareFunctionalSentence(self,sentence,elementDict):
        splittedFunctional = sentence.split('#')
        newElementNames = {k:'elementDict["'+k+'"]' for k in elementDict.keys()}
        preparedFunctional = ''.join([str(newElementNames.get(k,k)) for k in splittedFunctional])
        return preparedFunctional

    

class Equation:
    
    def __init__(self,constraints,valuesFunc,PDE):
        self.constraints = constraints[0]
        self.formula = constraints[1]
        self.values = valuesFunc
        self.PDE = PDE


    ###########################################################################
    #Functions related to applying the equation to the grid

    def getValues(self,gridObj,constants):
        
        setDict = self.evalConstraints(gridObj, constants)
        
        indexes = self.applyFormula(setDict)

        indexes = np.array(list(indexes))

        if type(self.values) == str:
            values = self.evalOverGrid(gridObj,indexes,constants)
            return((indexes[:,None],values[:,None]))
        else:
            return((indexes[:,None],self.values))

    
    def evalConstraints(self,gridObj,constants):
        setDict = {k:v.evalOverGrid(gridObj,constants) for (k,v) in self.constraints.items()}        
        setDict['Om'] = set(np.arange(gridObj.grid.shape[0]))
       
        return setDict
    
    
    def applyFormula(self,setDict):
        
        levels,operations,elementOrder = self.parseFormula(self.formula,list(setDict.keys()))
        elementOrder = [setDict.get(k) for k in elementOrder]
        
        while len(elementOrder)>1:
            index = levels.index(max(levels))
            sets = [elementOrder[index],elementOrder[index+1]]
            operation = operations[index]
            if operation == 'n':
                newSet = sets[0].intersection(sets[1])
            if operation == 'u':
                newSet = sets[0].union(sets[1])
                
            del levels[index]
            del operations[index]
            del elementOrder[index+1]
            elementOrder[index] = newSet
        return elementOrder[0]
        
    
    def parseFormula(self,formula,elementKeys):
        
        #This function receives a set operation formula such as AnBu(CnD) where sets
        #are always written in MAYUS and "n" means instersection and "u" means union.
        
        ops = formula
        for key in elementKeys:
            ops = ops.replace(key,'')
            
        levels = []
        operations = []
        level = 0
        for char in ops:
            if char in ('n','u'):
                operations.append(char)
                levels.append(level)
            if char == '(':
                level +=1
            if char == ')':
                level -=1
                
        elementOrder = re.sub('[() ]','', formula)
        elementOrder = re.split('n|u',elementOrder)
        
        return levels, operations, elementOrder

            
    def evalOverGrid(self,gridObj,indexes,constants):
        preparedFUN = self.prepareValuesFunc(gridObj.dimnames,constants)
        evaluationSentence = 'values = ' + preparedFUN
        exec(evaluationSentence) 
        return locals()['values']
        
    def prepareValuesFunc(self,dimnames,constants):
        splittedFUN = self.values.split('#')
        splittedFUN = [str(constants.get(k,k)) for k in splittedFUN]
        newDimnames = {k:'gridObj.grid[indexes,'+str(v)+']' for (k,v) in dimnames.items()}
        preparedFUN = ''.join([str(newDimnames.get(k,k)) for k in splittedFUN])
        return preparedFUN
    
    ###########################################################################
    #Functions related to applying PDE to the functional

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
        return set(locals()['indexes'][0])
        
         
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