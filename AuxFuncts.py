# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 00:05:05 2021

@author: Felix
"""

from scipy.interpolate import griddata
import numpy as np

def lee_u(u_file):
    values = []
    times = []
    fd = open(u_file,'r')
    for tupla in fd.read().split('\n\n'):
        if tupla != '':
            tiempo, datos = tupla.split('\n')
            times.append(float(tiempo))
            vector = []
            for value in datos.split('\t'):
                    if value != '':
                        vector.append(float(value))
            values.append(vector)
    values = np.array(values)
    fd.close()
    return times, values

def lee_mesh(mesh_file,ndim):
    fd = open(mesh_file)
    for batch in fd.read().split('\n\n'):
        if batch[0:8]=='Vertices':
            vertices = batch
    fd.close()
    coordenadas = []

    for vertice in vertices.split('\n'):
        if len(vertice.split(' ')) > 1:
            coordenadas.append([float(vertice.split(' ')[i]) for i in range(ndim-1)])
    coordenadas = np.array(coordenadas)
    return coordenadas

def preparadatos(u_file,mesh_file,ndim):
    datos = lee_u(u_file)
    mesh = lee_mesh(mesh_file,ndim)
    
    sec = datos[0][0]
    fullmesh = np.insert(mesh,ndim-1,sec,axis=1)
    
    for i in range(1,len(datos[0])):
        sec = datos[0][i]
        fullmesh = np.append(fullmesh,np.insert(mesh,ndim-1,sec,axis=1),axis=0)
    
    valores = datos[1].flatten()
    
    return fullmesh,valores

