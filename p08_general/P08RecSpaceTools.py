# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 10:49:00 2019

@author: Florian
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

import matplotlib.pyplot as plt

import cProfile

import fio_reader

import h5py

import glob

import fabio

import sys, traceback




class QSpace(object):

    def __init__(self, lattice, energy, ubmatrix):
        
        self.lattice = lattice
        self.energy = energy

        self.ubmatrix = ubmatrix


    def calcQ(self, om, omh, chi, phis, tt, tth):
        '''
        calculates Q space position from angular positions using:
        H. You J. Appl. Cryst. (1999). 32, 614-623
        Angle calculations for a '4S+2D' six-circle diffractometer


        '''

        # eta   = om
        # mu    = omh 
        # chi   = chi
        # phi   = phis
        # delta = tt
        # nu    = tth

        wavelength = 12380/self.energy

        k_val = 2*numpy.pi/wavelength
        
        
        om_rad   = om * numpy.pi/180.
        omh_rad  = omh * numpy.pi/180.
        chi_rad  = chi * numpy.pi/180.
        phis_rad = phis * numpy.pi/180.
        tt_rad   = tt * numpy.pi/180.
        tth_rad  = tth * numpy.pi/180.

        A = numpy.sin(om_rad) * (numpy.cos(omh_rad) - numpy.cos(omh_rad)*numpy.cos(tt_rad)*numpy.cos(tth_rad) - numpy.sin(omh_rad)*numpy.cos(tt_rad)*numpy.sin(tth_rad) ) + numpy.cos(om_rad) * numpy.sin(tt_rad)

        B = numpy.cos(om_rad) * (numpy.cos(omh_rad) - numpy.cos(omh_rad)*numpy.cos(tt_rad)*numpy.cos(tth_rad) - numpy.sin(omh_rad)*numpy.cos(tt_rad)*numpy.sin(tth_rad) ) - numpy.sin(om_rad) * numpy.sin(tt_rad)

        C = numpy.cos(omh_rad)*numpy.cos(tt_rad)*numpy.sin(tth_rad) - numpy.sin(omh_rad)*numpy.cos(tt_rad)*numpy.cos(tth_rad) + numpy.sin(omh_rad)


        qx = A*numpy.cos(phis_rad)*numpy.cos(chi_rad) + B*numpy.sin(phis_rad) - C*numpy.cos(phis_rad)*numpy.sin(chi_rad)
        qy = A*numpy.sin(phis_rad)*numpy.cos(chi_rad) - B*numpy.cos(phis_rad) - C*numpy.sin(phis_rad)*numpy.sin(chi_rad)
        qz = A*numpy.sin(chi_rad) + C*numpy.cos(chi_rad)

        qx *= k_val
        qy *= k_val
        qz *= k_val

        return [qx, qy, qz]
        

    def calc_HKL(self, qx, qy, qz):
        '''
        takes qx,y,z positions and converts it to hkl using the ub matrix


        ub matrix has to be given according to the spock definitions.
        it will be reshuffeld internally to work acoording to definitions by You.

        '''
        

        matrix = [self.ubmatrix[2],self.ubmatrix[0],self.ubmatrix[1]]

        inv_ub = numpy.linalg.inv(matrix)

        h = inv_ub[0,0]*qx + inv_ub[0,1]*qy + inv_ub[0,2]*qz
        k = inv_ub[1,0]*qx + inv_ub[1,1]*qy + inv_ub[1,2]*qz
        l = inv_ub[2,0]*qx + inv_ub[2,1]*qy + inv_ub[2,2]*qz

        return [h, k, l]

















