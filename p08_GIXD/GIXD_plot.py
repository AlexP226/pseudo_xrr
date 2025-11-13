#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# filename: GIXD_plot.py
"""
plotting tools for GIXD data
modified 2021 June
@author: shenc
version 1.1
require list with at least 'mat', 'Qxy', 'Qz' fields
proper color plot gap at Qxy = 0 for GISAXS
display channel on horizontal if needed
"""

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np


def plotqzgroup(inputmap, **kwargs):
    """
    line plot of grouped in Qz, offset
    kwargs: 
        range_axis = [[x1, x2], [y1, y2]]
        range_I = [Imin, Imax]
        Ngroup = 10
        offset = 0.2
    """
    # figure parameters:
    fontsize = 22
    margin = {'bottom': 0.15, 'top': 0.9, 'left': 0.2, 'right' :0.95}
    size = {'width': 150, 'height': 150, 'dpi':150}
    
    # set plot axis and range
    plot_setting = {    'range_axis': [],
                        'range_I': [],
                        'Ngroup': 10,
                        'offset': 0.08,
                        'title': []
                        }
    plot_setting.update(kwargs)

    if not plot_setting['range_axis']:
        Qxystart = np.min(inputmap['Qxy'][0,:])
        Qxyend = np.max(inputmap['Qxy'][0,:])
        Qzstart = 0
        Qzend = np.max(inputmap['Qz'])
    else:
        Qxystart = np.min(plot_setting['range_axis'][0])
        Qxyend = np.max(plot_setting['range_axis'][0])
        Qzstart = np.min(plot_setting['range_axis'][1])
        Qzend = np.max(plot_setting['range_axis'][1])
        
    # create dictionary for the map
    # 1. Qz range and group
    tmpmap = {'Qz_idx': []}
    tmpmap['Qz_idx'] = np.where(np.logical_and(inputmap['Qz']>=Qzstart, inputmap['Qz']<=Qzend))[0]
    tmpmap['mat'] = inputmap['mat'][tmpmap['Qz_idx'][0]:tmpmap['Qz_idx'][-1]+1,:]
    tmpmap['Qz'] = inputmap['Qz'][tmpmap['Qz_idx'][0]:tmpmap['Qz_idx'][-1]+1]
    if inputmap['Qxy'].shape[0]>1:    
        tmpmap['Qxy'] = inputmap['Qxy'][tmpmap['Qz_idx'][0]:tmpmap['Qz_idx'][-1]+1,:]
    else:
        tmpmap['Qxy'] = np.zeros([tmpmap['mat'].shape[0], tmpmap['mat'].shape[1]])
        for i in range(tmpmap['mat'].shape[0]):
            tmpmap['Qxy'][i,:] = inputmap['Qxy'][0]
            
    del inputmap

    Ngroup = int(plot_setting['Ngroup'])    
    group_size = int(np.floor(tmpmap['mat'].shape[0]/Ngroup))
    plotmap = {'Qxy': np.zeros([Ngroup, tmpmap['mat'].shape[1]]), 'Qz': np.zeros([Ngroup,1]), 'mat': np.zeros([Ngroup, tmpmap['mat'].shape[1]])}
    
    for i in range(Ngroup):
        plotmap['mat'][i,:] = np.sum(tmpmap['mat'][i*group_size:(i+1)*group_size,:],axis = 0)
        plotmap['Qxy'][i,:] = np.mean(tmpmap['Qxy'][i*group_size:(i+1)*group_size,:],axis = 0)
        plotmap['Qz'][i,0] = np.mean(tmpmap['Qz'][i*group_size:(i+1)*group_size],axis = 0)

    # setup offset    
    if not plot_setting['range_I']:
        ystart = 0
        yend = np.amax(plotmap['mat'])*plot_setting['offset']*(Ngroup+1)
        I_offset = np.amax(plotmap['mat'])*plot_setting['offset']
    else:
        ystart = plot_setting['range_I'][0]
        yend = plot_setting['range_I'][1]
        I_offset = (yend-ystart) * plot_setting['offset']

    # figure
    fig, ax = plt.subplots()
    for i in range(Ngroup):
        ax.plot(plotmap['Qxy'][i,:], plotmap['mat'][i,:]+I_offset*i, linewidth = 1.5, label="$%.2f\/\AA^{-1}$"%plotmap['Qz'][i])
    
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1], title = '$Q_z$', title_fontsize = fontsize*0.6, loc='upper right', fontsize = fontsize*0.6)
            
    plt.xlabel(r'$Q_{xy}\/[\AA^{-1}]$',fontsize=fontsize)
    Qxyaxis_step = 0.1
    yaxis_step = (10**np.floor(np.log10(yend-ystart)))/5
    plt.ylabel('intensity',fontsize=fontsize)
    
    plt.xlim(Qxystart, Qxyend+0.1)
    plt.ylim(ystart, yend)
    plt.xticks(np.arange(np.ceil(Qxystart/Qxyaxis_step)*Qxyaxis_step, (np.ceil(Qxyend/Qxyaxis_step)+1)*Qxyaxis_step, step=Qxyaxis_step), fontsize=fontsize)
    plt.yticks(np.arange(np.ceil(ystart/yaxis_step)*yaxis_step, (np.ceil(yend/yaxis_step)+1)*yaxis_step, step=yaxis_step), fontsize=fontsize)

    for key in ax.spines:
        ax.spines[key].set_lw(1.5)
    ax.tick_params(width = 1.5)
        # title
    if plot_setting['title']:
        plt.title(plot_setting['title'])
    fig.subplots_adjust(bottom = margin['bottom'],top = margin['top'], left=margin['left'],right = margin['right'])
    fig.set_size_inches(size['width']/25.4, size['height']/25.4)
    fig.dpi=size['dpi']
    
    return fig
    
def plotqzgroupN(*args, **kwargs):
    """
    comparison of N inputmaps
    args:
        inputmap1, inputmap2, etc
    line plot of grouped in Qz, offset
    kwargs: 
        range_axis = [[x1, x2], [y1, y2]]
        range_I = [Imin, Imax]
        Ngroup = 10
        offset = 0.2
        data0 = 'marker' or line
        title = or not
    """
    # figure parameters:
    fontsize = 22
    margin = {'bottom': 0.15, 'top': 0.9, 'left': 0.2, 'right' :0.95}
    size = {'width': 150, 'height': 150, 'dpi':150}
    
    # set plot axis and range
    plot_setting = {    'range_axis': [],
                        'range_I': [],
                        'Ngroup': 10,
                        'offset': 0.08,        
                        'data0' :'line',
                        'title': []
                        }
    plot_setting.update(kwargs)

    if not plot_setting['range_axis']:
        Qxystart = np.min(args[0]['Qxy'][0,:])
        Qxyend = np.max(args[0]['Qxy'][0,:])
        Qzstart = 0
        Qzend = np.max(args[0]['Qz'])
    else:
        Qxystart = np.min(plot_setting['range_axis'][0])
        Qxyend = np.max(plot_setting['range_axis'][0])
        Qzstart = np.min(plot_setting['range_axis'][1])
        Qzend = np.max(plot_setting['range_axis'][1])

    # create dictionary for the map
    # 1. Qz range
    tmpmap = []
    for k in range(len(args)):
        tmpmap.append({'Qz_idx': []})
        tmpmap[k]['Qz_idx'] = np.where(np.logical_and(args[k]['Qz']>=Qzstart, args[k]['Qz']<=Qzend))[0]
        tmpmap[k]['mat'] = args[k]['mat'][tmpmap[k]['Qz_idx'][0]:tmpmap[k]['Qz_idx'][-1]+1,:]
        tmpmap[k]['Qz'] = args[k]['Qz'][tmpmap[k]['Qz_idx'][0]:tmpmap[k]['Qz_idx'][-1]+1]
        if args[k]['Qxy'].shape[0]>1:
            tmpmap[k]['Qxy'] = args[k]['Qxy'][tmpmap[k]['Qz_idx'][0]:tmpmap[k]['Qz_idx'][-1]+1,:]
        else:
            tmpmap[k]['Qxy'] = np.zeros([tmpmap[k]['mat'].shape[0], tmpmap[k]['mat'].shape[1]])
            for i in range(tmpmap[k]['mat'].shape[0]):
                tmpmap[k]['Qxy'][i,:] = args[k]['Qxy']        

    # 2. grouping
    Ngroup = int(plot_setting['Ngroup'])    
    group_size = int(np.floor(tmpmap[0]['mat'].shape[0]/Ngroup))
    
    plotmap = []
    for k in range(len(args)):
        plotmap.append({'Qxy':[]})
        plotmap[k] = {'Qxy': np.zeros([Ngroup, tmpmap[k]['mat'].shape[1]]), 'Qz': np.zeros([Ngroup,1]), 'mat': np.zeros([Ngroup, tmpmap[k]['mat'].shape[1]])}
        for i in range(Ngroup):
            plotmap[k]['mat'][i,:] = np.sum(tmpmap[k]['mat'][i*group_size:(i+1)*group_size,:],axis = 0)
            plotmap[k]['Qxy'][i,:] = np.mean(tmpmap[k]['Qxy'][i*group_size:(i+1)*group_size,:],axis = 0)
            plotmap[k]['Qz'][i,0] = np.mean(tmpmap[k]['Qz'][i*group_size:(i+1)*group_size],axis = 0)

    # setup offset    
    if not plot_setting['range_I']:
        ystart = 0
        yend = np.amax(plotmap[0]['mat'])*plot_setting['offset']*(Ngroup+1)
        I_offset = np.amax(plotmap[0]['mat'])*plot_setting['offset']
    else:
        ystart = plot_setting['range_I'][0]
        yend = plot_setting['range_I'][1]
        I_offset = (yend-ystart) * plot_setting['offset']

    # figure
    fig, ax = plt.subplots()
    for i in range(Ngroup):
        if plot_setting['data0'] == 'marker':
            ax.plot(plotmap[0]['Qxy'][i,:], plotmap[0]['mat'][i,:]+I_offset*i, marker = "o",markeredgecolor = 'black',fillstyle = "none", markersize = 5, linestyle = 'none',label="$%.2f\/\AA^{-1}$"%plotmap[0]['Qz'][i])
        else:
            ax.plot(plotmap[0]['Qxy'][i,:], plotmap[0]['mat'][i,:]+I_offset*i, color = 'black', linewidth = 3, label="$%.2f\/\AA^{-1}$"%plotmap[0]['Qz'][i])
 
        for k in range(len(plotmap)-1):
            ax.plot(plotmap[k+1]['Qxy'][i,:], plotmap[k+1]['mat'][i,:]+I_offset*(i-0.02*(k+1)), linewidth = 1.5)
        
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1], title = '$Q_z$', title_fontsize = fontsize*0.6, loc='upper right', fontsize = fontsize*0.6)
            
    plt.xlabel(r'$Q_{xy}\/[\AA^{-1}]$',fontsize=fontsize)
    Qxyaxis_step = 0.2
    yaxis_step = (10**np.floor(np.log10(yend-ystart)))/2
    plt.ylabel('intensity',fontsize=fontsize)
    
    plt.xlim(Qxystart, Qxyend+0.1)
    plt.ylim(ystart, yend)
    plt.xticks(np.arange(np.ceil(Qxystart/Qxyaxis_step)*Qxyaxis_step, (np.ceil(Qxyend/Qxyaxis_step)+1)*Qxyaxis_step, step=Qxyaxis_step), fontsize=fontsize)
    plt.yticks(np.arange(np.ceil(ystart/yaxis_step)*yaxis_step, (np.ceil(yend/yaxis_step)+1)*yaxis_step, step=yaxis_step), fontsize=fontsize)
    
    ax.ticklabel_format(scilimits=(0,0))
    # title
    if plot_setting['title']:
        plt.title(plot_setting['title'])
    plt.rc('font', size = fontsize*0.8)
    for key in ax.spines:
        ax.spines[key].set_lw(1.5)
    ax.tick_params(width = 1.5)
    fig.subplots_adjust(bottom = margin['bottom'],top = margin['top'], left=margin['left'],right = margin['right'])
    fig.set_size_inches(size['width']/25.4, size['height']/25.4)
    fig.dpi=size['dpi']
    
    return fig
    
def plotqzN(*args, **kwargs):
    """
    comparison of N inputmaps
    args:
        inputmap1, inputmap2, etc
    line plot of grouped in Qz, offset
    kwargs: 
        range_axis = [[x1, x2], [y1, y2]]
        range_I = [Imin, Imax]
        Nqz = 10
        offset = 0.2
        data0 = 'marker' or line
        title = or nothing
    """
    # figure parameters:
    fontsize = 22
    margin = {'bottom': 0.15, 'top': 0.9, 'left': 0.2, 'right' :0.95}
    size = {'width': 200, 'height': 150, 'dpi':300}
    
    # set plot axis and range
    plot_setting = {    'range_axis': [],
                        'range_I': [],
                        'Nqz': 10,
                        'offset': 0.08,
                        'data0':'line',
                        'title': []
                        }
    plot_setting.update(kwargs)

    if not plot_setting['range_axis']:
        Qxystart = np.min(args[0]['Qxy'][0,:])
        Qxyend = np.max(args[0]['Qxy'][0,:])
        Qzstart = 0
        Qzend = np.max(args[0]['Qz'])
    else:
        Qxystart = np.min(plot_setting['range_axis'][0])
        Qxyend = np.max(plot_setting['range_axis'][0])
        Qzstart = np.min(plot_setting['range_axis'][1])
        Qzend = np.max(plot_setting['range_axis'][1])

    # create dictionary for the map
    # 1. Qz range
    tmpmap = []
    for k in range(len(args)):
        tmpmap.append({'Qz_idx': []})
        tmpmap[k]['Qz_idx'] = np.where(np.logical_and(args[k]['Qz']>=Qzstart, args[k]['Qz']<=Qzend))[0]
        tmpmap[k]['mat'] = args[k]['mat'][tmpmap[k]['Qz_idx'][0]:tmpmap[k]['Qz_idx'][-1]+1,:]
        tmpmap[k]['Qz'] = args[k]['Qz'][tmpmap[k]['Qz_idx'][0]:tmpmap[k]['Qz_idx'][-1]+1]
        if args[k]['Qxy'].shape[0]>1:
            tmpmap[k]['Qxy'] = args[k]['Qxy'][tmpmap[k]['Qz_idx'][0]:tmpmap[k]['Qz_idx'][-1]+1,:]
        else:
            tmpmap[k]['Qxy'] = np.zeros([tmpmap[k]['mat'].shape[0], tmpmap[k]['mat'].shape[1]])
            for i in range(tmpmap[k]['mat'].shape[0]):
                tmpmap[k]['Qxy'][i,:] = args[k]['Qxy']        

    # 2. grouping
    Nqz = int(plot_setting['Nqz'])    
    group_size = int(np.floor(tmpmap[0]['mat'].shape[0]/Nqz))
    
    plotmap = []
    for k in range(len(args)):
        plotmap.append({'Qxy':[]})
        plotmap[k] = {'Qxy': np.zeros([Nqz, tmpmap[k]['mat'].shape[1]]), 'Qz': np.zeros([Nqz,1]), 'mat': np.zeros([Nqz, tmpmap[k]['mat'].shape[1]])}
        for i in range(Nqz):
            plotmap[k]['mat'][i,:] = tmpmap[k]['mat'][i*group_size,:]
            plotmap[k]['Qxy'][i,:] = tmpmap[k]['Qxy'][i*group_size,:]
            plotmap[k]['Qz'][i,0] = tmpmap[k]['Qz'][i*group_size]

    # setup offset    
    if not plot_setting['range_I']:
        ystart = 0
        yend = np.amax(plotmap[0]['mat'])*plot_setting['offset']*(Nqz+1)
        I_offset = np.amax(plotmap[0]['mat'])*plot_setting['offset']
    else:
        ystart = plot_setting['range_I'][0]
        yend = plot_setting['range_I'][1]
        I_offset = (yend-ystart) * plot_setting['offset']

    # figure
    fig, ax = plt.subplots()
    for i in range(Nqz):
        if plot_setting['data0'] == 'marker':
            ax.plot(plotmap[0]['Qxy'][i,:], plotmap[0]['mat'][i,:]+I_offset*i, marker = "o",markeredgecolor = 'black',fillstyle = "none", markersize = 5, linestyle = 'none',label="$%.2f\/\AA^{-1}$"%plotmap[0]['Qz'][i])
        else:
            ax.plot(plotmap[0]['Qxy'][i,:], plotmap[0]['mat'][i,:]+I_offset*i, color = 'black', linewidth = 3, label="$%.2f\/\AA^{-1}$"%plotmap[0]['Qz'][i])
            
        for k in range(len(plotmap)-1):
            ax.plot(plotmap[k+1]['Qxy'][i,:], plotmap[k+1]['mat'][i,:]+I_offset*(i-0.02*(k+1)), linewidth = 1.5)
        
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1], title = '$Q_z$', title_fontsize = fontsize*0.6, bbox_to_anchor=(1, 1), loc='upper right', fontsize = fontsize*0.6)
                 
    plt.xlabel(r'$Q_{xy}\/[\AA^{-1}]$',fontsize=fontsize)
    Qxyaxis_step = 0.2
    yaxis_step = (10**np.floor(np.log10(yend-ystart)))/2
    plt.ylabel('intensity',fontsize=fontsize)
    
    plt.xlim(Qxystart, Qxyend+0.1)
    plt.ylim(ystart, yend)
    plt.xticks(np.arange(np.ceil(Qxystart/Qxyaxis_step)*Qxyaxis_step, (np.ceil(Qxyend/Qxyaxis_step)+1)*Qxyaxis_step, step=Qxyaxis_step), fontsize=fontsize)
    plt.yticks(np.arange(np.ceil(ystart/yaxis_step)*yaxis_step, (np.ceil(yend/yaxis_step)+1)*yaxis_step, step=yaxis_step), fontsize=fontsize)
    
    ax.ticklabel_format(scilimits=(0,0))
    # title
    if plot_setting['title']:
        plt.title(plot_setting['title'])
    plt.rc('font', size = fontsize*0.8)
    for key in ax.spines:
        ax.spines[key].set_lw(1.5)
    ax.tick_params(width = 1.5)
    fig.subplots_adjust(bottom = margin['bottom'],top = margin['top'], left=margin['left'],right = margin['right'])
    fig.set_size_inches(size['width']/25.4, size['height']/25.4)
    fig.dpi=size['dpi']
    
    return fig



def plot2d(inputmap, **kwargs):
    """
    plot 2D color map
    kwargs: 
        axis = ['xname' , 'yname'], xname is tth or Qxy, yname is mych, tt or Qz
        range_axis = [[x1, x2], [y1, y2]]
        range_I = [Imin, Imax]
    """
    # figure parameters:
    fontsize = 22
    margin = {'bottom': 0.18, 'top': 0.9, 'left': 0.2, 'right' :0.90}
    size = {'width': 150, 'height': 150, 'dpi':150}
    
    # set plot axis and range
    plot_setting = {    'axis':    ['tth', 'mych'],            
                        'range_axis': [],
                        'range_I': [1, 10**2.5]}
    plot_setting.update(kwargs)
    # setup NaN at tth 0 as an additional column
    plotmap = {plot_setting['axis'][0]: [], plot_setting['axis'][1]: [],'mat': []}
    # setup NaN at tth 0 as an additional column
    zero_crossings = np.where(np.diff(np.sign(inputmap[plot_setting['axis'][0]][0,:])))[0]
    #print("%d"%(zero_crossings[0]))
    if zero_crossings.size:
        #print("insert NaN into ploting matrix")
        plotmap[plot_setting['axis'][0]] = np.insert(inputmap[plot_setting['axis'][0]],zero_crossings+1, (0.9999*inputmap[plot_setting['axis'][0]][:,zero_crossings] + 0.0001*inputmap[plot_setting['axis'][0]][:,zero_crossings+1]), axis = 1)
        plotmap[plot_setting['axis'][0]] = np.insert(plotmap[plot_setting['axis'][0]],zero_crossings+2, (0.0001*inputmap[plot_setting['axis'][0]][:,zero_crossings] + 0.9999*inputmap[plot_setting['axis'][0]][:,zero_crossings+1]), axis = 1)
        
        plotmap['mat'] = np.insert(inputmap['mat'],zero_crossings+1, np.nan, axis = 1)
        plotmap['mat'] = np.insert(plotmap['mat'],zero_crossings+2, np.nan, axis = 1)
        
        if inputmap[plot_setting['axis'][1]].ndim !=1:
            plotmap[plot_setting['axis'][1]] = np.insert(inputmap[plot_setting['axis'][1]],zero_crossings+1, (0.9999*inputmap[plot_setting['axis'][1]][:,zero_crossings] + 0.0001*inputmap[plot_setting['axis'][1]][:,zero_crossings+1]), axis = 1)
            plotmap[plot_setting['axis'][1]] = np.insert(plotmap[plot_setting['axis'][1]],zero_crossings+2, (0.0001*inputmap[plot_setting['axis'][1]][:,zero_crossings] + 0.9999*inputmap[plot_setting['axis'][1]][:,zero_crossings+1]), axis = 1)
        else:
            plotmap[plot_setting['axis'][1]]=inputmap[plot_setting['axis'][1]]
    else:
        plotmap[plot_setting['axis'][0]]=inputmap[plot_setting['axis'][0]]
        plotmap[plot_setting['axis'][1]]=inputmap[plot_setting['axis'][1]]
        plotmap['mat']=inputmap['mat']
        
    #return kwargs    
    # plot Qxy Q
    fig= plt.figure()
    ax = fig.gca()
    plt.pcolormesh(plotmap[plot_setting['axis'][0]],plotmap[plot_setting['axis'][1]],plotmap['mat'], norm=colors.LogNorm(vmin=plot_setting['range_I'][0], vmax=plot_setting['range_I'][1]), cmap = 'hot')
    #plt.pcolormesh(plotmap[plot_setting['axis'][0]],plotmap[plot_setting['axis'][1]],plotmap['mat'], vmin=plot_setting['range_I'][0], vmax=plot_setting['range_I'][1], cmap = 'hot')
    #plt.pcolormesh(plotmap[plot_setting['axis'][0]],plotmap[plot_setting['axis'][1]],plotmap['mat'], norm=colors.LogNorm(vmin=plot_setting['range_I'][0], vmax=plot_setting['range_I'][1]), cmap = 'hsv')
    
    if not plot_setting['range_axis']:
        xstart, xend = ax.get_xlim()
        ystart, yend = ax.get_ylim()
    else:
        xstart = np.min(plot_setting['range_axis'][0])
        xend = np.max(plot_setting['range_axis'][0])
        ystart = np.min(plot_setting['range_axis'][1])
        yend = np.max(plot_setting['range_axis'][1])
        
    if plot_setting['axis'][0] == 'Qxy':
        plt.xlabel(r'$Q_{xy}\/[\AA^{-1}]$',fontsize=fontsize)
        xaxis_step = 0.1
    elif plot_setting['axis'][0] == 'tth':
        plt.xlabel(r'$2\theta_h\/[^o]$',fontsize=fontsize)
        xaxis_step = 1
    else:
        plt.xlabel('channel',fontsize=fontsize)      
        xaxis_step = 200

    if plot_setting['axis'][1] == 'Qz':
        plt.ylabel(r'$Q_{z}\/[\AA^{-1}]$',fontsize=fontsize)
        yaxis_step = 0.1
    elif plot_setting['axis'][1] == 'tt':
        plt.ylabel(r'$2\theta\/[^o]$',fontsize=fontsize)
        yaxis_step = 1
    else:
        plt.ylabel('channel',fontsize=fontsize)      
        yaxis_step = 200
    
    plt.xlim(xstart, xend)
    plt.ylim(ystart, yend)
    plt.xticks(np.arange(np.ceil(xstart/xaxis_step)*xaxis_step, (np.ceil(xend/xaxis_step)+1)*xaxis_step, step=xaxis_step*2), fontsize=fontsize)
    plt.yticks(np.arange(np.ceil(ystart/yaxis_step)*yaxis_step, (np.ceil(yend/yaxis_step)+1)*yaxis_step, step=yaxis_step*2), fontsize=fontsize)
    for key in ax.spines:
        ax.spines[key].set_lw(1.5)
    ax.tick_params(width = 1.5)
    
    fig.subplots_adjust(bottom = margin['bottom'],top = margin['top'], left=margin['left'],right = margin['right'])
    cbar = plt.colorbar()
    cbar.outline.set_linewidth(1.5)
    cbar.ax.tick_params( width = 1.5, labelsize=fontsize)
    
    fig.set_size_inches(size['width']/25.4, size['height']/25.4)
    fig.dpi=size['dpi']
    
    return fig

def export(inputfig, filename, exp_path):
    inputfig.savefig(exp_path+filename+".png", dpi=inputfig.dpi, format='png')
    