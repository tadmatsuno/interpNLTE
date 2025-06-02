from astropy.table import Table
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
from scipy.spatial import ConvexHull
import numpy as np
from scipy.interpolate import interp1d
import multiprocessing as mp
import time
import tqdm
from scipy.spatial import QhullError
import os

datadir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'data')

Aldata = Table.read(os.path.join(datadir,'al_NLTEabcorr.fits')).group_by('Line')
Mgdata = Table.read(os.path.join(datadir,'mg_NLTEabcorr.fits')).group_by('Line')
Nadata = Table.read(os.path.join(datadir,'na_NLTEabcorr.fits')).group_by('Line')

def isinhull(points, hull):
    # This is taken from 
    # The hull is defined as all points x for which Ax + b <= 0.
    # We compare to a small positive value to account for floating
    # point issues.
    #
    # Assuming x is shape (m, d), output is boolean shape (m,).

    A, b = hull.equations[:, :-1], hull.equations[:, -1:]
    eps = np.finfo(np.float32).eps
    return np.all(np.asarray(points) @ A.T + b.T < eps, axis=1)

def construct_NLTE_interpolator(corr,parameters={},scale_inputs={}):
    '''
    You need to give input parameters for the correction as a dictionary
    '''


    for key in parameters.keys():
        if not key in scale_inputs.keys():
            scale_inputs[key] = 1.
    mask = np.isfinite(corr)
    points = np.array([np.array(value)/scale_inputs[key] for key,value in parameters.items()]).T[mask]
    if len(points) == 0:
        hull_error = True
    else:
        try:
            param_hull = ConvexHull(points)
            hull_error = False
            f_corr = RBFInterpolator(points,np.atleast_2d(corr[mask]).T)
        except QhullError:
            hull_error = True

    def NLTE_interpolator(**kwargs):
        '''
        Input parameters are given as keyword arguments
        '''
        if hull_error:
            return np.nan,False
        for key in kwargs.keys():
            if not key in parameters.keys():
                raise ValueError('Unknown parameter {}\n'.format(key)+
                    'Available parameters are {}'.format(','.join(list(parameters.keys()))))
        for key in parameters.keys():
            if not key in kwargs.keys():
                raise ValueError('Missing parameter {}\n'.format(key)+
                    'Available parameters are {}'.format(','.join(list(parameters.keys()))))
        out_points = np.array([np.atleast_1d(value)/scale_inputs[key] for key,value in kwargs.items()]).T
        withingrid = isinhull(out_points,param_hull)
        corr_out = f_corr(out_points)
        return corr_out,withingrid
    
    return NLTE_interpolator

def Lind22interpolator(data,line,use_filled_abundance = False):
    if not line in data['Line']:
        raise ValueError('Unknown line {}'.format(line))
    dataline = data[data['Line']==line]
    fehs = np.unique(dataline['__Fe_H_'].data)
    f_corr1 = {}
    f_corr2 = {}
    for vt in [1.0,2.0]:
        f_corr1[vt] = {}
        f_corr2[vt] = {}
        for feh in fehs:
            d1 = dataline[(dataline['Vturb']==vt) & (dataline['__Fe_H_']==feh)]
            if use_filled_abundance:
                yy1 = d1['NLTEabcorr1_fill'].data
                yy2 = d1['NLTEabcorr2_fill'].data
            else:
                yy1 = d1['NLTEabcorr1'].data
                yy2 = d1['NLTEabcorr2'].data
            f_corr1[vt][feh] = construct_NLTE_interpolator(yy1,\
                parameters={'teff':d1['Teff'].data,'logg':d1['logg'].data,'logEW':d1['logWL'].data},
                scale_inputs={'teff':250,'logg':0.5,'logEW':0.2}
                )
            f_corr2[vt][feh] = construct_NLTE_interpolator(yy2,\
                parameters={'teff':d1['Teff'].data,'logg':d1['logg'].data,'logEW':d1['logWN'].data},
                scale_inputs={'teff':250,'logg':0.5,'logEW':0.2}
                )

    def get_correction(teff,logg,feh,vt,logew,return_flags=True):
        '''
        This function returns corrections that are based on LTE EW and NLTE EW.
        They should be more or less the same.
        It can return three flags if return_flags is True
        1. Extrapolation in vt
        2. Extrapolation in [Fe/H] for LTE EW-based correction
        3. Extrapolation in [Fe/H] for NLTE EW-based correction
        '''
        vt_clip = np.clip(vt,1.0,2.0)
        corr1,flag1 = {1.0:[],2.0:[]},{1.0:[],2.0:[]}
        corr2,flag2 = {1.0:[],2.0:[]},{1.0:[],2.0:[]}
        for mm in fehs:
            for tt in [1.0,2.0]:
                corr,flag = f_corr1[tt][mm](teff=teff,logg=logg,logEW=logew)
                corr1[tt].append(np.atleast_1d(corr).ravel()[0])
                flag1[tt].append(np.atleast_1d(flag).ravel()[0])
                corr,flag = f_corr2[tt][mm](teff=teff,logg=logg,logEW=logew)
                corr2[tt].append(np.atleast_1d(corr).ravel()[0])
                flag2[tt].append(np.atleast_1d(flag).ravel()[0])
        def interpolate_feh(corrections,flags):
            corrections = np.atleast_1d(corrections)
            flags = np.atleast_1d(flags)
            if np.sum(flags)==0:
                return np.nan,True
            elif np.sum(flags)==1:
                return corrections[flags][0],True
            elif np.sum(flags)==2:
                kind = 'linear'
            elif np.sum(flags)==3:
                kind = 'quadratic'
            else:
                kind = 'cubic'
            feh_in = np.clip(feh,np.min(fehs[flags]),np.max(fehs[flags]))
            return interp1d(fehs[flags],corrections[flags],kind=kind,bounds_error=False)(feh_in),feh_in-feh
        corr1_vt1,extrap1_feh_vt1 = interpolate_feh(corr1[1.0],flag1[1.0])
        corr1_vt2,extrap1_feh_vt2 = interpolate_feh(corr1[2.0],flag1[2.0])
        if np.abs(extrap1_feh_vt1)>np.abs(extrap1_feh_vt2):
            extrap1_feh_vt = extrap1_feh_vt1
        else:
            extrap1_feh_vt = extrap1_feh_vt2
        corr2_vt1,extrap2_feh_vt1 = interpolate_feh(corr2[1.0],flag2[1.0])
        corr2_vt2,extrap2_feh_vt2 = interpolate_feh(corr2[2.0],flag2[2.0])
        if np.abs(extrap2_feh_vt1)>np.abs(extrap2_feh_vt2):
            extrap2_feh_vt = extrap2_feh_vt1
        else:
            extrap2_feh_vt = extrap2_feh_vt2
        corr1 = (corr1_vt1*(2.0-vt_clip)+corr1_vt2*(vt_clip-1.0))
        corr2 = (corr2_vt1*(2.0-vt_clip)+corr2_vt2*(vt_clip-1.0))
        if return_flags:
            return corr1,corr2,extrap1_feh_vt,extrap2_feh_vt,vt_clip-vt
        else:
            return corr1,corr2
    return get_correction

def get_corrections(data, ew_columns, element):
    '''
    Returns a new table with NLTE corrections attached.
    It needs to have teff, logg, feh, vt, and EW columns. \
    EW columns should be in the format of XXX_<wavelength> where \
    XXX can be anything and the interger <wavelength> is the wavelength in Angstroms. \
    
    Element should be one of ['Na','Mg','Al'].
    '''
    assert element in ['Na','Mg','Al'], 'Element {} is not supported'.format(element)
    
    if element == 'Na':
        NLTEdata = Nadata
    elif element == 'Mg':
        NLTEdata = Mgdata
    elif element == 'Al':
        NLTEdata = Aldata
           
    for clm in ew_columns:
        wvl = int(clm.split('_')[-1])
        print(wvl)
        f_NLTE_f = Lind22interpolator(NLTEdata,wvl,use_filled_abundance=True)
        f_NLTE = Lind22interpolator(NLTEdata,wvl,use_filled_abundance=False)
        data[clm+"_NLTEcorr"] = np.nan
        data[clm+"_NLTEflag"] = -1
        for d in data:
            if np.isnan(d[clm]):
                continue
            corr_res = f_NLTE(teff=d['teff'],logg=d['logg'],feh=d['feh'],vt=d['vt'],logew=np.log10(d[clm]))
            corr_res_f = f_NLTE_f(teff=d['teff'],logg=d['logg'],feh=d['feh'],vt=d['vt'],logew=np.log10(d[clm]))
            if np.isfinite(corr_res[0]) & np.isfinite(corr_res[1]):
                if corr_res[2] <= corr_res[3]:
                    corr_adopt = corr_res[0]
                    flag = 0
                else:
                    corr_adopt = corr_res[1]
                    flag = 1
            elif np.isfinite(corr_res[0]):
                corr_adopt = corr_res[0]
                flag = 2
            elif np.isfinite(corr_res[1]):
                corr_adopt = corr_res[1]
                flag = 3
            elif np.isfinite(corr_res_f[0]) & np.isfinite(corr_res_f[1]):
                if corr_res_f[2] <= corr_res_f[3]:
                    corr_adopt = corr_res_f[0]
                    flag = 4
                else:
                    corr_adopt = corr_res_f[1]
                    flag = 5
            elif np.isfinite(corr_res_f[0]):
                corr_adopt = corr_res_f[0]
                flag = 6
            elif np.isfinite(corr_res_f[1]):
                corr_adopt = corr_res_f[1]
                flag = 7
            else:
                corr_adopt = np.nan
                flag = -1
            d[clm+"_NLTEcorr"] = corr_adopt
            d[clm+"_NLTEflag"] = flag
            
        
