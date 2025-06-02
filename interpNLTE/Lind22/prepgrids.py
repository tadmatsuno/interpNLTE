from astroquery.vizier import Vizier
import os
from astropy.table import Table,unique,vstack
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator,interp1d,NearestNDInterpolator,RBFInterpolator
from scipy.optimize import minimize_scalar
import numpy as np
import tqdm
import multiprocessing as mp
from multiprocessing.managers import BaseManager
import functools
import os

datadir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'data')


def download_tables():
    Vizier.ROW_LIMIT = int(1e9)
    id_catalog = "J/A+A/665/A33"
    lind22 = Vizier.get_catalogs(id_catalog)
    os.makedirs(datadir, exist_ok=True)
    for elem in ["na", "mg", "al"]:
        lind22[id_catalog + "/" + elem].write(
            os.path.join(datadir, f"{elem}.fits"),
            overwrite=True)
    
def select_element(elem):
    data = Table.read(os.path.join(datadir, f"{elem}.fits"))
    data['Line'] = data['Line'].data.astype(int)
    data.rename_column(f'__{elem.capitalize()}_Fe_','__X_Fe_')
    data['__X_H_'] = data['__X_Fe_'] + data['__Fe_H_']
    print('Lines:',np.unique(data['Line'].data))
    return data

def _funt_to_return(f):
    try:
        return lambda x,x0: f(x)
    except TypeError:
        return lambda x,x0: f(x,x0)

def get_cog(dline,teff,logg,feh,vturb,isNLTE,ewtoab=True):
    d1 = dline[(dline['Teff']==teff)&\
                (dline['logg']==logg)&\
                (dline['__Fe_H_']==feh)&\
                (dline['Vturb']==vturb)]
    x_h = d1['__X_H_'].data
    if isNLTE:
        logew = d1['logWN'].data
    else:
        logew = d1['logWL'].data

    if len(x_h)==0:
        return lambda x,x0: np.nan
    elif len(x_h) == 1:
        return lambda x,x0: np.where(x==x_h[0],logew[0],np.nan)
    elif len(x_h) == 2:
        kind = 'linear'
    elif len(x_h) == 3:
        kind = 'quadratic'
    else:
        kind = 'cubic'
    idx_sort = np.argsort(x_h)
    logew = logew[idx_sort]
    x_h = x_h[idx_sort]
    ismonotonic = np.all(logew[1:] - logew[:-1]>0)
    if ismonotonic:
        if ewtoab:
            return lambda x,x0: interp1d(logew,x_h,
                kind=kind,
                bounds_error=False,
                fill_value=np.nan)(x)
        else:
            return lambda x,x0: interp1d(x_h,logew,
                kind=kind,
                bounds_error=False,
                fill_value=np.nan)(x)
    else:
        fabtoew =  interp1d(x_h,logew,
                kind=kind,
                bounds_error=False,
                fill_value=np.nan)
        if ewtoab:
            print('EW changes non-monotonically w.r.t. abundance. Initial Guess required')
            def get_ab(ew,ab0):
                if (ew<logew[0])|(ew>logew[-1]):
                    # OoB
                    return np.nan
                if (np.argmax(logew)==0)|(np.argmax(logew)==(len(logew)-1)):
                    # logew has a minimum
                    res = minimize_scalar(fabtoew,
                        method='bounded',
                        bounds=(np.min(x_h),np.max(x_h)))    
                else:
                    # logew has a maximum
                    res = minimize_scalar(lambda x: -fabtoew(x),
                        method='bounded',
                        bounds=(np.min(x_h),np.max(x_h)))
                # Currently it assumes the function has only one extrema
                ab_extrem = res.x
                chi2 = lambda x: (fabtoew(x)-ew)**2.
                ab1 = minimize_scalar(chi2,
                        method='bounded',
                        bounds=(np.min(x_h),ab_extrem)).x
                ab2 = minimize_scalar(chi2,
                        method='bounded',
                        bounds=(ab_extrem,np.max(x_h))).x
                if (np.abs(ab1-np.min(x_h))<0.001) & (np.abs(ab2-np.max(x_h))<0.001):
                    # OoB
                    print('The solution is likely out of bound. Please double check')
                    return np.nan
                elif np.abs(ab1-np.min(x_h))<0.001 :
                    return ab2
                elif np.abs(ab2-np.max(x_h))<0.001 :
                    return ab1
                else: 
                    if np.abs(ab1-ab0) < np.abs(ab2-ab0):
                        return ab1
                    else:
                        return ab2
            return get_ab
        else:
            return lambda x,x0: fabtoew(x)

def get_NLTEabcorr1(idx,dline_param,fehs):
    d1 = dline_param.groups[idx]
    teff = d1['Teff'][0]
    logg = d1['logg'][0]
    feh = d1['__Fe_H_'][0]
    vturb = d1['Vturb'][0]
#    print(d1['Line'][0],teff,logg,feh,vturb)
    f_lte = get_cog(dline_param,teff=teff,logg=logg,feh=feh,vturb=vturb,isNLTE=False,ewtoab=True)
    f_nlte = get_cog(dline_param,teff=teff,logg=logg,feh=feh,vturb=vturb,isNLTE=True,ewtoab=True)

    try:
        feh_mp = np.max(fehs[fehs<feh])
    except ValueError:
        feh_mp = feh - 0.5
    try:
        feh_mr = np.min(fehs[fehs>feh])
    except ValueError:
        feh_mr = feh + 0.5
    f_lte_mp = get_cog(dline_param,teff=teff,logg=logg,feh=feh_mp,vturb=vturb,isNLTE=False,ewtoab=True)
    f_nlte_mp = get_cog(dline_param,teff=teff,logg=logg,feh=feh_mp,vturb=vturb,isNLTE=True,ewtoab=True)
    f_lte_mr = get_cog(dline_param,teff=teff,logg=logg,feh=feh_mr,vturb=vturb,isNLTE=False,ewtoab=True)
    f_nlte_mr = get_cog(dline_param,teff=teff,logg=logg,feh=feh_mr,vturb=vturb,isNLTE=True,ewtoab=True)

    lte_ab = np.array([f_lte(d['logWN'],d['__X_H_']) for d in d1])
    nlte_ab = np.array([f_nlte(d['logWL'],d['__X_H_']) for d in d1])
    lte_ab_mp = np.array([f_lte_mp(d['logWN'],d['__X_H_']) for d in d1])
    nlte_ab_mp = np.array([f_nlte_mp(d['logWL'],d['__X_H_']) for d in d1])
    lte_ab_mr = np.array([f_lte_mr(d['logWN'],d['__X_H_']) for d in d1])
    nlte_ab_mr = np.array([f_nlte_mr(d['logWL'],d['__X_H_']) for d in d1])    


    nlte_ab_fill = np.where(np.isnan(nlte_ab),
             np.where(d1['logWN']<d1['logWL'],
                nlte_ab_mr,
                # Since the EW is smaller in NLTE than in LTE, LTE EW is probably too large for NLTE.
                # Thus try a higher feh model above
                nlte_ab_mp),
            nlte_ab)
    lte_ab_fill = np.where(np.isnan(lte_ab),
                np.where(d1['logWN']<d1['logWL'],
                    lte_ab_mp,
                    # Since the EW is smaller in NLTE than in LTE, NLTE EW is probably too small for LTE.
                    # Thus try a lower feh model above.
                    lte_ab_mr),
                lte_ab)
    # 1 is to assume LTE [X/H] for the input
    d1['NLTEabcorr1'] = nlte_ab - d1['__X_H_'].data
    d1['NLTEabcorr1_fill'] = nlte_ab_fill - d1['__X_H_'].data
    # 2 is to assume NLTE [X/H] for the input
    d1['NLTEabcorr2'] = d1['__X_H_'].data - lte_ab
    d1['NLTEabcorr2_fill'] = d1['__X_H_'].data - lte_ab_fill
    return d1

def get_NLTEabcorr(dline):
    print(dline['Line'][0])
    dline_param = dline.group_by(['Teff','logg','__Fe_H_','Vturb'])
    fehs = np.unique(dline['__Fe_H_'].data)
    ngroups = len(dline_param.groups)
    return vstack([get_NLTEabcorr1(idx,dline_param,fehs) for idx in tqdm.tqdm(range(ngroups))])#ngroups))


def get_NLTE(data,teff_range=(4000,7000),wvl_range=(3000,7000)):
    print(len(data))
    data = data[(data['Teff']>teff_range[0])&(data['Teff']<teff_range[1])&(data['Line']>wvl_range[0])&(data['Line']<wvl_range[1])]
    print(len(data))
    data = data[data['logWN']!=data['logWL']] # remove all the entries with no LTE at all. Looks a bit odd to me
    print(len(data))
    data['NLTEabcorr1'] = np.nan
    data['NLTEabcorr2'] = np.nan
    data['NLTEabcorr1_fill'] = np.nan
    data['NLTEabcorr2_fill'] = np.nan
    data['NLTEabcorr1'].format = '%.3f'
    data['NLTEabcorr2'].format = '%.3f'
    data['NLTEabcorr1_fill'].format = '%.3f'
    data['NLTEabcorr2_fill'].format = '%.3f'


    #lines = np.unique(data['Line'].data)
    #fehs = np.unique(data['__Fe_H_'].data)

    datalines = data.group_by('Line')
    
    return vstack([get_NLTEabcorr(dline) for dline in datalines.groups])


def prepNLTEabcorrtables():
    download_tables()
    elements = ['na','mg','al']
    for elem in elements:
        data = select_element(elem)
        data = get_NLTE(data)
        data.write(os.path.join(datadir, f"{elem}_NLTEabcorr.fits"),overwrite=True)
        print(f'{elem} NLTEabcorr table prepared and saved.')
        
        
if __name__ == "__main__":
    prepNLTEabcorrtables()