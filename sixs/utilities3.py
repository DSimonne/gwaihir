
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 16:31:47 2016

@author: andrea
"""
#from __future__ import print_function, division
import numpy as np
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm
from scipy import ndimage
from scipy import interpolate
import glob, os
#from natsort import natsorted
from scipy import stats
from scipy import signal
import scipy as sp
#import pandas as pd
#from lmfit import Model, minimize, Parameters, Parameter, report_fit
import lmfit
from lmfit.models import LinearModel, LorentzianModel, GaussianModel, PseudoVoigtModel

#global test

def isFloat(string):
    '''Return True if the input can be interpreted as Float'''
    try:
        float(string)
        return True
    except ValueError:
        return False
def isStr(string):
    '''Return True if the input string can be interpreted as String'''
    if str == type(string):
        return True
    if str != type(string):
        return False
        
def getint(name):
    '''Given a file name it return its number when it is at the end:
        getint('NH3_Rh_MgO_hkl_scan_01601.nxs') 
        returns: 
                 1601'''
    basename = name.partition('.')[0]
    num = basename.split('_')[-1]
    return int(num)

def number2file(num, directory,extension = '*.nxs',splt = '_'):
    '''Given a number it returns the corresponding scan in the specified folder.'''
#    if directory == 'curr':
#        dirtotal = _recorder.projectdirectory + _recorder.subdirectory + '/'
#    else :
    dirtotal = directory
    li = glob.glob1(dirtotal,extension)
    for el in li:
        elnumber = el[:-4].split(splt)[-1]
        if isFloat(elnumber):
            number = float(elnumber)
            if number == num:
                return  el    



def frange(directory,nst,nend,extension = '*.nxs'):
    '''It select a files range with the numeration at the end of the file, 
    once is specified the file expention, default is nxs, splitting character is: "_"
    It returns a file list with the desired numbers'''    
    #li=os.listdir(directory)
    li = glob.glob1(directory,extension)
    #li.sort()    
    flist=[]
    '''File range number, return a list of filenames in the number range entered'''
    for el in li:
        elnumber = el[:-4].split('_')[-1]
        if isFloat(elnumber):
            if nst<=float(elnumber)<=nend:
                if os.path.isfile(directory+el):
                    flist.append(el)
             
    #flist.sort()
    flist.sort(key=getint)
    #print(flist)      
    return flist

def selectRootName(root2search,filelist):
    ''' find a txt pattern in the filen names of a file list
    it returns a files list containing such rootpattern ''' 
    outlist = []
    for el in filelist:
        if root2search in el:
            outlist.append(el)
    return outlist
  

def find_nearest(array,value):
    '''Given an array and a value it returns the index of the array where is located the closest number to the requested'''
    #array = np.asanyarray(array)
    array = array[np.logical_not(np.isnan(array))]
    idx = (np.abs(array-value)).argmin()
    return idx, array[idx]

def cropdata(x,y,xstart, xend):
    '''for a given x, y data it selects the range for y on the input: Xstart - Xend values '''
    x = np.asanyarray(x)
    y = np.asanyarray(y)
    idx1, idxV1 = find_nearest(x, xstart)
    idx2, idxV2 = find_nearest(x, xend)
    idx = [idx1, idx2]
    xout=x[min(idx):max(idx)]
    yout=y[min(idx):max(idx)] 
    return xout, yout


def filterOutlier(x,y, ts = 3):
    '''Meant to remove spikes from x,y arrays '''
    wz = detect_outlier(y, threshold = ts)
    #z = np.abs(stats.zscore(y))
    #wz = np.where(z > threshold)[0]
    xst = x
    yst = y
    wz.sort()
    wz[::-1] #starting from the last one otherwhise it changes!!!
    for el in wz:
        x = np.delete(x, el)
        y = np.delete(y, el)
    #plt.plot(xst, yst,'g')
    #plt.plot(x, y,'r')
#    
    return x, y


def detect_outlier(y, threshold = 3):
    '''It returns the position of the outliers as vector '''
    #ys = signal.savgol_filter(y, 67, 2)
    ys = signal.medfilt(y,kernel_size=5)
    z = np.abs(stats.zscore(y-ys))
    #plt.plot(y, '.g')
    #plt.plot(ys, '-r')
    #plt.plot(z, '.k')
    #plt.plot(y-ys, '-b')
    wz = np.where(z>threshold)[0]
    return wz
    
    


#def detect_outlier(data_1):
#    outliers=[]
#    threshold=3
#    mean_1 = np.mean(data_1)
#    std_1 =np.std(data_1)  
#    for y in data_1:
#        z_score= (y - mean_1)/std_1 
#        if np.abs(z_score) > threshold:
#            outliers.append(y)
#    return outliers


def bkgsub(xaxe,yaxe,limits=None):
    '''It subtract a linear bkg passing at the starting and and points of the inserted axis if 
    limits are not specified eg limits =[30.45,106.5].
    Bkg calculated passing through the first 5 and last 5 pts.'''

    if limits != None:
        Istart = np.abs(xaxe - limits[0]).argmin()
        #print (Istart)
        Iend = np.abs(xaxe - limits[1]).argmin()
        #print (Iend)
    else:
        Istart = 0
        Iend = -1
    y2 = smoothSvagol(yaxe[Istart:Iend])
    x2 = xaxe[Istart:Iend]
    #print(startE,endE,startI,endI)
    x2fit = np.append(x2[0:5],x2[-6:]) 
    y2fit = np.append(y2[0:5],y2[-6:])
    slope, intercept, r_value, p_value, std_err = stats.linregress(x2fit,y2fit)

    Bkg = slope * xaxe + intercept       #-0.05;
    yaxef = yaxe - Bkg

    return xaxe,yaxef 

def linFit(x,y):
    '''Return the parameter for a linear fit of x vs y '''
    
    #x = fn01.kinetic_energy
    #y = fn01.spectrum
    bkg = LinearModel(prefix = 'bkg_')
    pars = bkg.guess(y, x=x)
    init = bkg.eval(pars, x=x)

    #pars = mod.guess(y, x=x)
    out = bkg.fit(y, pars, x=x)
    #print(out.fit_report(min_correl=0.25))
    b = out.best_values['bkg_intercept']
    a = out.best_values['bkg_slope']
    return a, b

    
def numint(x,y):
    '''suppose a regular distribution of points'''
    dx = (max(x)-min(x))/np.shape(x)
    area = 0   
    for el in y:
        area = area + el*dx
    return  area

def smoothSvagol(vect):
    ''' it applyes the king of magic filter svagol to the y vector
    it returns a vector of the same lenght'''
    y = signal.savgol_filter(vect, 21, 3)
    return y

def smoothGauss(vect,degree=4):
     '''It smooths the vector vect by convoluting it with a gaussian 4 pts large.
     The output vector is 2*degree-1 shorter, ie:9 pts shorter for the default case.'''
     window=degree*2-1  
     weight=np.array([1.0]*window)  
     weightGauss=[]  
     for i in range(window):  
         i=i-degree+1  
         frac=i/float(window)  
         gauss=1/(np.exp((4*(frac))**2))  
         weightGauss.append(gauss)  
     weight=np.array(weightGauss)*weight  
     smoothed=[0.0]*(len(vect)-window)  
     for i in range(len(smoothed)):  
         smoothed[i]=sum(np.array(vect[i:i+window])*weight)/sum(weight) 
     #print  (type(degree))
         smoothed = np.asarray(smoothed)
     return smoothed

def gengauss(x,p=[1,1,1,10,0]):
    ''' generate a gaussian
    Parameters: [Intesnity, center, width, Bkg intercept, Bkg slope]
    using 3 parameters = flat gaussian
    using 4 parameters = flat gaussian on a Bkg
    using 5 parameters = Gaussian on a linear Bkg'''
    if np.shape(p)[0] == 3:
        #print('Flat Gaussian')
        y = p[0]*(1/p[2]*np.sqrt(2*np.pi))*np.exp(-1/2* ((x-p[1])/p[2])**2)
        #y = ((p[0]/(p[2]*np.sqrt(2*np.pi)))*np.exp(-1*(((x-p[1])**2)/(2*(p[2])**2)))) 
    if np.shape(p)[0] == 4:
        #print('Flat Gaussian + Bkg')
        y = p[0]*(1/p[2]*np.sqrt(2*np.pi))*np.exp(-1/2* ((x-p[1])/p[2])**2)+p[3]
        #y = ((p[0]/(p[2]*np.sqrt(2*np.pi)))*np.exp(-1*(((x-p[1])**2)/(2*(p[2])**2))))+p[3]
    if np.shape(p)[0] == 5:
        #print('Flat Gaussian + Bkg slope')
        y = p[0]*(1/p[2]*np.sqrt(2*np.pi))*np.exp(-1/2*((x-p[1])/p[2])**2)+p[3]+x*p[4]
        #   amp1*(1/width1*np.sqrt(2*np.pi))*np.exp(-1/2* ((xx-cen1)/width1)**2)
        #y = ((p[0]/(p[2]*np.sqrt(2*np.pi)))*np.exp(-1*(((x-p[1])**2)/(2*(p[2])**2))))+p[3]+x*p[4]
    if np.shape(p)[0] != 3 and np.shape(p)[0] != 4 and np.shape(p)[0] != 5:
        print('parameters p wrong dimension')
    return x, y

    
def fitgauss1d(x,y,p = [1,1,1,10,0.1],graph='NO'):
    '''Parameters: [Intesnity, center, width, bkg, bkg Slope]
    using 3 parameters = flat gaussian
    using 4 parameters = flat gaussian on a Bkg
    using 5 parameters = Gaussian on a linear Bkg
    '''    
    if np.shape(p)[0] == 3:
        #print('Flat Gaussian')
        gauss = lambda p, x:  ((p[0]/(p[2]*np.sqrt(2*np.pi)))*np.exp(-1*(((x-p[1])**2)/(2*(p[2])**2)))) 
    if np.shape(p)[0] == 4:
        #print('Flat Gaussian + Bkg')
        #gauss = lambda p, x:  p[0]*(1/p[2]*np.sqrt(2*np.pi))*np.exp(-1/2* ((x-p[1])/p[2])**2)+p[3]
        gauss = lambda p, x:  ((p[0]/(p[2]*np.sqrt(2*np.pi)))*np.exp(-1*(((x-p[1])**2)/(2*(p[2])**2))))+p[3]
        
    if np.shape(p)[0] == 5:
        #print('Flat Gaussian + Bkg slope')
        gauss = lambda p, x:  ((p[0]/(p[2]*np.sqrt(2*np.pi)))*np.exp(-1*(((x-p[1])**2)/(2*(p[2])**2))))+p[3]+x*p[4]    
    
    errfunc = lambda p, x, y: gauss(p, x) - y
    p0=p
    p1,success = sp.optimize.leastsq(errfunc, p0[:], args=(x, y))
    #p1,success = sp.optimize.leastsq(errfunc, p0[:], args=(x, y))
    
    #area = np.sqrt(2*np.pi)*p[0]*p[2]
    
    if graph != 'NO':
        plt.plot(x,y,'og')
        xfit,yfit = gengauss(x,p=p1)
        plt.plot(xfit,yfit,'r')
        plt.show()
        
    return p1,success

def gauss2min(params, xx, yy):
    amp1 = params['amp1'].value
    width1 = params['width1'].value    
    cen1 = params['cen1'].value
    slope = params['slope'].value    
    const = params['const'].value
    model = (amp1*(1/width1*np.sqrt(2*np.pi))*np.exp(-1/2* ((xx-cen1)/width1)**2)+
                 slope*xx + const)
    return model - yy

def gauss2min2(params, xx, yy):
    amp1 = params['amp1'].value
    width1 = params['width1'].value    
    cen1 = params['cen1'].value
    amp2 = params['amp2'].value
    width2 = params['width2'].value    
    cen2 = params['cen2'].value
    slope = params['slope'].value    
    const = params['const'].value
    model = (amp1*(1/width1*np.sqrt(2*np.pi))*np.exp(-1/2* ((xx-cen1)/width1)**2)+
             amp2*(1/width2*np.sqrt(2*np.pi))*np.exp(-1/2* ((xx-cen2)/width2)**2)+
                 slope*xx + const)
    return model - yy

def fitgauss1d_lim(x,y,aa=[10,0.1,100],ww=[0.2,0.01,0.6],cc=[0.8,0.55,1.1],
                   ss=[0.1,0.01,1],kk=[3,0.01,50],graph='NO'):
    '''
    a =  amplitude gaussian, min, max value eg: a=[10,0.5,100]
    w =  width peak, min, max value         eg: w=[0.2,0.01,0.6]
    c = center peak, min, max value
    s = background slope, min, max
    k = intercept linear bkg, min, max
    keywords returned in result.params.get('key').value
    are:
    amp1, width1, cen1, slope, const
    ''' 
    params = lmfit.Parameters()
    params.add('amp1',   value= aa[0],  min=aa[1], max=aa[2])
    params.add('width1',   value= ww[0],  min=ww[1], max=ww[2])
    params.add('cen1',   value= cc[0],  min=cc[1], max=cc[2])
    params.add('slope',   value= ss[0],  min=ss[1], max=ss[2]  )
    params.add('const',   value= kk[0],  min=kk[1], max=kk[2])
    
    ysize = np.shape(y)[0]
    indexes = ~np.isnan(y)
    nans = np.count_nonzero(np.isnan(y))
    #print('nans are: ', np.shape(nans))
    x = x[indexes]
    y = y[indexes]

    #print(np.shape(x))
    #print(np.shape(y))
    if nans < ysize/2:
        result = lmfit.minimize(gauss2min, params, args=(x, y))
    if nans > ysize/2:
        raise ValueError("Too many Nan")
    
  ####  result = lmfit.minimize(gaus2min, params, args=(gamCritHole, yHole))
  
    if graph != 'NO':
        plt.plot(x,y,'og')
        p = [result.params.get('amp1').value,
              result.params.get('width1').value,
              result.params.get('cen1').value,
              result.params.get('const').value,
              result.params.get('slope').value, ]
        print(p)
        yfit = (p[0]*(1/p[1]*np.sqrt(2*np.pi))*np.exp(-1/2* ((x-p[2])/p[1])**2)+p[4]*x + p[3])
        ystart = (aa[0]*(1/ww[0]*np.sqrt(2*np.pi))*np.exp(-1/2* ((x-cc[0])/ww[0])**2)+ss[0]*x + kk[0])
        #yfit = p[0]*(1/p[2]*np.sqrt(2*np.pi))*np.exp(-1/2*((x-p[1])/p[2])**2)+p[3]+x*p[4]
        
        #x,yfit = gengauss(x,p=p1)
        plt.plot(x,yfit,'r')
        plt.plot(x,ystart,'b')
        plt.show()
    return result  



def fitgauss1d2_lim(x,y,aa1=[10,0.1,100],ww1=[0.2,0.01,0.6],cc1=[0.8,0.55,1.1],
                    aa2=[10,0.1,100],ww2=[0.2,0.01,0.6],cc2=[0.8,0.55,1.1],
                    ss=[0.1,0.01,1],kk=[3,0.01,50],graph='NO'):
    '''
    a =  amplitude gaussian, min, max value eg: a=[10,0.5,100]
    w =  width peak, min, max value         eg: w=[0.2,0.01,0.6]
    c = center peak, min, max value
    s = background slope, min, max
    k = intercept linear bkg, min, max
    keywords returned in result.params.get('key').value
    are:
    amp1, width1, cen1, slope, const
    ''' 
    params = lmfit.Parameters()
    params.add('amp1',   value= aa1[0],  min=aa1[1], max=aa1[2])
    params.add('width1',   value= ww1[0],  min=ww1[1], max=ww1[2])
    params.add('cen1',   value= cc1[0],  min=cc1[1], max=cc1[2])
    params.add('amp2',   value= aa2[0],  min=aa2[1], max=aa2[2])
    params.add('width2',   value= ww2[0],  min=ww2[1], max=ww2[2])
    params.add('cen2',   value= cc2[0],  min=cc2[1], max=cc2[2])
    params.add('slope',   value= ss[0],  min=ss[1], max=ss[2]  )
    params.add('const',   value= kk[0],  min=kk[1], max=kk[2])
    
    ysize = np.shape(y)[0]
    indexes = ~np.isnan(y)
    nans = np.count_nonzero(np.isnan(y))
    #print('nans are: ', np.shape(nans))
    x = x[indexes]
    y = y[indexes]

    #print(np.shape(x))
    #print(np.shape(y))
    if nans < ysize/2:
        result = lmfit.minimize(gauss2min2, params, args=(x, y))
    if nans > ysize/2:
        raise ValueError("Too many Nan")
    
  ####  result = lmfit.minimize(gaus2min, params, args=(gamCritHole, yHole))
  
    if graph != 'NO':
        plt.plot(x,y,'og')
        p1 = [result.params.get('amp1').value,
              result.params.get('width1').value,
              result.params.get('cen1').value,
              result.params.get('const').value,
              result.params.get('slope').value, ]
        p2 = [result.params.get('amp2').value,
              result.params.get('width2').value,
              result.params.get('cen2').value ]
        print(p1)
        yfit = (p1[0]*(1/p1[1]*np.sqrt(2*np.pi))*np.exp(-1/2* ((x-p1[2])/p1[1])**2)) +p1[4]*x + p1[3] + (p2[0]*(1/p2[1]*np.sqrt(2*np.pi))*np.exp(-1/2* ((x-p2[2])/p2[1])**2))  
        g1   = (p1[0]*(1/p1[1]*np.sqrt(2*np.pi))*np.exp(-1/2* ((x-p1[2])/p1[1])**2)) +p1[4]*x + p1[3]
        g2   =                                                                       +p1[4]*x + p1[3] + (p2[0]*(1/p2[1]*np.sqrt(2*np.pi))*np.exp(-1/2* ((x-p2[2])/p2[1])**2)) 
        ystart = (aa1[0]*(1/ww1[0]*np.sqrt(2*np.pi))*np.exp(-1/2* ((x-cc1[0])/ww1[0])**2)) +ss[0]*x + kk[0] + (aa2[0]*(1/ww2[0]*np.sqrt(2*np.pi))*np.exp(-1/2* ((x-cc2[0])/ww2[0])**2))
        #ystart = (aa[0]*(1/ww[0]*np.sqrt(2*np.pi))*np.exp(-1/2* ((x-cc[0])/ww[0])**2)+ss[0]*x + kk[0])
        #yfit = p[0]*(1/p[2]*np.sqrt(2*np.pi))*np.exp(-1/2*((x-p[1])/p[2])**2)+p[3]+x*p[4]
        
        #x,yfit = gengauss(x,p=p1)
        plt.plot(x,yfit,'r')
        plt.plot(x,ystart,'b')
        plt.plot(x, g1, 'g')
        plt.plot(x, g2, 'k')
        plt.show()
    return result  

 ###More straighforwards way to fit

def fitPeaks(x,y, colorData = '.g',plot = 'NO', print_it = 'NO'):
    '''Not Tested yet
    should substitute the older fit fuctions at least for the predefined.'''
    bkg = LinearModel(prefix = 'bkg_')
    pars = bkg.guess(y, x=x)   # it make a guess to initiaalise the parameters for the linear part
    #pars = lmfit.Parameters()
    peak1 = GaussianModel(prefix = 'd1_')  #here fedine the peak shape model
    pars.update(peak1.make_params())
    peak2 = GaussianModel(prefix = 'd2_')
    pars.update(peak2.make_params())

    pars['d1_center'].set(397.6, min=397.3, max=397.9) # Variable Position,  Upper Limit, Lower Limit
    pars['d1_sigma'].set(0.88, min=0.7, max=1.15)
    pars['d1_amplitude'].set(0.15, min=0.01, max = 10)
        
    pars['d2_center'].set(400, min=399.7, max=400.3)
    pars['d2_sigma'].set(0.88, min=0.7, max=1.15)
    pars['d2_amplitude'].set(0.15, min=0.01, max = 10)


    mod = bkg + peak1 + peak2 # buols the total model
    init = mod.eval(pars, x=x) #evaluate the function value
    out = mod.fit(y, pars, x=x) #out is the object containing the results of the tit

    if plt != 'NO':   # just a quick plot way to visualise the fit results + data
        plt.plot(x, y, colorData)
        plt.plot(x, out.best_fit, '-b')
        comps = out.eval_components(x=x)
        bkgFit = comps['bkg_']
        plt.plot(x, comps['d1_']+bkgFit, '-r')
        plt.plot(x, comps['d2_']+bkgFit, '-r')
        plt.show()
    if print_it != 'NO':
        print(out.fit_report(min_correl=0.25)) # print report only if the fit result is minimally decent
    return init, out

def binto(x,y,Nintervals):
    '''For a given x, y return a new pair of vectors with Nintervals isospaced intervals
    it returns the data into the new vectors'''
    n, bins = np.histogram(x, bins =Nintervals, range = (np.nanmin(x), np.nanmax(x)))
    fy, bins = np.histogram(x, bins=Nintervals, range = (np.nanmin(x), np.nanmax(x)), weights=y)
    ny = fy / n
    nx = np.linspace(np.nanmin(x), np.nanmax(x), Nintervals)
    return nx, ny

def bintoM(x,y,Nintervals):
    '''For a given x, y return a new pair of vectors with Nintervals isospaced intervals
    it returns the data into the new vectors.
    y is intended to be a set of vectors to function of x, to be binned to the same number of intervals
    It retuns the new x and the new y'''
    col = np.shape(y)[1]
    ny = np.zeros((Nintervals, col))
    nx = np.zeros((Nintervals))    
    for n in np.arange(col):
        xt,yt = binto(x, y[:,n],Nintervals)
        ny[:,n] = yt
    nx = xt    
    return nx, ny

def fill_nan(A):
    '''
    interpolate to fill nan values
    '''
    inds = np.arange(A.shape[0])
    good = np.where(np.isfinite(A))
    f = interpolate.interp1d(inds[good], A[good],bounds_error=False)
    B = np.where(np.isfinite(A),A,f(inds))
    if np.isnan(B[0]):
        B[0] = B[1]
    if np.isnan(B[-1]):
        B[-1] = B[-2]
    return B

#def remove_nan(x,y):
#    nanlist = np.where(np.isnan(y))
#    nanlist =  np.flip(nanlist)
#    nanlist.sort()
#    for el in nanlist:
#        x= np.delete(x,el)
#        y= np.delete(y,el)     
#    return x,y

def removeNan(x,y):
    '''it checks if any of the two vectors contains nan and remove them therein 
    and from its pair in the other vector'''
    xff=[]
    yff=[]
    if np.shape(y) == np.shape(x):
        ynan = np.argwhere(np.isnan(y))
        yf = np.delete(y, ynan)
        xf = np.delete(x, ynan)
        xnan = np.argwhere(np.isnan(xf))
        xff = np.delete(xf, xnan)
        yff = np.delete(yf, xnan)
        
    return xff, yff
        

def varname(var):
  import inspect
  frame = inspect.currentframe()
  var_id = id(var)
  for name in frame.f_back.f_locals.keys():
    try:
      if id(eval(name)) == var_id:
        return(name)
    except:
      raise

###############################################################################
###############################################################################

#import lmfit
#from lmfit.models import LinearModel, LorentzianModel, GaussianModel, PseudoVoigtMode #get some of the pre-defined models available
#
#def fitPeaks(x,y, colorData = '.g',plot = 'NO', print_it = 'NO'):
#    bkg = LinearModel(prefix = 'bkg_')
#    pars = bkg.guess(y, x=x)   # it make a guess to initiaalise the parameters for the linear part
#    #pars = lmfit.Parameters()
#    peak1 = GaussianModel(prefix = 'd1_')  #here fedine the peak shape model
#    pars.update(peak1.make_params())
#    peak2 = GaussianModel(prefix = 'd2_')
#    pars.update(peak2.make_params())
#
#    pars['d1_center'].set(397.6, min=397.3, max=397.9) # Variable Position,  Upper Limit, Lower Limit
#    pars['d1_sigma'].set(0.88, min=0.7, max=1.15)
#    pars['d1_amplitude'].set(0.15, min=0.01, max = 10)
#        
#    pars['d2_center'].set(400, min=399.7, max=400.3)
#    pars['d2_sigma'].set(0.88, min=0.7, max=1.15)
#    pars['d2_amplitude'].set(0.15, min=0.01, max = 10)
#
#
#    mod = bkg + peak1 + peak2 # buols the total model
#    init = mod.eval(pars, x=x) #evaluate the function value
#
#    out = mod.fit(y, pars, x=x) #out is the object containing the results of the tit
#    if plt != 'NO':   # just a quick plot way to visualise the fit results + data
#        plt.plot(x, y, colorData)
#        plt.plot(x, out.best_fit, '-b')
#        comps = out.eval_components(x=x)
#        bkgFit = comps['bkg_']
#        plt.plot(x, comps['d1_']+bkgFit, '-r')
#        plt.plot(x, comps['d2_']+bkgFit, '-r')
#        plt.show()
#    if print_it != 'NO':
#        print(out.fit_report(min_correl=0.25)) # print report only if the fit result is minimally decent
#    return out






# create data to be fitted
#x = np.linspace(0, 15, 301)
#data = (5. * np.sin(2 * x - 0.1) * np.exp(-x*x*0.025) + np.random.normal(size=len(x), scale=0.2) )
#
## define objective function: returns the array to be minimized
#def fcn2min(params, x, data):
#    """ model decaying sine wave, subtract data"""
#    amp = params['amp'].value
#    shift = params['shift'].value
#    omega = params['omega'].value
#    decay = params['decay'].value
#
#    #gauss = lambda p, x:  p[0]*(1/p[2]*np.sqrt(2*np.pi))*np.exp(-1/2* ((x-p[1])/p[2])**2)+p[3]+x*p[4]
#    #model = amp * np.sin(x * omega + shift) * np.exp(-x*x*decay)
#    model = int1*(1/width1*np.sqrt(2*np.pi))*np.exp(-1/2* ((x-cen1)/width1)**2)   
#    return model - data
#
## create a set of Parameters
#params = lmfit.Parameters()
#params.add('amp',   value= 10,  min=0)
#params.add('decay', value= 0.1)
#params.add('shift', value= 0.0, min=-np.pi/2., max=np.pi/2)
#params.add('omega', value= 3.0)
#
#
## do fit, here with leastsq model
#result = minimize(fcn2min, params, args=(x, data))
#
## calculate final result
#final = data + result.residual
#
## write error report
#report_fit(result.params)
#
## try to plot results
#try:
#    import pylab
#    pylab.plot(x, data, 'k+')
#    pylab.plot(x, final, 'r')
#    pylab.show()
#except:
#    pass
############## or using another fitting routine
#from numpy import sqrt, pi, exp, loadtxt
#import matplotlib.pyplot as plt
#
#
#### to use pre-defined models
##from lmfit.models import ConstantModel, GaussianModel
##gaussian = GaussianModel()
##bkg  = ConstantModel()
##mod = gaussian + bkg
#
#
###### to define your own models:
#from lmfit import Model
#def gaussian(x, amplitude, center, sigma):
#    "1-d gaussian: gaussian(x, amp, cen, wid)"
#    return (amplitude/(sqrt(2*pi)*sigma)) * exp(-(x-center)**2 /(2*sigma**2))
#
#def bkg(x,  c):
#    "line"
#    return  c
#mod = Model(gaussian) + Model(bkg)
#
#
#
##data = loadtxt('model1d_gauss.dat')
#x = gamx
#y = y
#
#pars  = mod.make_params( amplitude=5, center=17.25, sigma=.01, c=120)
#result = mod.fit(y, pars, x=x)
#
#result.values

#print(result.fit_report())
#
#plt.plot(x, y,         'bo')
#plt.plot(x, result.init_fit, 'k--')
#plt.plot(x, result.best_fit, 'r-')
#plt.show()


