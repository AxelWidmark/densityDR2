import numpy as np
import healpy as hp
import matplotlib.pylab as plt
import math
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import RectBivariateSpline as spline2d
from scipy.integrate import quad
pi=math.pi

# loads dust map
red_ext = 3.1
rlims = [0.1,0.2]
absGlims = (3.0,3.7)
from scipy.interpolate import RegularGridInterpolator
import healpy
npzfile = np.load("../Data/3D_dust_interpolation.npz")
x = npzfile['x']
y = npzfile['y']
z = npzfile['z']
BmV_matrix = npzfile['BmV_matrix']
BmV_interp = RegularGridInterpolator((x,y,z), BmV_matrix, method='linear')

# creates a function that gives the distribution of parallax uncertainties, in bins of apparent G-mag and Galactic latitude b
Gmagvec = np.linspace(5.,15.,21)
bvec = np.arcsin(np.linspace(-1.,1.,11))
parerrpercentilesgrid = np.load('../Data/parerrpercentilesgrid.npy')
def get_parerrpercentiles(Gmag,b):
    if Gmag<5. or Gmag>15:
        return np.zeros(100)
    for i in range(len(Gmagvec)-1):
        if Gmag>=Gmagvec[i] and Gmagvec[i+1]>=Gmag:
            Gmag_i = i
    for k in range(len(bvec)-1):
        if b>=bvec[k] and bvec[k+1]>=b:
            b_k = k
    return parerrpercentilesgrid[Gmag_i][b_k]

# interpolate the luminosity function as a function of absolute G-band magnitude, see section 2.4 and figure 2 in the paper
npzfile = np.load('../Data/absG_histogram.npz')
bin_means = npzfile['bin_means']
histvals = npzfile['histvals']
luminosityPDF = UnivariateSpline(bin_means,histvals,s=5e5)

# completeness inferred from 2MASS xmatch, Gmag in 8--12
npzfile = np.load("../Data/completeness.npz")
map8to12 = npzfile['map8to12']
map12to15 = npzfile['map12to15']
def completeness(l,b,Gmag):
    if Gmag>8. and Gmag<12.:
        return hp.pixelfunc.get_interp_val(map8to12,b+pi/2.,l,nest=True)
    if Gmag>12. and Gmag<15.:
        return hp.pixelfunc.get_interp_val(map12to15,b+pi/2.,l,nest=True)

# probability for a star to have observed parallax and observed absolute magnitude that are in accordance with the sample cuts (see section 3.1 in the paper)
# parerrquantiles -- the star's distribution of parallax uncertaintes
# Dlims -- distance limits for which the star will be included in the sample
def selection_par(parreal,parerrquantiles,Dlims):
    due2par = 0.
    for parerr in parerrquantiles:
        due2par += (math.erf((1./Dlims[0]-parreal)/(math.sqrt(2.)*parerr))-math.erf((1./Dlims[1]-parreal)/(math.sqrt(2.)*parerr)))/2./len(parerrquantiles)
    return due2par

# probability of selection (both completeness of Gaia and sample construction effects)
def completeness_and_selection_par_Gmag(parreal,absG,l,b):
    xyzdir = np.array([math.cos(l)*math.cos(b), math.sin(l)*math.cos(b), math.sin(b)])
    magcorr = 5.*(np.log10(1e3/parreal)-1.)
    Gmag = absG+magcorr+red_ext*BmV_interp(1e3/parreal*xyzdir)[0]
    if Gmag<5. or Gmag>15:
        return 0.
    Dlims = [None,None]
    obs_dist_vec = np.linspace(rlims[0],rlims[1],10)
    obs_absG_vec = []
    for obs_dist in obs_dist_vec:
        obs_absG_vec.append( Gmag-5.*(np.log10(1e3*obs_dist)-1.)-red_ext*BmV_interp(1e3*obs_dist*xyzdir)[0] )
    if obs_absG_vec[0]<absGlims[0] or obs_absG_vec[-1]>absGlims[1]:
        return 0.
    else:
        if obs_absG_vec[-1]>absGlims[0] and obs_absG_vec[-1]<absGlims[1]:
            Dlims[1] = float(rlims[1])
        else:
            for i in range(len(obs_absG_vec)-1):
                if obs_absG_vec[i]>=absGlims[0] and obs_absG_vec[i+1]<=absGlims[0]:
                    Dlims[1] = obs_dist_vec[i]+abs((obs_absG_vec[i]-absGlims[0])/(obs_absG_vec[i]-obs_absG_vec[i+1]))*abs(obs_dist_vec[i+1]-obs_dist_vec[i])
                    break
        if obs_absG_vec[0]>absGlims[0] and obs_absG_vec[0]<absGlims[1]:
            Dlims[0] = float(rlims[0])
        else:
            for i in range(len(obs_absG_vec)-1):
                if obs_absG_vec[i]>=absGlims[1] and obs_absG_vec[i+1]<=absGlims[1]:
                    Dlims[0] = obs_dist_vec[i]+abs((obs_absG_vec[i]-absGlims[1])/(obs_absG_vec[i]-obs_absG_vec[i+1]))*abs(obs_dist_vec[i+1]-obs_dist_vec[i])
                    break
        if Dlims[0]>Dlims[1]:
            return 0.
        else:
            return completeness(l,b,Gmag)*selection_par(parreal,get_parerrpercentiles(Gmag,b),Dlims)

        
# in this part, we generate stars and calculate their probability of selection,
# this gives the effective area (see Eq. 34 in the paper) by Monte-Carlo integration
print('start, 50,000,000 iterations')
n_objects = 50000000
# given parallax error of 1.5 mas, the error to the calculated absG at 200 pc distance is less than 0.6
absGrange = [absGlims[0]-0.6, absGlims[1]+0.6]
rhorange = [1./(1./rlims[0]+1.5), 1./(1./rlims[1]-1.5)]
print(absGrange,rhorange)
absGlist = np.random.uniform(low=absGrange[0],high=absGrange[1],size=n_objects)
rholist = np.random.uniform(low=rhorange[0],high=rhorange[1],size=n_objects)
llist = np.random.uniform(low=0.,high=2.*pi,size=n_objects)
blist = np.arcsin( np.random.uniform(low=-1.,high=1.,size=n_objects) )
zlist = rholist*np.sin(blist)
weightlist = np.array([rholist[i]**2.*luminosityPDF(absGlist[i])*completeness_and_selection_par_Gmag(1./rholist[i],absGlist[i],llist[i],blist[i]) for i in range(n_objects)])
zmax = 1./(1./rlims[1]-1.5)
zmin = 1./(1./rlims[0]+1.5)
zvec = np.linspace(-zmax,zmax,1001)
effectiveareavec = np.zeros(len(zvec)-1)
for i in range(n_objects):
    pos = int((zlist[i]-zvec[0])/(2.*zmax)*(len(zvec)-1.))
    effectiveareavec[pos] += weightlist[i]
np.savez("../Data/effective_area_absGlims-"+str(absGlims[0])+"-"+str(absGlims[1])+'_dust',effectivearea=effectiveareavec/np.sum(effectiveareavec),zlist=(zvec[0:1000]+zvec[1:1001])/2.)
