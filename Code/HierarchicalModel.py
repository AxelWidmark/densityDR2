import numpy as np
import scipy as sp
import math
pi=math.pi
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
import time


class stellarsample():
    # inialize class, this specifies what sample of stars to load
    # dust -- whether or not to include dust extinction corrections
    # toy -- if true, loads the mock data samples
    # in_plane -- if true, the velocity information of stars for which b<5 degrees is considered
    def __init__(self,absGlims,rlims=[0.1,0.2],dust=True,toy=False,in_plane=False):
        self.in_plane = in_plane
        self.rlims = rlims
        self.k = 4.74057 # unit conversion
        self.red_ext = 3.1 # reddening to extinction coefficient
        self.dust = dust
        self.data_set = np.load('../Data/'+toy*'mock_'+'dataset_absGlims-'+str(absGlims[0])+'-'+str(absGlims[1])+'_rlims-'+str(rlims[0])+'-'+str(rlims[1])+dust*'_dust'+'.npy')
        self.number_of_objects = len(self.data_set)
        
        # dust map, this takes units in pc
        if self.dust:
            from scipy.interpolate import RegularGridInterpolator
            npzfile = np.load('../Data/3D_dust_interpolation.npz')
            x = npzfile['x']
            y = npzfile['y']
            z = npzfile['z']
            BmV_matrix = npzfile['BmV_matrix']
            self.BmV_interp = RegularGridInterpolator((x,y,z), BmV_matrix, method='linear')
        
        # PDF luminosity function in absG (not normalized)
        npzfile = np.load('../Data/absG_histogram.npz')
        bin_means = npzfile['bin_means']
        histvals = npzfile['histvals']
        self.luminosityPDF = UnivariateSpline(bin_means,histvals,s=5e5)
        
        # effective area
        npzfile = np.load('../Data/effective_area_absGlims-'+str(absGlims[0])+"-"+str(absGlims[1])+dust*'_dust'+'.npz')
        self.effectivearea = npzfile['effectivearea']
        self.effectivearea_zsun = npzfile['zlist']
        
        # vx and vy distribution Gaussian mixtures
        npzfile = np.load('../Data/vx-vy_GMM.npz')
        self.vxvy_amps = npzfile['amps']
        self.vxvy_means = npzfile['means']
        self.vxvy_covars = npzfile['covars']
        
        
        print('*** INITIALIZATION COMPLETE ***')
        print('absG = ( '+str(absGlims[0])+' - '+str(absGlims[1])+' )')
        print('Number of objects:  ',self.number_of_objects)
        print('RV fraction:  ',1.-np.sum(np.isnan(self.data_set[:,5]))/self.number_of_objects,'\n\n')
    
    # multivariate gaussian
    def multivariate_normal(self,diff,cov):
        dim = len(diff)
        assert dim==np.shape(cov)[0] and dim==np.shape(cov)[1] and (dim,)==np.shape(diff)
        cov_inv = np.linalg.inv(cov)
        cov_det = np.linalg.det(cov)
        return math.exp(-1./2.*np.dot(np.dot(cov_inv,diff),diff))/math.sqrt((2.*pi)**dim*cov_det)
    
    # this is the stellar count norm, see section 4.3 in the paper
    def stellar_count_norm_f(self,hyperparams):
        rho1,rho2,rho3,rho4,GzA,GzB,gz1,gz2,gz3,z0,vz0 = hyperparams
        res = 0.
        for i in range(len(self.effectivearea_zsun)):
            z = self.effectivearea_zsun[i]+z0
            res += self.nuofz(z,hyperparams)*self.effectivearea[i]
        return res
    
    # this creates an interpolated function of the gravitational potential as a function of height z
    def set_phiofz(self,hyperparams):
        rho1,rho2,rho3,rho4,GzA,GzB,gz1,gz2,gz3,z0,vz0 = hyperparams
        z_vec = np.linspace(0.,0.4,40001)
        force_vec = np.zeros(len(z_vec))
        phi_vec = np.zeros(len(z_vec))
        rho_vec = np.zeros(len(z_vec))
        rho_vec[0] = rho1+rho2+rho3+rho4
        for i in range(1,len(z_vec)):
            rho_vec[i] += rho1/np.cosh(z_vec[i]/0.030)**2. #*math.exp(-z_vec[i]**2./(2.*0.030**2.))
            rho_vec[i] += rho2/np.cosh(z_vec[i]/0.060)**2. #*math.exp(-z_vec[i]**2./(2.*0.060**2.))
            rho_vec[i] += rho3/np.cosh(z_vec[i]/0.120)**2. #*math.exp(-z_vec[i]**2./(2.*0.120**2.))
            rho_vec[i] += rho4
            force_vec[i] = force_vec[i-1]+2e-2/37.*( (rho_vec[i]+rho_vec[i-1])/2.)
            phi_vec[i] = phi_vec[i-1]+1e-2*(force_vec[i]+force_vec[i-1])/2.
        self.phiofz = interp1d(z_vec,phi_vec)

    # the stellar number density as a function gravitational potential, see Eq. 5 in the paper
    def nuofphi(self,phi,hyperparams):
        rho1,rho2,rho3,rho4,GzA,GzB,gz1,gz2,gz3,z0,vz0 = hyperparams
        return (1.-GzA-GzB)*np.exp(-phi/gz1**2.)+GzA*np.exp(-phi/gz2**2.)+GzB*np.exp(-phi/gz3**2.)
    
    # the stellar number density as a function height z, see Eq. 5 in the paper
    def nuofz(self,z,hyperparams):
        phi = self.phiofz(abs(z))
        return self.nuofphi(phi,hyperparams)
    
    # the unnormalized log posterior, see Eq. 15 in the paper
    # par_in_num -- specifies the number of steps by which the numerical integration over parallax is computed
    # thinningfactor -- a thinning factor for the number of stars considered, to be used for a quicker estimate of the posterior density
    def lnposterior(self,hyperparams,par_int_num=20,thinningfactor=1):
        rho1,rho2,rho3,rho4,GzA,GzB,gz1,gz2,gz3,z0,vz0 = hyperparams
        self.set_phiofz(hyperparams)
        if rho1<0. or rho2<0. or rho3<0. or rho4<0.:
            return -np.inf
        elif GzA<0. or GzA>1. or GzB<0. or GzB>1.:
            return -np.inf
        elif gz1<0. or gz2<gz1 or gz3<gz2:
            return -np.inf
        elif abs(z0)>0.08 or gz3>200.:
            return -np.inf
        def lnprior(hyperparams):
            return 1.
        res = lnprior(hyperparams)
        stellar_count_norm = self.stellar_count_norm_f(hyperparams)
        for i in range(0,self.number_of_objects,thinningfactor):
            res += np.log( self.obj_posterior(i,hyperparams,par_int_num) )
            res -= np.log( stellar_count_norm )
        return res
    
    # the unnormalized posterior of the ith stellar object, i.e. the integrand of Eq. 15 in the paper
    # par_in_num -- specifies the number of steps by which the numerical integration over parallax is computed
    # thinningfactor -- a thinning factor for the number of stars considered, to be used for a quicker estimate of the posterior density
    def obj_posterior(self,i,hyperparams,par_int_num):
        rho1,rho2,rho3,rho4,GzA,GzB,gz1,gz2,gz3,z0,vz0 = hyperparams
        l,b,par,mul,mub,vlos,sigma_mul,sigma_mub,sigma_par,mulmubcorr,mulparcorr,mubparcorr,sigma_vlos,Gmag = self.data_set[i]
        # this rotation matrix goes from (vl,vb,vlos) to (vx,vy,vz)
        rot_matrix = np.matrix( [   [-math.sin(pi/180.*l),  -math.cos(pi/180.*l)*math.sin(pi/180.*b),   math.cos(pi/180.*l)*math.cos(pi/180.*b)],
                                    [math.cos(pi/180.*l),   -math.sin(pi/180.*l)*math.sin(pi/180.*b),   math.sin(pi/180.*l)*math.cos(pi/180.*b)],
                                    [0.,            math.cos(pi/180.*b),                math.sin(pi/180.*b)]            ] )
        rot_matrix_inv = rot_matrix.getI()
        res = 0.
        if par_int_num==1:
            dist_vec = [1./par]
        else:
            dist_width = 2.5*np.sqrt(par_int_num/20.)
            dist_vec = np.linspace(1./(par+dist_width*sigma_par),min(1./(par-dist_width*sigma_par),0.295),par_int_num)
        for dist in dist_vec:
            rel_par_shift = (1./dist-par)/sigma_par
            mul_shift = rel_par_shift*mulparcorr*sigma_mul
            mub_shift = rel_par_shift*mubparcorr*sigma_mub
            z = math.sin(pi/180.*b)*dist+z0
            phi = self.phiofz(abs(z))
            zcorr_1 = np.exp(-phi/gz1**2.)
            zcorr_2 = np.exp(-phi/gz2**2.)
            zcorr_3 = np.exp(-phi/gz3**2.)
            dist_lh = np.exp(-(1./dist-par)**2./(2.*sigma_par**2.))/np.sqrt(2.*pi*sigma_par**2.)
            if self.dust:
                dist_pc = min(295.,1e3*dist)
                xyzcoords = [dist_pc*math.cos(pi/180.*l)*math.cos(pi/180.*b), dist_pc*math.sin(pi/180.*l)*math.cos(pi/180.*b), dist_pc*math.sin(pi/180.*b)]
                absG = Gmag-5.*(np.log10(1e3*dist)-1.)-self.red_ext*self.BmV_interp(xyzcoords)
            else:
                absG = Gmag-5.*(np.log10(1e3*dist)-1.)
            if (self.in_plane and abs(b)>5.):
                total_amplitude = (1.-GzA-GzB)*zcorr_1+GzA*zcorr_2+GzB*zcorr_3
                res += total_amplitude*dist_lh*dist**2.*self.luminosityPDF(absG)
            else:
                if np.isnan(vlos): # case if line-of-sight velocity IS NOT available
                    cov_vlvb = np.matrix([
                        [(self.k*dist*sigma_mul)**2.*(1.-mulparcorr**2.), (self.k*dist)**2.*sigma_mul*sigma_mub*(mulmubcorr-mulparcorr*mubparcorr)],
                        [(self.k*dist)**2.*sigma_mul*sigma_mub*(mulmubcorr-mulparcorr*mubparcorr), (self.k*dist*sigma_mub)**2.*(1.-mubparcorr**2.)]     ])
                    sph_vels = np.array([self.k*dist*(mul+mul_shift),self.k*dist*(mub+mub_shift)])
                    for ixy in range(len(self.vxvy_means)):
                        vxvy_amp = self.vxvy_amps[ixy]
                        vxvy_mean = self.vxvy_means[ixy]
                        vxvy_covar = self.vxvy_covars[ixy]
                        svel = np.array([-vxvy_mean[0],-vxvy_mean[1],vz0])
                        svel_sphrot = rot_matrix_inv.dot( svel ).A1
                        for iz in [[(1.-GzA-GzB)*zcorr_1,gz1],[GzA*zcorr_2,gz2],[GzB*zcorr_3,gz3]]:
                            amplitude = vxvy_amp*iz[0]
                            cov_vzdist = np.matrix([    [vxvy_covar[0,0],   vxvy_covar[0,1],    0.       ],
                                                        [vxvy_covar[1,0],   vxvy_covar[1,1],    0.       ],
                                                        [0.,                0.,                 iz[1]**2.]   ]   )
                            cov_vzdist_rot = rot_matrix_inv.dot( cov_vzdist.dot(rot_matrix) )
                            res += amplitude*self.multivariate_normal(sph_vels+svel_sphrot[0:2],cov_vlvb+cov_vzdist_rot[0:2,0:2])*dist_lh*dist**2.*self.luminosityPDF(absG)
                else: # case if line-of-sight velocity IS available
                    cov_vlvbvlos = np.matrix([
                        [(self.k*dist*sigma_mul)**2.*(1.-mulparcorr**2.), (self.k*dist)**2.*sigma_mul*sigma_mub*(mulmubcorr-mulparcorr*mubparcorr), 0.],
                        [(self.k*dist)**2.*sigma_mul*sigma_mub*(mulmubcorr-mulparcorr*mubparcorr), (self.k*dist*sigma_mub)**2.*(1.-mubparcorr**2.), 0.],
                        [0., 0., sigma_vlos**2.]                ])
                    cov_uvw = rot_matrix.dot( cov_vlvbvlos.dot(rot_matrix_inv) )
                    uvw = rot_matrix.dot( np.array([self.k*dist*(mul+mul_shift),self.k*dist*(mub+mub_shift),vlos]) ).A1
                    for ixy in range(len(self.vxvy_means)):
                        vxvy_amp = self.vxvy_amps[ixy]
                        vxvy_mean = self.vxvy_means[ixy]
                        vxvy_covar = self.vxvy_covars[ixy]
                        svel = np.array([-vxvy_mean[0],-vxvy_mean[1],vz0])
                        for iz in [[(1.-GzA-GzB)*zcorr_1,gz1],[GzA*zcorr_2,gz2],[GzB*zcorr_3,gz3]]:
                            amplitude = vxvy_amp*iz[0]
                            cov_vzdist = np.matrix([    [vxvy_covar[0,0],   vxvy_covar[0,1],    0.       ],
                                                        [vxvy_covar[1,0],   vxvy_covar[1,1],    0.       ],
                                                        [0.,                0.,                 iz[1]**2.]   ]   )
                            res += amplitude*self.multivariate_normal(uvw+svel,cov_uvw+cov_vzdist)*dist_lh*dist**2.*self.luminosityPDF(absG)
        return res
