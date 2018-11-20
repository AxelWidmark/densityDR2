import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import math
from scipy.interpolate import UnivariateSpline
from scipy.integrate import quad
pi=math.pi


# This function takes a Gaia data catalogue (specify the path2catalogue)
# and makes a data sample. Input catalogue must be in csv format,
# with columns listed below.
def make_data_set(absGlims,rlims,path2catalogue):
    from ConvertCoordDR2 import getgalmu
    df = pd.read_csv(path2catalogue)
    # global parallax offset
    df['Plx'] = df['Plx']+0.03
    if self.dust:
        from scipy.interpolate import RegularGridInterpolator
        npzfile = np.load("../Data/3D_dust_interpolation.npz")
        x = npzfile['x']
        y = npzfile['y']
        z = npzfile['z']
        BmV_matrix = npzfile['BmV_matrix']
        self.BmV_interp = RegularGridInterpolator((x,y,z), BmV_matrix, method='linear')
        absGmags = []
        for i in range(df.shape[0]):
            l_rad = pi/180.*df['GLON'][i]
            b_rad = pi/180.*df['GLAT'][i]
            dist_pc = 1e3/df['Plx'][i]
            xyzcooords = [dist_pc*np.cos(l_rad)*np.cos(b_rad), dist_pc*np.sin(l_rad)*np.cos(b_rad), dist_pc*np.sin(b_rad)]
            absGmags.append( df['Gmag'][i]-5.*(np.log10(1e3/df['Plx'][i])-1.)-self.red_ext*self.BmV_interp(xyzcooords)[0] )
        df['absG'] = absGmags
    else:
        df['absG'] = df['Gmag']-5.*(np.log10(1e3/df['Plx'])-1.)
    df2 = df[  (df['absG']>absGlims[0]) & (df['absG']<absGlims[1]) & (df['Plx']>1./rlims[1]) & (df['Plx']<1./rlims[0]) ]
    if self.clean:
        df2 = df2[ (df2['epsi']<1.) ]
    mul = np.zeros(df2.shape[0])
    mub = np.zeros(df2.shape[0])
    sigma_mul = np.zeros(df2.shape[0])
    sigma_mub = np.zeros(df2.shape[0])
    mulmubcorr = np.zeros(df2.shape[0])
    mulparcorr = np.zeros(df2.shape[0])
    mubparcorr = np.zeros(df2.shape[0])
    for i in range(df2.shape[0]):
        index_i = df2.index[i]
        covar_pre = np.matrix([ [df['e_pmRA'][index_i]**2.,         df['e_pmRA'][index_i]*df['e_pmDE'][index_i]*df['pmRApmDEcor'][index_i],         df['e_pmRA'][index_i]*df['e_Plx'][index_i]*df['PlxpmRAcor'][index_i]],
                                [df['e_pmRA'][index_i]*df['e_pmDE'][index_i]*df['pmRApmDEcor'][index_i],                df['e_pmDE'][index_i]**2.,              df['e_pmDE'][index_i]*df['e_Plx'][index_i]*df['PlxpmDEcor'][index_i]],
                                [df['e_pmRA'][index_i]*df['e_Plx'][index_i]*df['PlxpmRAcor'][index_i],              df['e_pmDE'][index_i]*df['e_Plx'][index_i]*df['PlxpmDEcor'][index_i],             df['e_Plx'][index_i]**2.],        ] )
        mus_i,covar_i = getgalmu(pi/180.*df['RA_ICRS'][index_i],  pi/180.*df['DE_ICRS'][index_i],  df['pmRA'][index_i],  df['pmDE'][index_i],  covar_pre)
        mul[i] = mus_i[0]
        mub[i] = mus_i[1]
        sigma_mul[i] = math.sqrt(covar_i[0,0])
        sigma_mub[i] = math.sqrt(covar_i[1,1])
        sigma_par = math.sqrt(covar_i[2,2])
        mulmubcorr[i] = covar_i[0,1]/(sigma_mul[i]*sigma_mub[i])
        mulparcorr[i] = covar_i[0,2]/(sigma_mul[i]*sigma_par)
        mubparcorr[i] = covar_i[1,2]/(sigma_mub[i]*sigma_par)
    data_set = np.transpose([ df2['GLON'].values, df2['GLAT'].values, df2['Plx'].values, mul, mub,
                    df2['RV'].values, sigma_mul, sigma_mub, df2['e_Plx'].values,
                    mulmubcorr, mulparcorr, mubparcorr, df2['e_RV'].values,df2['Gmag'].values      ])
    np.save('../Data/dataset_absGlims-'+str(absGlims[0])+'-'+str(absGlims[1])+'_rlims-'+str(rlims[0])+'-'+str(rlims[1])+self.dust*'_dust'+self.clean*'_clean',data_set)
    


def make_toy_data_set(absGlims,rlims):
    true_params = [0.05,0.05,0.03,0.03,0.50,0.05,9.,20.,50.,0.,7.2]
    rho1,rho2,rho3,rho4,GzA,GzB,gz1,gz2,gz3,z0,vz0 = true_params
    
    sample_size = 24000
    from HierarchicalModel3 import stellarsample
    truemodel = stellarsample(absGlims=absGlims,dust=True)
    truemodel.set_phiofz(true_params)
    
    # load the corresponding true data set (to mimic uncertainty distributions etc)
    data_set = np.load('../Data/dataset_absGlims-'+str(absGlims[0])+'-'+str(absGlims[1])+'_rlims-'+str(rlims[0])+'-'+str(rlims[1])+'_dust.npy')
    
    # horizontal velocity distribution
    npzfile = np.load('../Data/vx-vy_GMM.npz')
    vxvy_amps = npzfile['amps']
    vxvy_means = npzfile['means']
    vxvy_covars = npzfile['covars']
    def random_vxvy():
        rand_vel = np.random.uniform(low=0.,high=np.sum(vxvy_amps))
        if rand_vel<np.sum(vxvy_amps[0:1]):
            i_vz = 0
        elif rand_vel<np.sum(vxvy_amps[0:2]):
            i_vz = 1
        elif rand_vel<np.sum(vxvy_amps[0:3]):
            i_vz = 2
        elif rand_vel<np.sum(vxvy_amps[0:4]):
            i_vz = 3
        elif rand_vel<np.sum(vxvy_amps[0:5]):
            i_vz = 4
        return np.random.multivariate_normal(mean=vxvy_means[i_vz],cov=vxvy_covars[i_vz])
    def make_rot_matrix(l,b):
        return np.matrix( [     [-math.sin(pi/180.*l),  -math.cos(pi/180.*l)*math.sin(pi/180.*b),   math.cos(pi/180.*l)*math.cos(pi/180.*b)],
                                [math.cos(pi/180.*l),   -math.sin(pi/180.*l)*math.sin(pi/180.*b),   math.sin(pi/180.*l)*math.cos(pi/180.*b)],
                                [0.,            math.cos(pi/180.*b),                math.sin(pi/180.*b)]            ] )
    
    RVerrs = data_set[:,12]
    RVerrs = RVerrs[np.isnan(RVerrs)==False]
    RVerrpercentiles = [np.percentile(RVerrs,xp) for xp in np.linspace(0.5,99.5,100)]
    mulerrpercentiles = [np.percentile(data_set[:,6],xp) for xp in np.linspace(0.5,99.5,100)]
    muberrpercentiles = [np.percentile(data_set[:,7],xp) for xp in np.linspace(0.5,99.5,100)]
    
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
    npzfile = np.load('../Data/absG_histogram.npz')
    bin_means = npzfile['bin_means']
    histvals = npzfile['histvals']
    # PDF luminosity function in absG (not normalized)
    luminosityPDF = UnivariateSpline(bin_means,histvals,s=5e5)
    
    # completeness inferred from 2MASS xmatch
    import healpy as hp
    npzfile = np.load("../Data/completeness.npz")
    map8to12 = npzfile['map8to12']
    map12to15 = npzfile['map12to15']
    def completeness(l,b,Gmag):
        if Gmag>8. and Gmag<12.:
            return hp.pixelfunc.get_interp_val(map8to12,b+pi/2.,l,nest=True)
        if Gmag>12. and Gmag<15.:
            return hp.pixelfunc.get_interp_val(map12to15,b+pi/2.,l,nest=True)
    
    # dust map
    from scipy.interpolate import RegularGridInterpolator
    npzfile = np.load("../Data/3D_dust_interpolation.npz")
    x = npzfile['x']
    y = npzfile['y']
    z = npzfile['z']
    BmV_matrix = npzfile['BmV_matrix']
    BmV_interp = RegularGridInterpolator((x,y,z), BmV_matrix, method='linear')
    red_ext = 3.1
    
    mock_data_set = []
    absGrange = [absGlims[0]-0.6, absGlims[1]+0.6]
    rhorange = [1./(1./rlims[0]+1.5), 1./(1./rlims[1]-1.5)]
    while len(mock_data_set)<sample_size:
        absG = np.random.uniform(low=absGrange[0],high=absGrange[1])
        if np.random.rand()<absG/luminosityPDF(absGrange[1]): # rejection sample for absG
            b_radians = np.arcsin( np.random.uniform(low=-1.,high=1.) )
            l = np.random.uniform(0.,360.)
            b = b_radians*180./pi
            r = np.random.uniform(low=rhorange[0]**3.,high=rhorange[1]**3.)**(1./3.)
            dist_pc = 1e3*r
            xyzcoords = [dist_pc*math.cos(pi/180.*l)*math.cos(pi/180.*b), dist_pc*math.sin(pi/180.*l)*math.cos(pi/180.*b), dist_pc*math.sin(pi/180.*b)]
            Gmag = absG+5.*(np.log10(1e3*r)-1.)+red_ext*BmV_interp(xyzcoords)
            # completeness
            if Gmag>8. and Gmag<15. and np.random.rand()<completeness(l*pi/180.,b_radians,Gmag):
                rot_matrix = make_rot_matrix(l,b)
                rot_matrix_inv = rot_matrix.getI()
                z = r*np.sin(b_radians)
                # rejection sampling of stellar density
                if np.random.rand()<truemodel.nuofz(z,true_params):
                    mulparcorr = 0.22*np.random.normal()
                    mubparcorr = 0.22*np.random.normal()
                    mulmubcorr = mulparcorr*mubparcorr+0.25*np.random.normal()                
                    if abs(mulmubcorr)<1. and abs(mulparcorr)<1. and abs(mubparcorr)<1. and (mulmubcorr-mulparcorr*mubparcorr)**2./((1.-mulparcorr**2.)*(1.-mubparcorr**2.))<1.:
                        phi = truemodel.phiofz(abs(z))
                        velmix = [(1.-GzA-GzB)*np.exp(-phi/gz1**2.),GzA*np.exp(-phi/gz2**2.),GzB*np.exp(-phi/gz3**2.)]
                        rand_vel = np.random.uniform(low=0.,high=np.sum(velmix))
                        if rand_vel<velmix[0]:
                            vz = gz1*np.random.normal()-vz0
                        elif rand_vel<velmix[0]+velmix[1]:
                            vz = gz2*np.random.normal()-vz0
                        elif rand_vel<velmix[0]+velmix[1]+velmix[2]:
                            vz = gz3*np.random.normal()-vz0
                        vx,vy = random_vxvy()
                        vl,vb,true_vlos = rot_matrix_inv.dot( [vx,vy,vz] ).A1
                        k = 4.74057
                        true_mul = vl/(k*r)
                        true_mub = vb/(k*r)
                        error_percentile = np.random.randint(low=0,high=100)
                        sigma_par = get_parerrpercentiles(Gmag,b_radians)[error_percentile]
                        sigma_mul = mulerrpercentiles[error_percentile]
                        sigma_mub = muberrpercentiles[error_percentile]
                        sigma_vlos = RVerrpercentiles[np.random.randint(low=0,high=100)]
                        error_covar = np.matrix([   [sigma_mul**2., mulmubcorr*sigma_mul*sigma_mub,     sigma_mul*sigma_par*mulparcorr ],
                                                    [mulmubcorr*sigma_mul*sigma_mub,    sigma_mub**2.,  sigma_mub*sigma_par*mubparcorr ],
                                                    [sigma_mul*sigma_par*mulparcorr,    sigma_mub*sigma_par*mubparcorr,     sigma_par**2.]   ]   )
                        mul,mub,par = np.random.multivariate_normal(mean=[true_mul,true_mub,1./r],cov=error_covar)
                        vlos = true_vlos+sigma_vlos*np.random.normal()
                        obs_dist_pc = 1e3/par
                        if 1./par>rlims[0] and 1./par<rlims[1]:
                            xyzcoords = [obs_dist_pc*math.cos(pi/180.*l)*math.cos(pi/180.*b), obs_dist_pc*math.sin(pi/180.*l)*math.cos(pi/180.*b), obs_dist_pc*math.sin(pi/180.*b)]
                            obs_absG = Gmag-5.*(np.log10(1e3/par)-1.)-red_ext*BmV_interp(xyzcoords)
                            # now sample selection
                            if obs_absG>absGrange[0] and obs_absG<absGrange[1]:
                                if np.random.rand()>0.80:
                                    vlos = np.nan
                                    sigma_vlos = np.nan
                                mock_data_set.append([l,b,par,mul,mub,vlos,sigma_mul,sigma_mub,sigma_par,mulmubcorr,mulparcorr,mubparcorr,sigma_vlos,Gmag])
                                if len(mock_data_set)%100==0:
                                    print(len(mock_data_set))
    np.save('../Data/mock_dataset_absGlims-'+str(absGlims[0])+'-'+str(absGlims[1])+'_rlims-'+str(rlims[0])+'-'+str(rlims[1])+'_dust',mock_data_set)