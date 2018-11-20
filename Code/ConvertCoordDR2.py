import numpy as np
import math
pi=math.pi


# NGP is north galactic pole
# numbers from https://gea.esac.esa.int/archive/documentation/GDR2/Data_processing/chap_cu3ast/sec_cu3ast_intro/ssec_cu3ast_intro_tansforms.html
alphaNGP=192.85948/180.*pi
deltaNGP=27.12825/180.*pi
#pos of north celestial pole
theta0=122.93192/180.*pi
#make matrices
Tsub1=[[math.cos(theta0),math.sin(theta0),0.],[math.sin(theta0),-math.cos(theta0),0.],[0.,0.,1.]]
Tsub2=[[-math.sin(deltaNGP),0.,math.cos(deltaNGP)],[0.,-1.,0.],[math.cos(deltaNGP),0.,math.sin(deltaNGP)]]
Tsub3=[[math.cos(alphaNGP),math.sin(alphaNGP),0.],[math.sin(alphaNGP),-math.cos(alphaNGP),0.],[0.,0.,1.]]
matrixT=np.dot(Tsub1,np.dot(Tsub2,Tsub3))
matrixTinv=np.linalg.inv(matrixT)
solarvel=[11.1,12.24,7.25]#km/s, from Schonrich, Binney, Dehnen (2010)
#solarvel=[10.,24.,7.25]#km/s, Bovy et al. (2015)
oort=[15.3,-11.9,-3.2,-3.3]#km/s/kpc
#alpha and delta are equitorial coordinates
def radec2lb(alpha,delta):
    rhs=np.dot(matrixT,[math.cos(delta)*math.cos(alpha),math.cos(delta)*math.sin(alpha),math.sin(delta)])
    b=math.asin(rhs[2])
    cosl=rhs[0]/math.cos(b)
    sinl=rhs[1]/math.cos(b)
    if abs(sinl)<0.6:
        l=math.asin(sinl)
        if cosl<0.:
            l=pi-l
        if l<0.:
            l+=2.*pi
    else:
        l=math.acos(cosl)
        if sinl<0.:
            l=2.*pi-l
    return [l,b]
def lb2radec(l,b):
    rhs=np.dot(matrixTinv,[math.cos(b)*math.cos(l),math.cos(b)*math.sin(l),math.sin(b)])
    dec=math.asin(rhs[2])
    cosra=rhs[0]/math.cos(dec)
    sinra=rhs[1]/math.cos(dec)
    if abs(sinra)<0.6:
        ra=math.asin(sinra)
        if cosra<0.:
            ra=pi-ra
        if ra<0.:
            ra+=2.*pi
    else:
        ra=math.acos(cosra)
        if sinra<0.:
            ra=2.*pi-ra
    return [ra,dec]
def getmatrixA(alpha,delta):
    Asub1=[[math.cos(alpha),math.sin(alpha),0.],[math.sin(alpha),-math.cos(alpha),0.],[0.,0.,-1.]]
    Asub2=[[math.cos(delta),0.,-math.sin(delta)],[0.,-1.,0.],[-math.sin(delta),0.,-math.cos(delta)]]
    matrixA=np.dot(Asub1,Asub2)
    return matrixA
def getgalmu(alpha,delta,mualpha,mudelta,covar=None):#returns mu_l and mu_b and errors
    c1=math.sin(deltaNGP)*math.cos(delta)-math.cos(deltaNGP)*math.sin(delta)*math.cos(alpha-alphaNGP)
    c2=math.cos(deltaNGP)*math.sin(alpha-alphaNGP)
    norm=math.sqrt(c1**2.+c2**2.)
    c1=c1/norm
    c2=c2/norm
    matrixC=np.matrix( [[c1,c2],[-c2,c1]] )
    mu=np.array( matrixC.dot([mualpha,mudelta]) )[0]
    if np.shape(covar)==(3,3):
        matrixC33=np.matrix( [[c1,c2,0.],[-c2,c1,0.],[0.,0.,1.]] )
        covar_out = matrixC33.dot(  covar.dot( matrixC33.getI() )  )
        return mu,covar_out
    else:
        return mu
#mu's are equitorial proper motions, rho is radial velocity
def getUVW(alpha,delta,mualpha,mudelta,parallax,rho,solarmotion=False):
    #if solarmotion is set to True (default), the Sun's motion is detracted from the UVW velocity
    k=4.74057 #to km/s from astronomical unit per tropical year
    matrixB=np.dot(matrixT,getmatrixA(alpha,delta))
    vec=[rho,k*mualpha/parallax,k*mudelta/parallax]
    uvw=np.dot(matrixB,vec)#+solarmotion*solarvel
    return uvw