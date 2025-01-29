# This file is simply a repackaging of the functions and math found in CovaPT
# Source, Jay Wadekar: https://github.com/JayWadekar/CovaPT

import os
import scipy
from scipy.integrate import quad
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from numpy import pi
from scipy.misc import derivative
import itertools

from covapt_mt import T0

class covariance_model():
    """
    Class that defines the EFTofLSS model used to calculate predictions
    for the galaxy power spectrum multipole covariance matrices.
    Said model is cureently being upgraded to handle multi-tracer covariances for SPHEREx,
    and draws heavily from CovaPT (https://github.com/JayWadekar/CovaPT)
    
    """

    def __init__(self, num_tracers, num_zbins, k_array_file, alpha=None,
                 window_dir=""):
        """Initializes power spectrum and covariance model
        
        Args:
            z: effective / mean redshift of the sample
            k: np array of k bin centers
            alpha: ratio of number of objects in the galaxy vs random catalogs. Should be less than 1
            window_dir: location of precomputed window functions. Detault directory provided by the repo
            key: specific data sample to model. Must be one of ["HighZ_NGC", "HighZ_SGC", "LowZ_NGZ", "LowZ_SGC"]
        """
        self.num_tracers = num_tracers
        self.num_zbins = num_zbins
        self.set_k_bins(k_array_file)
        self.set_number_densities(alpha)
        self._load_G_window_functions(window_dir)
        #self._load_NG_window_functions(key, window_dir)

        #self.set_survey_pars(alpha) # <- this is for NG covariance only!
        self.pk_galaxy = None

        # number of k-bins on each side of the diaganal to calculate
        # Should be kept small, since the Gaussian covariance drops quickly away from the diag
        self.delta_k_max = 3

        # vectorize some of the more expensive functions
        self.vec_Z12Multipoles = np.vectorize(self.Z12Multipoles)
        self.vec_trisp = np.vectorize(self.trisp)

    # NOTE: This function is here for completeness, but is NOT necesary for making
    # multi-tracer covariances
    def _load_NG_window_functions(self, key:str, window_dir:str):

        try:
            data=np.load(window_dir+'WindowPowers_'+key+'.npy')
            self.kwin = data[0]
            self.powW22 = data[1:7]
            self.powW10 = data[7:13]
            self.powW22x10 = data[13:]
        except IOError: # try the old file format instead
            self.powW22=np.loadtxt(window_dir+'WindowPower_W22_'+key+'.dat')
            self.kwin = self.powW22[:,0]
            self.powW10=np.loadtxt(window_dir+'WindowPower_W10_'+key+'.dat')
            self.powW22x10=np.loadtxt(window_dir+'WindowPower_W22xW10_'+key+'.dat')

    #-------------------------------------------------------------------
    def _load_G_window_functions(self, window_dir:str):
        """Loads gaussian window functions from file"""
        # Loading window power spectra calculated from the survey random catalog (code will be uploaded in a different notebook)
        # These are needed to calculate the sigma^2 terms
        # Columns are k P00 P02 P04 P22 P24 P44 | P00 P02 P04 P22 P24 P44 | P00 P02 P04 P20 P22 P24 P40 P42 P44
        # First section of these is for W22, second for W10 and third for W22xW10

        #Using the window kernels calculated from the survey random catalog as input
        self.WijFile = []
        for zbin in range(self.num_zbins):

            window_file = window_dir + str(zbin)+".npy"
            print("Loading window file from:", window_file)
            if not os.path.exists(window_file):
                raise IOError("ERROR! Could not find", window_file)
        
            self.WijFile.append(np.load(window_file))

    #-------------------------------------------------------------------
    def load_power_spectrum(self, pk_file):
        print("Loading power spectrum from:", pk_file)
        pk_data = np.load(pk_file)
        if pk_file[-4:] == ".npz":
            pk_galaxy_raw = []
            for z in range(self.num_zbins):
                pk_galaxy_raw.append(pk_data["pk_"+str(z)])
                assert len(self.k[z]) == pk_galaxy_raw[z].shape[2], "ERROR: Mismatched kbin lengths!"
            self.num_spectra = pk_galaxy_raw[0].shape[0]
        else:
            pk_galaxy_raw = pk_data
            # TODO: Come up with more robust way to do this
            pk_galaxy_raw = pk_galaxy_raw.transpose(1, 0, 3, 2)
            self.num_spectra = pk_galaxy_raw.shape[1]

        # reformat into form Elisabeth's code expects
        # TODO: simplify this
        self.pk_galaxy = []
        for z in range(self.num_zbins):
            idx = 0
            pk_galaxy = np.zeros((self.num_tracers, self.num_tracers, 5, self.num_kbins[z]))
            for i, j in itertools.product(range(self.num_tracers), repeat=2):
                if i > j: continue
                pk_galaxy[i, j, 0, :] = pk_galaxy_raw[z][idx, 0, :]
                pk_galaxy[j, i, 0, :] = pk_galaxy_raw[z][idx, 0, :]
                pk_galaxy[i, j, 2, :] = pk_galaxy_raw[z][idx, 1, :]
                pk_galaxy[j, i, 2, :] = pk_galaxy_raw[z][idx, 1, :]
                pk_galaxy[i, j, 4, :] = pk_galaxy_raw[z][idx, 2, :]
                pk_galaxy[j, i, 4, :] = pk_galaxy_raw[z][idx, 2, :]
                idx +=1
            self.pk_galaxy.append(pk_galaxy)
            
    #-------------------------------------------------------------------
    def set_k_bins(self, k_array_file):
        k_data = np.load(k_array_file)
        self.k = []
        self.num_kbins = np.zeros(self.num_zbins, dtype=np.int16)
        for zbin in range(self.num_zbins):
            key = "k_"+str(zbin)
            self.k.append(k_data[key])
            self.num_kbins[zbin] = len(self.k[zbin])

    #-------------------------------------------------------------------
    def get_k_bins(self):
        return self.k
    
    def set_number_densities(self, alpha=None, n_galaxy=None):
        self.alpha_mt = np.zeros((self.num_tracers,self.num_tracers))
        if alpha != None:
            # This code is a work in progress!
            # Tim's idea, alpha for cross-tracer should be 0
            for i in range(self.num_tracers):
                self.alpha_mt[i, i] = alpha[i]
            # for i, j in itertools.product(range(self.num_tracers), repeat=2):
            #     self.alpha_mt[i,j] = (alpha[i] + alpha[j]) / 2.
            #self.alpha_mt = np.diag(alpha)
            print(self.alpha_mt)
        
        if n_galaxy == None: self.invng_mt =np.ones((self.num_tracers,self.num_tracers))
        else:                self.invng_mt = 1 / np.array(n_galaxy)
        
    def set_survey_pars(self, alpha):
        # WARNING: This function should not be used as is!
        raise NotImplementedError
        # The following parameters are calculated from the survey random catalog
        # Using Iij convention in Eq.(3)
        # self.i22 = 454.2155*alpha; self.i11 = 7367534.87288*alpha; self.i12 = 2825379.84558*alpha;
        # self.i10 = 23612072*alpha; self.i24 = 58.49444652*alpha; 
        # self.i14 = 756107.6916375*alpha; self.i34 = 8.993832235e-3*alpha;
        # self.i44 = 2.158444115e-6*alpha; self.i32 = 0.11702382*alpha;
        # self.i12oi22 = 2825379.84558/454.2155; #Effective shot noise

    #-------------------------------------------------------------------
    # Covariance Helper functions
    #-------------------------------------------------------------------

    #-------------------------------------------------------------------
    def Cij(self, kt, Wij, Pfit):
        """Generates intermidiate product for the Gaussian covariance matrix"""
        temp=np.zeros((2*self.delta_k_max+1,6))
        for dk in range(-self.delta_k_max, self.delta_k_max+1):
            if(kt+dk<0 or kt+dk>=self.num_kbins):
                temp[dk+3]=0
                continue
            temp[dk+3]=Wij[dk+3,0]*Pfit[0][kt]*Pfit[0][kt+dk]+\
                       Wij[dk+3,1]*Pfit[0][kt]*Pfit[2][kt+dk]+\
                       Wij[dk+3,2]*Pfit[0][kt]*Pfit[4][kt+dk]+\
                       Wij[dk+3,3]*Pfit[2][kt]*Pfit[0][kt+dk]+\
                       Wij[dk+3,4]*Pfit[2][kt]*Pfit[2][kt+dk]+\
                       Wij[dk+3,5]*Pfit[2][kt]*Pfit[4][kt+dk]+\
                       Wij[dk+3,6]*Pfit[4][kt]*Pfit[0][kt+dk]+\
                       Wij[dk+3,7]*Pfit[4][kt]*Pfit[2][kt+dk]+\
                       Wij[dk+3,8]*Pfit[4][kt]*Pfit[4][kt+dk]+\
                       1.01*(Wij[dk+3,9]*(Pfit[0][kt]+Pfit[0][kt+dk])/2.+\
                       Wij[dk+3,10]*Pfit[2][kt]+Wij[dk+3,11]*Pfit[4][kt]+\
                       Wij[dk+3,12]*Pfit[2][kt+dk]+Wij[dk+3,13]*Pfit[4][kt+dk])+\
                       1.01**2*Wij[dk+3,14]
        return(temp)

    #-------------------------------------------------------------------
    def Cij_MT(self, kt, dk, Wij, Pfit, A = 0, B= 0, C = 0, D = 0):
        """Generates specific elements of the Gaussian covariance matrix

        Returns:
            cov_sub {np array} array of size 6 containing covariance elements corresponding to k + dk. 
            The indices correspond to the following combination of l's:
            0: (0,0), 1: (2,2), 2: (4,4), 3: (0,2), 4: (0,4), 5: (2,4)
        """
        cosmic_variance_term = Wij[dk+self.delta_k_max,0]*(Pfit[A,C,0][kt]*Pfit[B,D,0][kt+dk] + Pfit[A,D,0][kt]*Pfit[B,C,0][kt+dk])/2.+\
                        Wij[dk+self.delta_k_max,1]*(Pfit[A,C,0][kt]*Pfit[B,D,2][kt+dk] + Pfit[A,D,0][kt]*Pfit[B,C,2][kt+dk])/2.+\
                        Wij[dk+self.delta_k_max,2]*(Pfit[A,C,0][kt]*Pfit[B,D,4][kt+dk] + Pfit[A,D,0][kt]*Pfit[B,C,4][kt+dk])/2.+\
                        Wij[dk+self.delta_k_max,3]*(Pfit[A,C,2][kt]*Pfit[B,D,0][kt+dk] + Pfit[A,D,2][kt]*Pfit[B,C,0][kt+dk])/2.+\
                        Wij[dk+self.delta_k_max,4]*(Pfit[A,C,2][kt]*Pfit[B,D,2][kt+dk] + Pfit[A,D,2][kt]*Pfit[B,C,2][kt+dk])/2.+\
                        Wij[dk+self.delta_k_max,5]*(Pfit[A,C,2][kt]*Pfit[B,D,4][kt+dk] + Pfit[A,D,2][kt]*Pfit[B,C,4][kt+dk])/2.+\
                        Wij[dk+self.delta_k_max,6]*(Pfit[A,C,4][kt]*Pfit[B,D,0][kt+dk] + Pfit[A,D,4][kt]*Pfit[B,C,0][kt+dk])/2.+\
                        Wij[dk+self.delta_k_max,7]*(Pfit[A,C,4][kt]*Pfit[B,D,2][kt+dk] + Pfit[A,D,4][kt]*Pfit[B,C,2][kt+dk])/2.+\
                        Wij[dk+self.delta_k_max,8]*(Pfit[A,C,4][kt]*Pfit[B,D,4][kt+dk] + Pfit[A,D,4][kt]*Pfit[B,C,4][kt+dk])/2.
        # Terms with (1+alpha) are 1/nbar like term
        mixed_term = 0.5*(1+self.alpha_mt[A,C])*(Wij[dk+self.delta_k_max,9]*(Pfit[B,D,0][kt]+Pfit[B,D,0][kt+dk])/2.+\
                        Wij[dk+self.delta_k_max,10]*Pfit[B,D,2][kt]   +Wij[dk+self.delta_k_max,11]*Pfit[B,D,4][kt]+\
                        Wij[dk+self.delta_k_max,12]*Pfit[B,D,2][kt+dk]+Wij[dk+self.delta_k_max,13]*Pfit[B,D,4][kt+dk])+\
                        0.5*(1+self.alpha_mt[A,D])*(Wij[dk+self.delta_k_max,9]*(Pfit[B,C,0][kt]+Pfit[B,C,0][kt+dk])/2.+\
                        Wij[dk+self.delta_k_max,10]*Pfit[B,C,2][kt]   +Wij[dk+self.delta_k_max,11]*Pfit[B,C,4][kt]+\
                        Wij[dk+self.delta_k_max,12]*Pfit[B,C,2][kt+dk]+Wij[dk+self.delta_k_max,13]*Pfit[B,C,4][kt+dk])
        # Term with (1+alpha)**2 is 1/nbar**2 like term
        shotnoise_term = ((1+self.alpha_mt[A,C])*(1+self.alpha_mt[B,D])+(1+self.alpha_mt[A,D])*(1+self.alpha_mt[B,C]))/2.*Wij[dk+self.delta_k_max,14] 
        
        cov_sub = cosmic_variance_term + mixed_term + shotnoise_term
        return(cov_sub)

    #-------------------------------------------------------------------
    def Dz(self, z,Om0):
        """Calculates the LambdaCDM growth factor D(z, Om0)

        Args:
            z: cosmological redshift
            Om0: matter density parameter
        """
        return(scipy.special.hyp2f1(1/3., 1, 11/6., (1-1/Om0)/(1+z)**3)
                                    /scipy.special.hyp2f1(1/3., 1, 11/6., 1-1/Om0)/(1+z))

    #-------------------------------------------------------------------
    def fgrowth(self, z,Om0):
        """Calculates the LambdaCDM growth rate f_growth(z, Om0)

        Args:
            z: cosmological redshift
            Om0: matter density parameter
        """
        return(1. + 6*(Om0-1)*scipy.special.hyp2f1(4/3., 2, 17/6., (1-1/Om0)/(1+z)**3)
                    /( 11*Om0*(1+z)**3*scipy.special.hyp2f1(1/3., 1, 11/6., (1-1/Om0)/(1+z)**3) ))

    #-------------------------------------------------------------------
    def trispIntegrand(self, u12,k1, k2, l1, l2, Plin):
        """Integrand function for use when calculating the trispectrum"""
        return( (8*self.i44*(Plin(k1)**2*T0.e44o44_1(u12,k1,k2,l1,l2) + Plin(k2)**2*T0.e44o44_1(u12,k2,k1,l1,l2))
                +16*self.i44*Plin(k1)*Plin(k2)*T0.e44o44_2(u12,k1,k2,l1,l2)
                +8*self.i34*(Plin(k1)*T0.e34o44_2(u12,k1,k2,l1,l2)+Plin(k2)*T0.e34o44_2(u12,k2,k1,l1,l2))
                +2*self.i24*T0.e24o44(u12,k1,k2,l1,l2))
                *Plin(np.sqrt(k1**2+k2**2+2*k1*k2*u12)) )

    #-------------------------------------------------------------------
    def trisp(self, l1:int, l2:int, k1:float, k2:float, Plin):
        """Returns the tree-level trispectrum as a function of multipoles and k1, k2"""
        expr = self.i44*(Plin(k1)**2*Plin(k2)*T0.ez3(k1,k2,l1,l2) + Plin(k2)**2*Plin(k1)*T0.ez3(k2,k1,l1,l2))\
            +8*self.i34*Plin(k1)*Plin(k2)*T0.e34o44_1(k1,k2,l1,l2)

        T_kk = (quad(self.trispIntegrand, -1, 1,args=(k1,k2,l1,l2,Plin), limit=150)[0]/2. + expr)/self.i22**2
        return(T_kk)

    #-------------------------------------------------------------------
    def Z12Kernel(self, l:int, mu,be,b1,b2,g2,dlnpk):
        """Using the Z12 kernel which is defined in Eq. (A9) (equations copy-pasted from Generating_T0_Z12_expressions.nb)"""
        if(l==0):
            exp=(7*b1**2*be*(70 + 42*be + (-35*(-3 + dlnpk) + 3*be*(28 + 13*be - 14*dlnpk - 5*be*dlnpk))*mu**2) + 
                b1*(35*(47 - 7*dlnpk) + be*(798 + 153*be - 98*dlnpk - 21*be*dlnpk + 
                4*(84 + be*(48 - 21*dlnpk) - 49*dlnpk)*mu**2)) + 
                98*(5*b2*(3 + be) + 4*g2*(-5 + be*(-2 + mu**2))))/(735.*b1**2)
        elif(l==2):
            exp=(14*b1**2*be*(14 + 12*be + (2*be*(12 + 7*be) - (1 + be)*(7 + 5*be)*dlnpk)*mu**2) + 
                b1*(4*be*(69 + 19*be) - 7*be*(2 + be)*dlnpk + 
                (24*be*(11 + 6*be) - 7*(21 + be*(22 + 9*be))*dlnpk)*mu**2 + 7*(-8 + 7*dlnpk + 24*mu**2)) + 
                28*(7*b2*be + g2*(-7 - 13*be + (21 + 11*be)*mu**2)))/(147.*b1**2)
        elif(l==4):
            exp=(8*be*(b1*(-132 + 77*dlnpk + be*(23 + 154*b1 + 14*dlnpk)) - 154*g2 + 
                (b1*(396 - 231*dlnpk + be*(272 + 308*b1 + 343*b1*be - 7*(17 + b1*(22 + 15*be))*dlnpk)) + 
                462*g2)*mu**2))/(2695.*b1**2)
        return(exp)

    #-------------------------------------------------------------------
    def lp(self, l:int, mu:float):
        """Returns the legenre polynomial of order l evaluated at mu"""
        if (l==0): exp=1
        if (l==2): exp=((3*mu**2 - 1)/2.)
        if (l==4): exp=((35*mu**4 - 30*mu**2 + 3)/8.)
        return(exp)

    #-------------------------------------------------------------------
    def MatrixForm(self, a):
        """transforms the linear np-array a to a matrix"""
        b=np.zeros((3,3))
        if len(a)==6:
            b[0,0]=a[0]; b[1,0]=b[0,1]=a[1]; 
            b[2,0]=b[0,2]=a[2]; b[1,1]=a[3];
            b[2,1]=b[1,2]=a[4]; b[2,2]=a[5];
        if len(a)==9:
            b[0,0]=a[0]; b[0,1]=a[1]; b[0,2]=a[2]; 
            b[1,0]=a[3]; b[1,1]=a[4]; b[1,2]=a[5];
            b[2,0]=a[6]; b[2,1]=a[7]; b[2,2]=a[8];
        return(b)

    #-------------------------------------------------------------------
    def Z12Multipoles(self, i,l,be,b1,b2,g2,dlnpk):
        """Calculates multipoles of the Z12 kernel"""
        return(quad(lambda mu: self.lp(i,mu)*self.Z12Kernel(l,mu,be,b1,b2,g2,dlnpk), -1, 1, limit=60)[0])

    #-------------------------------------------------------------------
    def CovBC(self, l1, l2, sigma22Sq, sigma10Sq, Plin, dlnPk,
                          be:float,b1:float,b2:float,g2:float):
        """Returns the BC portion of the SSC covariance term"""

        covaBC=np.zeros((len(self.k),len(self.k)))
        for i in range(3):
            for j in range(3):
                covaBC+=1/4.*sigma22Sq[i,j]*np.outer(Plin(self.k)*self.vec_Z12Multipoles(2*i,l1,be,b1,b2,g2,dlnPk),Plin(self.k)*self.vec_Z12Multipoles(2*j,l2,be,b1,b2,g2,dlnPk))
                sigma10Sq[i,j]=1/4.*sigma10Sq[i,j]*quad(lambda mu: self.lp(2*i,mu)*(1 + be*mu**2), -1, 1, limit=60)[0]\
                *quad(lambda mu: self.lp(2*j,mu)*(1 + be*mu**2), -1, 1, limit=80)[0]

        return covaBC

    #-------------------------------------------------------------------
    def CovLA(self, l1, l2, sigma22x10, sigma10Sq, Plin, dlnPk, rsd, 
                  be:float,b1:float,b2:float,g2:float):
        """Returns the LA portion of the SSC covariance term"""

        covaLAterm=np.zeros((3,len(self.k)))
        for l in range(3):
            for i in range(3):
                for j in range(3):
                    covaLAterm[l]+=1/4.*sigma22x10[i,j]*self.vec_Z12Multipoles(2*i,2*l,be,b1,b2,g2,dlnPk)\
                    *quad(lambda mu: self.lp(2*j,mu)*(1 + be*mu**2), -1, 1, limit=80)[0]
        
        covaLA=-rsd[l2]*np.outer(Plin(self.k)*(covaLAterm[int(l1/2)]+self.i32/self.i22/self.i10*rsd[l1]*Plin(self.k)*b2/b1**2+2/self.i10*rsd[l1]),Plin(self.k))\
            -rsd[l1]*np.outer(Plin(self.k),Plin(self.k)*(covaLAterm[int(l2/2)]+self.i32/self.i22/self.i10*rsd[l2]*Plin(self.k)*b2/b1**2+2/self.i10*rsd[l2]))\
            +(np.sum(sigma10Sq)+1/self.i10)*rsd[l1]*rsd[l2]*np.outer(Plin(self.k),Plin(self.k))

        return covaLA
            
    #-------------------------------------------------------------------
    def covaSSC(self, l1:int, l2:int, sigma22Sq, sigma10Sq, sigma22x10, 
                rsd, be:float,b1:float,b2:float,g2:float, Plin, dlnPk):
        """Returns the SSC covariance matrix term for a given l1 and l2

        This term is composed of two effects, those being beat-coupling (BC), 
        and local average effects (LA). Both effects are calculated in 
        seperate helper functions
        
        Args:
            l1: multipole 1
            l2: multipole 2
            be: growth factor scaled by b1
            b1: linear bias
            b2: quadratic bias
            g2: tidal bias (similar to bG2)
            Plin: linear power spectrum
            dlnPk: derivative of the linear matter power spectrum
        """
        covaBC = self.CovBC(l1, l2, sigma22Sq, sigma10Sq, Plin, dlnPk,
                             be, b1, b2, g2)

        covaLA = self.CovLA(l1, l2, sigma22x10, sigma10Sq, Plin, dlnPk,
                             rsd, b2, b1, b2, g2)
        
        return(covaBC+covaLA)

    def get_mt_gaussian_covariance(self, pk_galaxy=None):
        """Returns the (Monopole+Quadrupole) Gaussian covariance matrix

        Args:
            Pk_galaxy: Optional tuple of galaxy power spectrum multipoles [monopole, quadropole]. \
        
        Raises:
            RuntimeError: If generating the power spectrum fails
            AssertionError: If the input galaxy power spectrum is a different size than \
            the k bins
        """
        if pk_galaxy == None:
            pk_galaxy = self.pk_galaxy

        n_multi = int(self.num_tracers*(self.num_tracers+1)/2) # <- total # of auto + cross spectra
        covMat_all = []
        for z in range(self.num_zbins):
            covMat=np.zeros((3*self.num_kbins[z]*n_multi,3*self.num_kbins[z]*n_multi))
            n_AB = -1
            for A, B in itertools.product(range(self.num_tracers), repeat=2):
                if B < A: continue
                n_AB += 1
                n_CD = -1
                for C, D in itertools.product(range(self.num_tracers), repeat=2):
                    if D < C: continue
                    n_CD += 1
                    for k in range(self.num_kbins[z]):
                        for dk in range(-self.delta_k_max,self.delta_k_max+1):
                            if(k+dk>=0 and k+dk<self.num_kbins[z]):
                                # NOTE: This code can be upgraded to give l=4 gaussian covariance by using the other elements of cov_sub
                                cov_sub = self.Cij_MT(k, dk, self.WijFile[z][k], pk_galaxy[z], A=A,B=B,C=C,D=D)
                                blk_idx = self.num_kbins[z]
                                covMat[n_AB*3*blk_idx+k, n_CD*3*blk_idx+k+dk]                        = cov_sub[0]
                                covMat[n_AB*3*blk_idx+blk_idx+k, n_CD*3*blk_idx+blk_idx+k+dk]        = cov_sub[1]
                                covMat[n_AB*3*blk_idx+2*blk_idx+k, n_CD*3*blk_idx+(2*blk_idx)+k+dk]  = cov_sub[2]
                                covMat[n_AB*3*blk_idx+blk_idx+k, n_CD*3*blk_idx+k+dk]                = cov_sub[3]
                                covMat[n_AB*3*blk_idx+k, n_CD*3*blk_idx+blk_idx+k+dk]                = cov_sub[3]
                                covMat[n_AB*3*blk_idx+2*blk_idx+k, n_CD*3*blk_idx+k+dk]              = cov_sub[4]
                                covMat[n_AB*3*blk_idx+2*blk_idx+k, n_CD*3*blk_idx+blk_idx+k+dk]      = cov_sub[5]
                        #covMat[n_AB*2*num_kbins:n_AB*2*num_kbins+num_kbins,n_CD*2*num_kbins+num_kbins:n_CD*2*num_kbins+num_kbins*2]=np.transpose(covMat[num_kbins:num_kbins*2,:num_kbins])
            
            covMat_all.append((covMat+np.transpose(covMat))/2.)
            #covMat_all.append(covMat)
        return(covMat_all)

    #-------------------------------------------------------------------
    def get_gaussian_covariance(self, params, return_Pk=False, Pk_galaxy=[]):
        """Returns the (Monopole+Quadrupole) Gaussian covariance matrix

        Args:
            params: np array of cosmology parameters. Should be ordered  \
            like [H0, omch2, As, b1, b2, bG2, cs0, cs2, cbar, Pshot]
            return_Pk: Whether or not to return the galaxy power spectrum as well
            Pk_galaxy: Optional tuple of galaxy power spectrum multipoles [monopole, quadropole]. \
            If empty, this function calculates it from the input params
        
        Raises:
            RuntimeError: If generating the power spectrum fails
            AssertionError: If the input galaxy power spectrum is a different size than \
            the k bins
        """
        # generate galaxy redshift-space power spectrum if necesary
        if len(Pk_galaxy) == 0:
            Pk_galaxy = self.Pk_CLASS_PT(params)

        # sanity check to make sure the galaxy power spectrum has the correct dimensions
        assert len(Pk_galaxy[0]) == len(self.k), "Galaxy power spectrum has wrong dimensions! Double check your k-bins"

        covMat=np.zeros((2*self.num_kbins,2*self.num_kbins))
        for i in range(self.num_kbins):
            temp=self.Cij(i,self.WijFile[i], Pk_galaxy)
            C00=temp[:,0]; C22=temp[:,1]; C20=temp[:,3]
            for j in range(-3,4):
                if(i+j>=0 and i+j<self.num_kbins):
                    covMat[i,i+j]=C00[j+3]
                    covMat[self.num_kbins+i,self.num_kbins+i+j]=C22[j+3]
                    covMat[self.num_kbins+i,i+j]=C20[j+3]
        covMat[:self.num_kbins,self.num_kbins:self.num_kbins*2]=np.transpose(covMat[self.num_kbins:self.num_kbins*2,:self.num_kbins])
        #covMat=(covMat+np.transpose(covMat))/2.

        if return_Pk == False: return covMat
        else: return covMat, Pk_galaxy

    #-------------------------------------------------------------------
    # def get_non_gaussian_covariance(self, params, seperate_SSC=False):
    #     """Returns the Non-Gaussian portion of the covariance matrix
        
    #     This function specifically calculates both covariance from the regular
    #     trispectrum (T0), and Super-Sample Covariance (SSC). Currently all third-order
    #     bias terms are set to 0. Due to the trispectrum calculations, this process
    #     is compuationally expensive (~minutes)

    #     Args:
    #         params: np array of cosmology parameters. Should be ordered  \
    #         like [H0, omch2, As, b1, b2, bG2]. Recommended to use the same object \
    #         that is passed to get_gaussian_covariance
    #     """
    #     # Cosmology parameters
    #     H0, omch2, As = params[0], params[1], params[2]
    #     ombh2 = self.ombh2_planck
    #     ns = self.ns_planck

    #     # local bias terms
    #     b1 = params[3]
    #     b2 = params[4]
    #     b3 = 0.

    #     # non-local bias terms
    #     g2 = params[5] #<- bG2
    #     g3 = 0.  #<- bG3 (third order?)
    #     g2x = 0. #<- bdG2 (third order)
    #     g21 = 0. #<- bGamma3*

    #     Omega_m = (omch2 + ombh2 + 0.00064) / (H0/100)**2
        
    #     # Below are expressions for non-local bias (g_i) from local lagrangian approximation
    #     # and non-linear bias (b_i) from peak-background split fit of 
    #     # Lazyeras et al. 2016 (rescaled using Appendix C.2 of arXiv:1812.03208),
    #     # which could used if those parameters aren't constrained.
    #     # g2 = -2/7*(b1 - 1)
    #     # g3 = 11/63*(b1 - 1)
    #     # #b2 = 0.412 - 2.143*b1 + 0.929*b1**2 + 0.008*b1**3 + 4/3*g2 
    #     # g2x = -2/7*b2
    #     # g21 = -22/147*(b1 - 1)
    #     # b3 = -1.028 + 7.646*b1 - 6.227*b1**2 + 0.912*b1**3 + 4*g2x - 4/3*g3 - 8/3*g21 - 32/21*g2
        
    #     # ---Growth Factor---
    #     be = self.fgrowth(self.z, Omega_m)/b1; #beta = f/b1, zero for real space

    #     # initializing bias parameters for trispectrum
    #     T0.InitParameters([b1,be,g2,b2,g3,g2x,g21,b3])

    #     # Get initial power spectrum
    #     pdata = self.Pk_lin_CLASS(H0, omch2, ombh2, As, ns)
    #     Plin=InterpolatedUnivariateSpline(pdata[:,0], self.Dz(self.z, Omega_m)**2*b1**2*pdata[:,1])

    #     # Get the derivative of the linear power spectrum
    #     dlnPk=derivative(Plin,self.k,dx=1e-4)*self.k/Plin(self.k)
        
    #     # Kaiser terms
    #     rsd=np.zeros(5)
    #     rsd[0]=1 + (2*be)/3 + be**2/5
    #     rsd[2]=(4*be)/3 + (4*be**2)/7
    #     rsd[4]=(8*be**2)/35
        
    #     # Calculating the RMS fluctuations of supersurvey modes 
    #     #(e.g., sigma22Sq which was defined in Eq. (33) and later calculated in Eq.(65)
    #     [temp,temp2]=np.zeros((2,6)); temp3 = np.zeros(9)
    #     for i in range(9):
    #         #Pwin=InterpolatedUnivariateSpline(self.kwin, self.powW22x10[:,1+i])
    #         Pwin=InterpolatedUnivariateSpline(self.kwin, self.powW22x10[i])
    #         temp3[i]=quad(lambda q: q**2*Plin(q)*Pwin(q)/2/pi**2, 0, self.kwin[-1], limit=100)[0]

    #         if(i<6):
    #             #Pwin=InterpolatedUnivariateSpline(self.kwin, self.powW22[:,1+i])
    #             Pwin=InterpolatedUnivariateSpline(self.kwin, self.powW22[i])
    #             temp[i]=quad(lambda q: q**2*Plin(q)*Pwin(q)/2/pi**2, 0, self.kwin[-1], limit=100)[0]
    #             #Pwin=InterpolatedUnivariateSpline(self.kwin, self.powW10[:,1+i])
    #             Pwin=InterpolatedUnivariateSpline(self.kwin, self.powW10[i])
    #             temp2[i]=quad(lambda q: q**2*Plin(q)*Pwin(q)/2/pi**2, 0, self.kwin[-1], limit=100)[0]
    #         else:
    #             continue
        
    #     sigma22Sq  = self.MatrixForm(temp)
    #     sigma10Sq  = self.MatrixForm(temp2)
    #     sigma22x10 = self.MatrixForm(temp3)
    
    #     # Calculate SSC covariance    
    #     covaSSCmult=np.zeros((2*self.num_kbins,2*self.num_kbins))
    #     covaSSCmult[:self.num_kbins,:self.num_kbins]=self.covaSSC(0,0, sigma22Sq, sigma10Sq, sigma22x10, rsd, be,b1,b2,g2, Plin, dlnPk)
    #     covaSSCmult[self.num_kbins:,self.num_kbins:]=self.covaSSC(2,2, sigma22Sq, sigma10Sq, sigma22x10, rsd, be,b1,b2,g2, Plin, dlnPk)
    #     covaSSCmult[:self.num_kbins,self.num_kbins:]=self.covaSSC(0,2, sigma22Sq, sigma10Sq, sigma22x10, rsd, be,b1,b2,g2, Plin, dlnPk); 
    #     covaSSCmult[self.num_kbins:,:self.num_kbins]=np.transpose(covaSSCmult[:self.num_kbins,self.num_kbins:])

    #     # Calculate the Non-Gaussian multipole covariance
    #     # Warning: the trispectrum takes a while to run
    #     covaT0mult=np.zeros((2*self.num_kbins,2*self.num_kbins))
    #     for i in range(len(self.k)):
    #         covaT0mult[i,:self.num_kbins]=self.vec_trisp(0,0,self.k[i],self.k, Plin)
    #         covaT0mult[i,self.num_kbins:]=self.vec_trisp(0,2,self.k[i],self.k, Plin)
    #         covaT0mult[self.num_kbins+i,self.num_kbins:]=self.vec_trisp(2,2,self.k[i],self.k, Plin)

    #     covaT0mult[self.num_kbins:,:self.num_kbins]=np.transpose(covaT0mult[:self.num_kbins,self.num_kbins:])

    #     #return covaNG
    #     return covaSSCmult + covaT0mult

    # #-------------------------------------------------------------------
    # def get_full_covariance(self, params, Pk_galaxy=[], seperate_terms=False):
    #     """Returns the full analytic covariance matrix

    #     This function sequencially calls get_gaussian_covariance() and 
    #     get_non_gaussian_covariance()

    #     Args:
    #         params: np array of cosmology parameters. Should be ordered  \
    #         like [H0, omch2, As, b1, b2, bG2, cs0, cs2, cbar, Pshot]
    #         Pk_galaxy: Optional tuple of galaxy power spectrum multipoles [monopole, quadropole]. \
    #         If empty, this function calculates it from the input params
    #         seperate_terms: If True, return C_G and C_NG as seperate arrays. Default False
    #     """
    #     cov_G = self.get_gaussian_covariance(params, False, Pk_galaxy)
    #     cov_NG = self.get_non_gaussian_covariance(params)

    #     if seperate_terms: return cov_G, cov_NG
    #     else             : return cov_G + cov_NG
    
    # def get_marginalized_covariance(self, params, C, window_dir=covapt_data_dir):
    #     """Returns the marginalized covariance matrix as well as the convolved model vector
        
    #     taken from Misha Ivanov's Montepython Likelihood 
    #     )https://github.com/Michalychforever/lss_montepython)

    #     Args:
    #         params: {np array} array of input cosmology parameters [h, omch2, As, b1, b2, bG2, cs0, cs2, cbar, Pshot]
    #         C: {2D numpy array} Covariance matrix to marginalize
    #         window_dir: location of window / wide angle functions. Default to the directory \
    #         defined in config.py
    #     Returns:
    #         C_marg: Marginalized covariance matrix
    #         model_vector: Power spectrum multipoles calculated during marginalization
    #         Omega_m: (float) value for matter parameter Omega_m
    #         sigma8: (float) value for the power spectrum normalization sigma8
    #     """

    #     all_theory, theory0, theory2, theory4, fz, Omega_m, sigma8 = self.Pk_CLASS_PT(params, False)
    #     norm = 1; Nmarg = 4
    #     omit = 0; ksize = len(self.k)
    #     if len(self.wmat) == 0:
    #         self.wmat = np.loadtxt(window_dir+"W_ngc_z3.matrix", skiprows = 0)
    #         self.mmat = np.loadtxt(window_dir+"M_ngc_z3.matrix", skiprows = 0)

    #     h    = params[0] / 100.
    #     b1   = params[3]
    #     css0sig = 30.
    #     css2sig = 30.
    #     b4sig = 500.
    #     Pshotsig = 5000.

    #     dtheory4_dcss0 = np.zeros_like(self.k_theory)
    #     dtheory4_dcss2 = np.zeros_like(self.k_theory)
    #     dtheory4_db4 = fz**2*self.k_theory**2*(norm**2*fz**2*48./143. + 48.*fz*b1*norm/77.+8.*b1**2/35.)*(35./8.)*all_theory[13]*h
    #     dtheory4_dPshot = np.zeros_like(self.k_theory)

    #     dtheory2_dcss0 = np.zeros_like(self.k_theory)
    #     dtheory2_dcss2 = (2.*norm**2.*all_theory[12]/h**2.)*h**3.
    #     dtheory2_db4 = fz**2.*self.k_theory**2.*((norm**2.*fz**2.*70. + 165.*fz*b1*norm+99.*b1**2.)*4./693.)*(35./8.)*all_theory[13]*h
    #     dtheory2_dPshot = np.zeros_like(self.k_theory)

    #     dtheory0_dcss0 = (2.*norm**2.*all_theory[11]/h**2.)*h**3.
    #     dtheory0_dcss2 = np.zeros_like(self.k_theory)
    #     dtheory0_db4 = fz**2.*self.k_theory**2.*(norm**2.*fz**2./9. + 2.*fz*b1*norm/7. + b1**2./5)*(35./8.)*all_theory[13]*h
    #     dtheory0_dPshot = np.ones_like(self.k_theory)

    #     theory0vec = np.vstack([theory0,dtheory0_dcss0,dtheory0_dcss2,dtheory0_db4,dtheory0_dPshot])
    #     theory2vec = np.vstack([theory2,dtheory2_dcss0,dtheory2_dcss2,dtheory2_db4,dtheory2_dPshot])
    #     theory4vec = np.vstack([theory4,dtheory4_dcss0,dtheory4_dcss2,dtheory4_db4,dtheory4_dPshot])

    #     #PW = np.zeros((5*self.ksize1,Nmarg+1))
    #     P0int = np.zeros((ksize,Nmarg+1))
    #     P2int = np.zeros((ksize,Nmarg+1))
    #     P4int = np.zeros((ksize,Nmarg+1))

    #     for i in range(Nmarg+1):
    #             Pintvec = np.hstack([theory0vec[i,:],theory2vec[i,:],theory4vec[i,:]]) 
    #             PW = np.matmul(self.wmat,np.matmul(self.mmat,Pintvec))
    #             P0int[:,i] = np.asarray([PW[j] for j in range(omit,omit+ksize)]).T	
    #             P2int[:,i] = np.asarray([PW[j] for j in range(80+omit,80+omit+ksize)]).T
    #             #P4int[:,i] = np.asarray([PW[j] for j in range(160+omit,160+omit+self.ksize)]).T

    #     dcss0_stack = np.hstack([P0int[:,1],P2int[:,1]])#,P4int[:,1]])
    #     dcss2_stack = np.hstack([P0int[:,2],P2int[:,2]])#,P4int[:,2]])
    #     db4_stack = np.hstack([P0int[:,3],P2int[:,3]])#,P4int[:,4]])
    #     dPshot_stack = np.hstack([P0int[:,4],P2int[:,4]])#,P4int[:,5]])

    #     C_marg = (C
    #             + css0sig**2*np.outer(dcss0_stack,dcss0_stack)
    #             + css2sig**2*np.outer(dcss2_stack,dcss2_stack)
    #             + b4sig**2*np.outer(db4_stack, db4_stack)
    #             + Pshotsig**2*np.outer(dPshot_stack,dPshot_stack)
    #             )
        
    #     model_vector = np.hstack([P0int[:,0],P2int[:,0]])
    #     return C_marg, model_vector, Omega_m, sigma8