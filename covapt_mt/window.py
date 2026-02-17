# This file is a repackaging of the "Survey_window_kernels.ipynb" found in CovaPT
# Source, Jay Wadekar: https://github.com/JayWadekar/CovaPT

import os
import numpy as np
from numpy import conj
from scipy.interpolate import InterpolatedUnivariateSpline
from nbodykit.source.catalog import ArrayCatalog, FITSCatalog
from nbodykit.lab import cosmology, transform
import dask.array as da
import itertools as itt
import h5py
import pdb

from nbodykit import set_options
#set_options(global_cache_size=2e9)

from covapt_mt.utils import sample_from_shell, nmodes

class Survey_Geometry_Kernels():
    """
    Class that contains functions needed to calculate the covariance matrix window functions
    given a survey-like geometry 
    """
    def __init__ (self, config_dict, k_centers:list, sampling_mode="linear"):
        """Constructs Survey_Window_Kernels object

        Args:
            config_dict: A dictionary containing all the necessary information about the survey geometry and random catalogs.
            k_centers (list): A list of np arrays containing the k-bin centers for each redshift bin. The k-bin edges are inferred from the bin centers in the get_k_bin_edges function.
            sampling_mode (str): One of "linear", "log". Default "linear"
        
        Raises:
            IOError: If random catalog doesn't exist in the specified directory
        """
        self.cosmo = cosmology.Cosmology(h=config_dict["h"]).match(Omega0_m=config_dict["Om0"])

        # load in random catalog
        self.load_survey_randoms(config_dict["zbins"],
                                 config_dict["input_dir"], 
                                 config_dict["random_file_prefix"])

        # convert redshifts to physical distances based on some catalog cosmology
        self.convert_to_distances()

        # calculate bin edges and width from the k centers
        self.sampling_mode = config_dict["sampling_mode"]
        print(f"Using {self.sampling_mode} for k-bin sampling")
        self.get_k_bin_edges(k_centers, mode=self.sampling_mode)
        #self.get_k_bin_edges(k_centers)

        # As the window falls steeply with k, only low-k regions are needed for the calculation.
        # Therefore cutting out the high-k modes in the FFTs using the self.icut parameter
        self.icut=15; # needs to be less than Lm//2 (Lm: size of FFT)

        # infer the box size from random catalogs
        self.calculate_survey_properties(config_dict)

        # number of k-bins on each side of the diaganal to calculate
        # Should be kept small, since the Gaussian covariance drops quickly away from the diag
        self.delta_k_max = 3

        #if self.kfun > self.kbin_edges[1]:
        #    print("WARNING! fundamental k mode is larger than the smallest k bin! This might cause issues!")

    def load_survey_randoms(self, zbins, data_dir, random_file_prefix=""):
        """Loads random survey catalog from an hdf5 file
        
        Args:
            data_dir: location of survey random catalogs. Default the directory specified in config.py
        
        Raises:
            FileNotFoundError: If random catalog doesn't exist in the specified directory
        """
        self.num_zbins = int(len(zbins) / 2)
        self.randoms = []
        self.I22 = np.zeros(self.num_zbins)

        # NOTE: TheCov will have a more robust method for treating the window function for mulatiple tracers. 
        # The current multi-tracer code assumes you have one window function per redshift bin, which is technically incorrect. For now,
        # we will only load one random catalog per redshift bin, with the intention of investigating how much this
        # choice matters later
        for zbin in range(self.num_zbins):

                if self.num_zbins == 1:
                    random_file = random_file_prefix
                else:
                    random_file = random_file_prefix+"_"+str(zbin)

                if os.path.exists(data_dir+random_file+".h5"):
                    print("loading " + data_dir+random_file+".h5 ...")
                    randoms = self.load_h5_catalog(data_dir+random_file+".h5")
                elif os.path.exists(data_dir+random_file+".fits"):
                    print("loading " + data_dir+random_file+".fits ...")
                    randoms = FITSCatalog(data_dir+random_file+".fits")
                    randoms = self.calculate_nz(randoms)

                else:
                    raise FileNotFoundError("Could not find survey randoms (.h5 or .fits) catalog:", data_dir+random_file)
            
                bin_name = "bin"+str(zbin+1)
                subset_idx = (randoms["Z"] < zbins[bin_name+"_hi"]) & (randoms["Z"] > zbins[bin_name+"_lo"])
                self.randoms.append(randoms[subset_idx])

                print("randoms length after loading = ", len(self.randoms))
                # pdb.set_trace()
                self.I22[zbin] = np.sum(self.randoms[zbin]['NZ']**1 * self.randoms[zbin]['WEIGHT_FKP']**2)
                # self.I22[zbin] = np.sum(self.randoms[zbin]['NZ']**1 * self.randoms[zbin]['Weight']**2)
                #self.I22[zbin] = 9.047 # HACK
                print("I_22 for bin {:0.0f} = {:0.2f}".format(zbin, self.I22[zbin]))
                #print(self.I22[zbin] / I12.compute())

    def load_h5_catalog(self, file_path):
        with h5py.File(file_path, "r") as f:
            # Print all root level object names (aka keys) 
            # these can be group or dataset names 
            print("loading " + str(len(f["position_x"])) + " galaxies...")

            random = ArrayCatalog({'OriginalPosition' : np.array([f["position_x"], f["position_y"], f["position_z"]]).T,\
                                   "WEIGHT_FKP" : np.array(f["fkp_weights"])})

            random = self.calculate_nz(random)
            return random

    def get_dVolume_dz(self, z_bins):
        dummy_dict = np.empty(len(z_bins), dtype=[("ra", "f4"), ("dec", "f4"), ("z", "f4")])
        dummy_dict["ra"] = np.zeros(len(z_bins)); dummy_dict["dec"] = np.zeros(len(z_bins)); dummy_dict["z"] = z_bins
        dummy_cat = ArrayCatalog(dummy_dict)

        # this outputs positions in Mpc/h
        position = (transform.SkyToCartesian(dummy_cat["ra"], dummy_cat["dec"], dummy_cat["z"],degrees=True, cosmo=self.cosmo)).compute()
        d = np.sqrt(position[:,0]**2 + position[:,1]**2 + position[:,2]**2)
        dV = np.zeros(len(d)-1)
        
        for i in range(len(dV)):
            dV[i] = 4./3. * np.pi * (d[i+1]**3 - d[i]**3)
        return dV

    def calculate_nz(self, randoms):
        
        if "NZ" in randoms.columns: return randoms

        if "Z" not in randoms.columns:
            randoms["RA"], randoms["DEC"], randoms["Z"] = transform.CartesianToSky(randoms["OriginalPosition"], self.cosmo)
        
        print("Computing n(z) from input random catalog...")

        if "NZ" not in randoms:
            print("'NZ' not in randoms! Estimating it with RedshiftInterpolator...")
            from nbodykit.algorithms import RedshiftHistogram
            nz_hist = RedshiftHistogram(randoms, fsky=1.0, cosmo=self.cosmo, redshift="Z")
            randoms["NZ"] = nz_hist.interpolate(randoms["Z"])

        # nbar, z_edges = np.histogram(randoms["Z"].compute(), bins=100) # Gives N = n*V
        # dV_dz = self.get_dVolume_dz(z_edges)
        
        # nbar = nbar / dV_dz
        # z_centers = np.zeros(len(nbar))
        # for i in range(len(z_centers)):
        #     z_centers[i] = (z_edges[i] + z_edges[i+1]) / 2.

        # # finally, interpolate
        # nbar_func = InterpolatedUnivariateSpline(z_centers, nbar)
        # randoms["NZ"] = nbar_func(randoms["Z"])

        if "WEIGHT_FKP" not in randoms:
            print("WEIGHT FKP not found, calculating now")
            randoms["WEIGHT_FKP"] = 1. / (1 + randoms["NZ"] * 1e4)

        return randoms

    def convert_to_distances(self):
        """Converts catalog redshifts to physical distances
        
        To convert to distances this function uses an assumed "catalog" cosmology
        that is specified by the user.
        """
        for bin in range(self.num_zbins):
            if 'OriginalPosition' in self.randoms[bin].columns: continue
            self.randoms[bin]['OriginalPosition'] = transform.SkyToCartesian(
                self.randoms[bin]['RA'],
                self.randoms[bin]['DEC'],
                self.randoms[bin]['Z'], 
                degrees=True, cosmo=self.cosmo)

    def num_ffts(self, n):
        """Returns the number of FFTs to do at a given order n"""
        return int((n+1)*(n+2)/2)

    def shift_positions(self):
        """Shifts positions to be centered on the box"""
        for bin in range(len(self.randoms)):
            #self.randoms[bin]['Position'] = self.randoms[bin]['OriginalPosition']
            print("Here is is my box size = ", self.box_size[bin])
            self.randoms[bin]['Position'] = self.randoms[bin]['OriginalPosition'] + da.array(3*[self.box_size[bin]/2])

    def PowerCalc(self, arr, nBins, sort):
        """Calculates window power spectrum from FFT array"""
        window_p=np.zeros(nBins,dtype='<c8')
        for i in range(nBins):
            ind=(sort==i)
            window_p[i]=np.average(arr[ind])
        return(np.real(window_p))

    def calculate_survey_properties(self, config_dict):
        """Infers the box size and fundamental k mode from the input random catalog"""
        self.box_size = []
        self.Lm2 = []
        self.kfun = []
        for idx in range(len(self.randoms)):
            self.box_size.append(max(da.max(self.randoms[idx]["OriginalPosition"][0]),
                                     da.max(self.randoms[idx]["OriginalPosition"][1]),
                                     da.max(self.randoms[idx]["OriginalPosition"][2])) \
                               - min(da.min(self.randoms[idx]["OriginalPosition"][0]),
                                     da.min(self.randoms[idx]["OriginalPosition"][1]),
                                     da.min(self.randoms[idx]["OriginalPosition"][2])))
            if "box_size" in config_dict.keys():
                self.box_size[idx] = config_dict["box_size"]
            elif "box_padding" in config_dict.keys():
                self.box_size[idx] = self.box_size[idx].compute() * config_dict["box_padding"]
            else:
                print("WARNING! Neither box_size or padding specified! Defaulting to infered size")

            self.kfun.append(2.*np.pi/self.box_size[idx])
            # print("k func = ", self.kfun)
            # print("bin edges = ", self.kbin_edges)
            if self.sampling_mode == "linear":
                self.Lm2.append(int(self.kbin_width[idx]*self.nBins[idx]/self.kfun[idx])+1)
                print("self Lm2 = ", self.Lm2)
                print("kbin width = ", self.kbin_width[idx])
            elif self.sampling_mode == 'log':
                self.Lm2.append(int((self.kbin_edges[idx][-1])/self.kfun[idx])+1)
                print("self Lm2 when calculated = ", self.Lm2)
            elif self.sampling_mode == 'hybrid':
                # log_index = np.where()
                self.Lm2.append(39)#int(np.ceil(np.max(self.kbin_edges[idx]) / self.kfun[idx])) + 1)
                # self.Lm2.append(int(self.kbin_width[idx][0]*self.nBins[idx]/self.kfun[idx])+1)
                # self.Lm2.append(int((self.kbin_edges[idx][-1] - self.kbin_edges[idx][6])/self.kfun[idx])+1)
                print("self Lm2 when calculated = ", self.Lm2)

            
            assert self.icut < (self.Lm2[idx] / 2)
            print("min / max redshift: [{:0.2f}, {:0.2f}]".format(da.min(self.randoms[idx]["Z"]).compute(),
                                                                  da.max(self.randoms[idx]["Z"]).compute()))
            print("using box size of {:0.1f} Mpc/h, fundamental k-mode = {:0.3e} h/Mpc".format(self.box_size[idx], self.kfun[idx]))

    def calc_FFTs(self, Nmesh, names):
        """Calculates and returns Fast Fourier Transforms of the random catalog

        NOTE: This function is computationally expensive.
        
        Args:
            Nmesh: The size of the FFT mesh
        """

        export=np.zeros((len(self.randoms), 2*(1+self.num_ffts(2)+self.num_ffts(4)),Nmesh,Nmesh,Nmesh),dtype='complex128')

        for bin in range(len(self.randoms)):
            print("window " + str(bin))
            r = self.randoms[bin]['OriginalPosition'].T
            ind=0
            for w in names:
                print(f'Computing FFTs of {w}')
                print('Computing 0th order FFTs')
                Wij = np.fft.fftn(self.randoms[bin].to_mesh(Nmesh=Nmesh, BoxSize=self.box_size[bin], value=w, resampler='tsc', interlaced=True, compensated=True).paint())

                Wij *= (da.sum(self.randoms[bin][w]).compute())/np.real(Wij[0,0,0]) #Fixing normalization, e.g., zero mode should be I22 for 'W22'
                export[bin, ind]=Wij; ind+=1
                
                print('Computing 2nd order FFTs')
                for (i,i_label),(j,j_label) in itt.combinations_with_replacement(enumerate(['x', 'y', 'z']), r=2):
                    label = w + i_label + j_label
                    self.randoms[bin][label] = self.randoms[bin][w] * r[i]*r[j] /(r[0]**2 + r[1]**2 + r[2]**2)
                    Wij = np.fft.fftn(self.randoms[bin].to_mesh(Nmesh=Nmesh, BoxSize=self.box_size[bin], value=label, resampler='tsc', interlaced=True, compensated=True).paint())
                    Wij *= (da.sum(self.randoms[bin][label]).compute())/np.real(Wij[0,0,0])
                    export[bin, ind]=Wij; ind+=1

                print('Computing 4th order FFTs')
                for (i,i_label),(j,j_label),(k,k_label),(l,l_label) in itt.combinations_with_replacement(enumerate(['x', 'y', 'z']), r=4):
                    label = w + i_label + j_label + k_label + l_label
                    self.randoms[bin][label] = self.randoms[bin][w] * r[i]*r[j]*r[k]*r[l] /(r[0]**2 + r[1]**2 + r[2]**2)**2
                    Wij = np.fft.fftn(self.randoms[bin].to_mesh(Nmesh=Nmesh, BoxSize=self.box_size[bin], value=label, resampler='tsc', interlaced=True, compensated=True).paint())
                    Wij *= (da.sum(self.randoms[bin][label]).compute())/np.real(Wij[0,0,0])
                    export[bin, ind]=Wij; ind+=1

        return export

    def load_fft_file(self, data_dir, z_idx):
        """Loads and organizes information from the random catalog FFTs"""

        self.randoms = None # <- delete cashed randoms to save memory / allow multiprocessing to work
        fft_file = data_dir+'FFTWinFun.npy'
        if not os.path.exists(fft_file):
            print("ERROR! could not find fft file", fft_file)
            raise IOError

        Wij = np.load(fft_file)
        self.Lm = Wij.shape[1] #size of FFT

        self.Wij = []
        for i in range(Wij.shape[1]//2): #W22
            self.Wij.append(self.fft(Wij[z_idx][i]))

        for i in range(Wij.shape[1]//2,Wij.shape[1]): #W12, I'm taking conjugate as that is used in the 'WinFun' function later
            self.Wij.append(conj(self.fft(Wij[z_idx][i])))
        
        print("shape of window fuction =", np.array(self.Wij).shape)

    def calc_gaussian_kernels(self, Nmesh=48):
        """Calculates the gaussian kernels required for the Gaussian window function.
        
        This object only needs to be calculated once per data chunk
        
        Args:
            Nmesh: The size of the FFT mesh
            BoxSize: The survey box size in Mpc/h. Should encompass all galaxies in the survey
        """
        # Shifting the points such that the survey center is in the center of the box
        self.shift_positions()
        for idx in range(len(self.randoms)):
            print("shape of randoms = ", len(self.randoms))
            print("randoms = ", self.randoms)
            self.randoms[idx]['W12'] = self.randoms[idx]['WEIGHT_FKP']**2 
            self.randoms[idx]['W22'] = (self.randoms[idx]['WEIGHT_FKP']**2) * self.randoms[idx]['NZ']

        return self.calc_FFTs(Nmesh, ["W22", "W12"])
    
    def calc_SSC_window_function(self, Nmesh=300): #, BoxSize=7200):
        """Calculates the SSC window functions
        
        This object only needs to be calculated once per data chunk
        
        Args:
            Nmesh: The size of the FFT mesh
            BoxSize: The survey box size in Mpc/h. Should encompass all galaxies in the survey
        """

        # Fundamental k-mode
        # kfun=2.*np.pi/BoxSize
        print("box size in ssc window functions = ", self.box_size)
        kfun=2.*np.pi/self.box_size[0]
        nBins=int(Nmesh/2) # Number of bins in which power spectrum will be calculated

        # Shifting the points such that the survey center is in the center of the box
        self.shift_positions()#BoxSize)
        self.randoms[0]['W22'] = (self.randoms[0]['WEIGHT_FKP']**2) * self.randoms[0]['NZ']
        self.randoms[0]['W10'] = self.randoms[0]['W22']/self.randoms[0]['W22']

        export = self.calc_FFTs(Nmesh, ["W22", "W10"])#BoxSize, self.box_size[0]

        # For shifting the zero-frequency component to the center of the FFT array
        for i in range(len(export)):
            export[i]=np.fft.fftshift(export[i])

        # Recording the k-modes in different shells
        # Bin_kmodes contains [kx,ky,kz,radius] values of all the modes in the bin
        [kx,ky,kz] = np.zeros((3,Nmesh,Nmesh,Nmesh));

        for i in range(len(kx)):
            kx[i,:,:]+=i-Nmesh/2; ky[:,i,:]+=i-Nmesh/2; kz[:,:,i]+=i-Nmesh/2

        rk=np.sqrt(kx**2+ky**2+kz**2)
        sort=(rk).astype(int)

        rk[nBins,nBins,nBins]=1e10; kx/=rk; ky/=rk; kz/=rk; rk[nBins,nBins,nBins]=0 #rk being zero at the center causes issues so fixed that

        # Reading the FFT files for W22 (referred to as W hereafter for brevity) and W10
        [W, Wxx, Wxy, Wxz, Wyy, Wyz, Wzz, Wxxxx, Wxxxy, Wxxxz, Wxxyy, Wxxyz, Wxxzz, Wxyyy, Wxyyz, Wxyzz,\
        Wxzzz, Wyyyy, Wyyyz, Wyyzz, Wyzzz, Wzzzz, W10, W10xx, W10xy, W10xz, W10yy, W10yz, W10zz, W10xxxx,\
        W10xxxy, W10xxxz, W10xxyy, W10xxyz, W10xxzz, W10xyyy, W10xyyz, W10xyzz, W10xzzz, W10yyyy, W10yyyz,\
        W10yyzz, W10yzzz, W10zzzz] = export

        W_L0 = W
                
        W_L2=1.5*(Wxx*kx**2+Wyy*ky**2+Wzz*kz**2+2.*Wxy*kx*ky+2.*Wyz*ky*kz+2.*Wxz*kz*kx)-0.5*W
                
        W_L4=35./8.*(Wxxxx*kx**4 +Wyyyy*ky**4+Wzzzz*kz**4 \
            +4.*Wxxxy*kx**3*ky +4.*Wxxxz*kx**3*kz +4.*Wxyyy*ky**3*kx \
            +4.*Wyyyz*ky**3*kz +4.*Wxzzz*kz**3*kx +4.*Wyzzz*kz**3*ky \
            +6.*Wxxyy*kx**2*ky**2+6.*Wxxzz*kx**2*kz**2+6.*Wyyzz*ky**2*kz**2 \
            +12.*Wxxyz*kx**2*ky*kz+12.*Wxyyz*ky**2*kx*kz +12.*Wxyzz*kz**2*kx*ky) \
            -5./2.*W_L2 -7./8.*W_L0

        W10_L0 = W10
                
        W10_L2=1.5*(W10xx*kx**2+W10yy*ky**2+W10zz*kz**2+2.*W10xy*kx*ky+2.*W10yz*ky*kz+2.*W10xz*kz*kx)-0.5*W10
                
        W10_L4=35./8.*(W10xxxx*kx**4 +W10yyyy*ky**4+W10zzzz*kz**4 \
            +4.*W10xxxy*kx**3*ky +4.*W10xxxz*kx**3*kz +4.*W10xyyy*ky**3*kx \
            +4.*W10yyyz*ky**3*kz +4.*W10xzzz*kz**3*kx +4.*W10yzzz*kz**3*ky \
            +6.*W10xxyy*kx**2*ky**2+6.*W10xxzz*kx**2*kz**2+6.*W10yyzz*ky**2*kz**2 \
            +12.*W10xxyz*kx**2*ky*kz+12.*W10xyyz*ky**2*kx*kz +12.*W10xyzz*kz**2*kx*ky) \
            -5./2.*W10_L2 -7./8.*W10_L0

        P_W=np.zeros((22,nBins))
        P_W[0]=self.PowerCalc(rk, nBins, sort)*kfun # Mean |k| in the bin

        P_W[1]=self.PowerCalc(W_L0*conj(W_L0), nBins, sort) - da.sum(self.randoms[0]['NZ']**2*self.randoms[0]['WEIGHT_FKP']**4).compute() # P00 with shot noise subtracted
        P_W[2]=self.PowerCalc(W_L0*conj(W_L2), nBins, sort)*5 # P02
        P_W[3]=self.PowerCalc(W_L0*conj(W_L4), nBins, sort)*9 # P04
        P_W[4]=self.PowerCalc(W_L2*conj(W_L2), nBins, sort)*25 # P22
        P_W[5]=self.PowerCalc(W_L2*conj(W_L4), nBins, sort)*45 # P24
        P_W[6]=self.PowerCalc(W_L4*conj(W_L4), nBins, sort)*81 # P44

        P_W[7]=self.PowerCalc(W10_L0*conj(W10_L0), nBins, sort) - da.sum(self.randoms[0]['NZ']**0*self.randoms[0]['WEIGHT_FKP']**0).compute() # P00 with shot noise subtracted
        P_W[8]=self.PowerCalc(W10_L0*conj(W10_L2), nBins, sort)*5 # P02
        P_W[9]=self.PowerCalc(W10_L0*conj(W10_L4), nBins, sort)*9 # P04
        P_W[10]=self.PowerCalc(W10_L2*conj(W10_L2), nBins, sort)*25 # P22
        P_W[11]=self.PowerCalc(W10_L2*conj(W10_L4), nBins, sort)*45 # P24
        P_W[12]=self.PowerCalc(W10_L4*conj(W10_L4), nBins, sort)*81 # P44

        P_W[13]=self.PowerCalc(W_L0*conj(W10_L0), nBins, sort) - da.sum(self.randoms[0]['NZ']**1*self.randoms[0]['WEIGHT_FKP']**2).compute() # P00 with shot noise subtracted
        P_W[14]=self.PowerCalc(W_L0*conj(W10_L2), nBins, sort)*5 # P02
        P_W[15]=self.PowerCalc(W_L0*conj(W10_L4), nBins, sort)*9 # P04
        P_W[16]=self.PowerCalc(W_L2*conj(W10_L0), nBins, sort)*5 # P20
        P_W[17]=self.PowerCalc(W_L2*conj(W10_L2), nBins, sort)*25 # P22
        P_W[18]=self.PowerCalc(W_L2*conj(W10_L4), nBins, sort)*45 # P24
        P_W[19]=self.PowerCalc(W_L4*conj(W10_L0), nBins, sort)*9 # P40
        P_W[20]=self.PowerCalc(W_L4*conj(W10_L2), nBins, sort)*45 # P42
        P_W[21]=self.PowerCalc(W_L4*conj(W10_L4), nBins, sort)*81 # P44

        P_W[1:7]/=(da.sum(self.randoms[0]['W22']).compute())**2
        P_W[7:13]/=(da.sum(self.randoms[0]['W10']).compute())**2
        P_W[13:]/=(da.sum(self.randoms[0]['W10']).compute()*da.sum(self.randoms[0]['W22']).compute())

        # Minor point: setting k=0 modes by hand to avoid spurious values
        P_W[1:7,0]=[1,0,0,1,0,1]; P_W[7:13,0]=[1,0,0,1,0,1]; P_W[13:,0]=[1,0,0,0,1,0,0,0,1]
        return P_W

    def get_k_bin_edges(self, k_centers, mode="linear"):
        """calculates bin edges from an array of bin centers

        Args:
            k_centers: An np array of evenly-spaced bin centers
            mode: One of "linear", "log". Default "linear". Whether to use linear or log spacing for the k-bin edges. Note that the k-bin width is assumed to be constant, so if log spacing is used, the bin widths will not be constant.
        """

        self.kbin_width = []
        self.kbin_edges = []
        self.nBins = []

        kbin_edges_lin = []
        kbin_width_lin = []

        kbin_edges_log = []
        kbin_width_log = []

        for idx in range(len(self.randoms)):
            if mode == "linear":
                self.kbin_width.append(k_centers[idx][-1] - k_centers[idx][-2])
                kbin_half_width = self.kbin_width[idx] / 2.
                self.kbin_edges.append(np.zeros(len(k_centers[idx])+1))
                self.kbin_edges[idx][0] = k_centers[idx][0] - kbin_half_width

                assert self.kbin_edges[idx][0] > 0.
                for i in range(1, len(self.kbin_edges[idx])):
                    self.kbin_edges[idx][i] = k_centers[idx][i-1] + kbin_half_width

            elif mode == "log":

                self.kbin_width.append(k_centers[idx][1] / k_centers[idx][0])
                self.kbin_edges.append(np.zeros(len(k_centers[idx])+1))
                self.kbin_edges[idx][0] = 10 ** (np.log10(k_centers[idx][0]) - (np.log10(self.kbin_width[idx]) / 2))

                assert self.kbin_edges[idx][0] > 0.
                for i in range(1, len(self.kbin_edges[idx])):
                    self.kbin_edges[idx][i] = self.kbin_edges[idx][i-1] * (self.kbin_width[idx]) 

            elif mode == 'hybrid':
                # pass
                lin_idx = np.where(k_centers[idx]<0.009)

                k_centers_lin = k_centers[idx][lin_idx]

                print("k centers lin = ", k_centers_lin)

                kbin_width_lin.append(k_centers_lin[1] - k_centers_lin[0])
                kbin_half_width = kbin_width_lin[idx] / 2.
                kbin_edges_lin.append(np.zeros(len(k_centers_lin)+1))
                kbin_edges_lin[idx][0] = k_centers_lin[0] - kbin_half_width

                assert kbin_edges_lin[idx][0] > 0.
                for i in range(1, len(kbin_edges_lin[idx])):
                    kbin_edges_lin[idx][i] = k_centers_lin[i-1] + kbin_half_width
                print("kbin edges = ", kbin_edges_lin)
                

                log_idx = np.where(k_centers[idx]>0.009)

                k_centers_log = k_centers[idx][log_idx]

                print("k centers log = ", k_centers_log)

                kbin_width_log.append(k_centers_log[1] / k_centers_log[0])
                kbin_edges_log.append(np.zeros(len(k_centers[idx][log_idx])+1))
                # kbin_edges_log[idx][0] = 10 ** (np.log10(k_centers_log[idx][0]) - (np.log10(kbin_width_log) / 2))
                # kbin_edges_log[idx][0] = 10 ** (np.log10(k_centers_log[0]) - (np.log10(kbin_width_log[idx]) / 2))
                kbin_edges_log[idx][0] = kbin_edges_lin[idx][-1]
                # self.kbin_edges[idx][0] = 10 ** (np.log10(k_centers[idx][0]) - (np.log10(self.kbin_width[idx]) / 2))

                # assert kbin_edges_log[0] > 0.
                # for i in range(1, len(kbin_edges_log)):
                #     kbin_edges_log[i] = kbin_edges_log[i-1] * (kbin_width_log)

                assert kbin_edges_log[idx][0] > 0.
                for i in range(1, len(kbin_edges_log[idx])):
                    kbin_edges_log[idx][i] = kbin_edges_log[idx][i-1] * (kbin_width_log[idx])

                print("k edges log = ", kbin_edges_log)
                print('    ')
                print('   ')
                # kbin_edges_total = np.concatenate([kbin_edges_lin,kbin_edges_log])
                # self.kbin_edges.append(kbin_edges_total)

                kbin_edges_total = np.array([*kbin_edges_lin[idx],*kbin_edges_log[idx][1:]])
                self.kbin_edges.append(kbin_edges_total)

                print(len(self.kbin_edges[idx]))

                # kbin_width_total = np.concatenate([kbin_width_lin, kbin_width_log])
                # self.kbin_width.append(kbin_width_total)

                self.kbin_width.append([kbin_width_lin[idx],kbin_width_log[idx]])
                print(len(self.kbin_width[idx]))



            self.nBins.append(len(k_centers[idx]))
            print(f"k bin edges for bin {idx} has : {self.nBins[idx]} bins: {self.kbin_edges[idx]}")
            print(f"Type of kbin_edges: {type(self.kbin_edges[0])}")
            print(f"Type of kbin_width: {type(self.kbin_width[0])}")

    def fft(self, temp):
        """Does some shifting of the fft arrays"""

        ia=self.Lm//2-1; ib=self.Lm//2+1
        temp2=np.zeros((self.Lm,self.Lm,self.Lm),dtype='<c8')
        temp2[ia:self.Lm,ia:self.Lm,ia:self.Lm]=temp[0:ib,0:ib,0:ib]; temp2[0:ia,ia:self.Lm,ia:self.Lm]=temp[ib:self.Lm,0:ib,0:ib]
        temp2[ia:self.Lm,0:ia,ia:self.Lm]=temp[0:ib,ib:self.Lm,0:ib]; temp2[ia:self.Lm,ia:self.Lm,0:ia]=temp[0:ib,0:ib,ib:self.Lm]
        temp2[0:ia,0:ia,ia:self.Lm]=temp[ib:self.Lm,ib:self.Lm,0:ib]; temp2[0:ia,ia:self.Lm,0:ia]=temp[ib:self.Lm,0:ib,ib:self.Lm]
        temp2[ia:self.Lm,0:ia,0:ia]=temp[0:ib,ib:self.Lm,ib:self.Lm]; temp2[0:ia,0:ia,0:ia]=temp[ib:self.Lm,ib:self.Lm,ib:self.Lm]
    
        return(temp2[ia-self.icut:ia+self.icut+1,ia-self.icut:ia+self.icut+1,ia-self.icut:ia+self.icut+1])

    def get_shell_modes(self, bin_idx:int):
        """Calculates the specific kmodes present in a given k-bin based on the given survey propereies
        
        Raises:
            AssertionError: If one of the given k-bins has 0 k modes. This can happen if your box size is too small relative to your k-bin width
        """

        Lm2 = self.Lm2[bin_idx]
        print("Lm2 in shell modes =  ", Lm2)
        [ix,iy,iz] = np.zeros((3,2*Lm2+1,2*Lm2+1,2*Lm2+1))
        Bin_kmodes=[]
        Bin_ModeNum=np.zeros(self.nBins[bin_idx],dtype=int)

        for i in range(self.nBins[bin_idx]): Bin_kmodes.append([])
        for i in range(len(ix)):
            ix[i,:,:]+=i-Lm2
            iy[:,i,:]+=i-Lm2
            iz[:,:,i]+=i-Lm2

        rk=np.sqrt(ix**2+iy**2+iz**2)
        if self.sampling_mode == "linear":
            # sort=((rk*self.kfun[bin_idx]-self.kbin_edges[bin_idx][0])/self.kbin_width[bin_idx]).astype(int)
            sort = np.ones_like(rk) * -1
            for kbin in range(self.nBins[bin_idx]):
                idx = np.where((rk*self.kfun[bin_idx] >= self.kbin_edges[bin_idx][kbin]) & 
                               (rk*self.kfun[bin_idx] < self.kbin_edges[bin_idx][kbin+1]))
                sort[idx] = kbin

            sort = sort.astype(int)

        # NOTE: This code might not be the correct way to do this...
        elif self.sampling_mode == "log":
            sort = np.ones_like(rk) * -1
            for kbin in range(self.nBins[bin_idx]):
                idx = np.where((rk*self.kfun[bin_idx] >= self.kbin_edges[bin_idx][kbin]) & 
                               (rk*self.kfun[bin_idx] < self.kbin_edges[bin_idx][kbin+1]))
                sort[idx] = kbin

            sort = sort.astype(int)
        
        elif self.sampling_mode == 'hybrid':
            sort = np.ones_like(rk) * -1
            for kbin in range(self.nBins[bin_idx]):
                idx = np.where((rk*self.kfun[bin_idx] >= self.kbin_edges[bin_idx][kbin]) & 
                            (rk*self.kfun[bin_idx] < self.kbin_edges[bin_idx][kbin+1]))
                sort[idx] = kbin
            
            sort = sort.astype(int)

        for i in range(self.nBins[bin_idx]):
            ind=(sort==i)
            # print("ind in loop after sample = ",len(ix[ind]))
            Bin_ModeNum[i]=len(ix[ind])
            # print("Bin mode num = ", Bin_ModeNum)
            Bin_kmodes[i]=np.hstack((ix[ind].reshape(-1,1),iy[ind].reshape(-1,1),iz[ind].reshape(-1,1),rk[ind].reshape(-1,1)))
        
        assert np.all(Bin_ModeNum != 0), "ERROR! some bins have 0 k modes! Your box-size or kbin-width is probably too small"
        print(f"Sampling mode: {self.sampling_mode}")
        print(f"Nmodes[0:6]: {Bin_ModeNum[0:min(6, len(Bin_ModeNum))]}")
        return Bin_kmodes, Bin_ModeNum
    
    def ell_factor(self, l1, l2):
        """window function prefactors"""
        return (2*l1 + 1) * (2*l2 + 1) * (2 if 0 in (l1, l2) else 1)

    def calc_gaussian_window_function(self, kbin_idx : int, zbin_idx, kmodes_sampled : int =400):
        """Returns the window function of a specific k-bin for l=0,2,and 4 auto + cross covariance
        
        NOTE: This function is computationally expensive and should be run in parralel
        
        Args:
            kbin_idx: the specific k-bin index to calculate the window function for
            zbin_idx: the specific redshift bin to calculate the window function for
            kmodes_sampled: The number of random samples to use
        """
        [W, Wxx, Wxy, Wxz, Wyy, Wyz, Wzz, Wxxxx, Wxxxy, Wxxxz, Wxxyy, Wxxyz, Wxxzz, Wxyyy, Wxyyz, Wxyzz,\
        Wxzzz, Wyyyy, Wyyyz, Wyyzz, Wyzzz, Wzzzz, W12, W12xx, W12xy, W12xz, W12yy, W12yz, W12zz, W12xxxx,\
        W12xxxy, W12xxxz, W12xxyy, W12xxyz, W12xxzz, W12xyyy, W12xyyz, W12xyzz, W12xzzz, W12yyyy, W12yyyz,\
        W12yyzz, W12yzzz, W12zzzz] = self.Wij

        print("kbin sampled in gauss window func = ", kbin_idx)
        avgWij=np.zeros((2*self.delta_k_max+1,15,6))
        [ix,iy,iz,k2xh,k2yh,k2zh]=np.zeros((6,2*self.icut+1,2*self.icut+1,2*self.icut+1))

        for i in range(2*self.icut+1): 
            ix[i,:,:]+=i-self.icut; iy[:,i,:]+=i-self.icut; iz[:,:,i]+=i-self.icut
            
        # randomly select kmodes_sampled number of k-modes

        kmodes, Nmodes = self.get_shell_modes(zbin_idx)
        #kmodes = np.array([[sample_from_shell(kmin/self.kfun[bin_idx], kmax/self.kfun[bin_idx]) for _ in range(
        #                    kmodes_sampled)] for kmin, kmax in zip(self.kbin_edges[bin_idx][:-1], self.kbin_edges[bin_idx][1:])])
        #Nmodes = nmodes(self.box_size[bin_idx]**3, self.kbin_edges[bin_idx][:-1], self.kbin_edges[bin_idx][1:])
        np.random.seed(42)
        if (kmodes_sampled<Nmodes[kbin_idx]):
           norm = kmodes_sampled
           sampled=(np.random.rand(kmodes_sampled)*Nmodes[kbin_idx]).astype(int)
        else:
           norm = Nmodes[kbin_idx]
           sampled=np.arange(Nmodes[kbin_idx],dtype=int)
        # Loop thru randomly-selected k-modes
        #for mode in sampled:

        print(f"kbin {kbin_idx}: norm={norm}, I22={self.I22[zbin_idx]:.6f}")
        print(f"Nmodes for this bin: {Nmodes[kbin_idx]}")

        for n in sampled:
            [ik1x,ik1y,ik1z,rk1]=kmodes[kbin_idx][n]

        #for mode in range(kmodes_sampled):
            #[ik1x,ik1y,ik1z,rk1] = kmodes[kbin_idx, mode]
            #[ik1x,ik1y,ik1z,rk1] = kmodes[kbin_idx, mode, :]
            if (rk1<=1e-10): 
                k1xh=0
                k1yh=0
                k1zh=0
            else:
                k1xh=ik1x/rk1
                k1yh=ik1y/rk1
                k1zh=ik1z/rk1
                
            # Build a 3D array of modes around the selected mode   
            k2xh=ik1x-ix
            k2yh=ik1y-iy
            k2zh=ik1z-iz

            rk2=np.sqrt(k2xh**2+k2yh**2+k2zh**2)
            if self.sampling_mode == "linear":
                # sort=((rk2*self.kfun - self.kbin_edges[zbin_idx][0])/self.kbin_width).astype(int)-kbin_idx # to decide later which shell the k2 mode belongs to
                sort = np.ones_like(rk2) * -1
                for kbin in range(self.nBins[zbin_idx]):
                    idx = np.where((rk2*self.kfun[zbin_idx] >= self.kbin_edges[zbin_idx][kbin]) & 
                                (rk2*self.kfun[zbin_idx] < self.kbin_edges[zbin_idx][kbin+1]))
                    sort[idx] = kbin - kbin_idx
                sort = sort.astype(int)
            
            elif self.sampling_mode == "log":
                sort = np.ones_like(rk2) * -1
                for kbin in range(self.nBins[zbin_idx]):
                    idx = np.where((rk2*self.kfun[zbin_idx] >= self.kbin_edges[zbin_idx][kbin]) & 
                                   (rk2*self.kfun[zbin_idx] < self.kbin_edges[zbin_idx][kbin+1]))
                    sort[idx] = kbin - kbin_idx
                sort = sort.astype(int)
            
            elif self.sampling_mode == 'hybrid':
                sort = np.ones_like(rk2) * -1
                # print(len(sort))
                # print(sort.shape)
                for kbin in range(self.nBins[zbin_idx]):
                    idx = np.where((rk2*self.kfun[zbin_idx] >= self.kbin_edges[zbin_idx][kbin]) & 
                                (rk2*self.kfun[zbin_idx] < self.kbin_edges[zbin_idx][kbin+1]))
                    sort[idx] = kbin - kbin_idx
                sort = sort.astype(int)

            ind=(rk2==0)
            if (ind.any()>0): rk2[ind]=1e10
            k2xh/=rk2; k2yh/=rk2; k2zh/=rk2
            
            # Now calculating window multipole kernels by taking dot products of cartesian FFTs with k1-hat, k2-hat arrays
            # W corresponds to W22(k) and Wc corresponds to conjugate of W22(k)
            # L(i) refers to multipoles
            W_L0 = W
            Wc_L0 = conj(W)
            
            xx=Wxx*k1xh**2+Wyy*k1yh**2+Wzz*k1zh**2+2.*Wxy*k1xh*k1yh+2.*Wyz*k1yh*k1zh+2.*Wxz*k1zh*k1xh
            
            W_k1L2=1.5*xx-0.5*W
            W_k2L2=1.5*(Wxx*k2xh**2+Wyy*k2yh**2+Wzz*k2zh**2 \
                +2.*Wxy*k2xh*k2yh+2.*Wyz*k2yh*k2zh+2.*Wxz*k2zh*k2xh)-0.5*W
            Wc_k1L2=conj(W_k1L2)
            Wc_k2L2=conj(W_k2L2)
            
            W_k1L4=35./8.*(Wxxxx*k1xh**4 +Wyyyy*k1yh**4+Wzzzz*k1zh**4 \
                +4.*Wxxxy*k1xh**3*k1yh +4.*Wxxxz*k1xh**3*k1zh +4.*Wxyyy*k1yh**3*k1xh \
                +4.*Wyyyz*k1yh**3*k1zh +4.*Wxzzz*k1zh**3*k1xh +4.*Wyzzz*k1zh**3*k1yh \
                +6.*Wxxyy*k1xh**2*k1yh**2+6.*Wxxzz*k1xh**2*k1zh**2+6.*Wyyzz*k1yh**2*k1zh**2 \
                +12.*Wxxyz*k1xh**2*k1yh*k1zh+12.*Wxyyz*k1yh**2*k1xh*k1zh +12.*Wxyzz*k1zh**2*k1xh*k1yh) \
                -5./2.*W_k1L2 -7./8.*W_L0
            Wc_k1L4=conj(W_k1L4)
            
            k1k2=Wxxxx*(k1xh*k2xh)**2+Wyyyy*(k1yh*k2yh)**2+Wzzzz*(k1zh*k2zh)**2 \
                +Wxxxy*(k1xh*k1yh*k2xh**2+k1xh**2*k2xh*k2yh)*2\
                +Wxxxz*(k1xh*k1zh*k2xh**2+k1xh**2*k2xh*k2zh)*2\
                +Wyyyz*(k1yh*k1zh*k2yh**2+k1yh**2*k2yh*k2zh)*2\
                +Wyzzz*(k1zh*k1yh*k2zh**2+k1zh**2*k2zh*k2yh)*2\
                +Wxyyy*(k1yh*k1xh*k2yh**2+k1yh**2*k2yh*k2xh)*2\
                +Wxzzz*(k1zh*k1xh*k2zh**2+k1zh**2*k2zh*k2xh)*2\
                +Wxxyy*(k1xh**2*k2yh**2+k1yh**2*k2xh**2+4.*k1xh*k1yh*k2xh*k2yh)\
                +Wxxzz*(k1xh**2*k2zh**2+k1zh**2*k2xh**2+4.*k1xh*k1zh*k2xh*k2zh)\
                +Wyyzz*(k1yh**2*k2zh**2+k1zh**2*k2yh**2+4.*k1yh*k1zh*k2yh*k2zh)\
                +Wxyyz*(k1xh*k1zh*k2yh**2+k1yh**2*k2xh*k2zh+2.*k1yh*k2yh*(k1zh*k2xh+k1xh*k2zh))*2\
                +Wxxyz*(k1yh*k1zh*k2xh**2+k1xh**2*k2yh*k2zh+2.*k1xh*k2xh*(k1zh*k2yh+k1yh*k2zh))*2\
                +Wxyzz*(k1yh*k1xh*k2zh**2+k1zh**2*k2yh*k2xh+2.*k1zh*k2zh*(k1xh*k2yh+k1yh*k2xh))*2
            
            W_k2L4=35./8.*(Wxxxx*k2xh**4 +Wyyyy*k2yh**4+Wzzzz*k2zh**4 \
                +4.*Wxxxy*k2xh**3*k2yh +4.*Wxxxz*k2xh**3*k2zh +4.*Wxyyy*k2yh**3*k2xh \
                +4.*Wyyyz*k2yh**3*k2zh +4.*Wxzzz*k2zh**3*k2xh +4.*Wyzzz*k2zh**3*k2yh \
                +6.*Wxxyy*k2xh**2*k2yh**2+6.*Wxxzz*k2xh**2*k2zh**2+6.*Wyyzz*k2yh**2*k2zh**2 \
                +12.*Wxxyz*k2xh**2*k2yh*k2zh+12.*Wxyyz*k2yh**2*k2xh*k2zh +12.*Wxyzz*k2zh**2*k2xh*k2yh) \
                -5./2.*W_k2L2 -7./8.*W_L0
            Wc_k2L4=conj(W_k2L4)
            
            W_k1L2_k2L2= 9./4.*k1k2 -3./4.*xx -1./2.*W_k2L2
            W_k1L2_k2L4=2/7.*W_k1L2+20/77.*W_k1L4 #approximate as 6th order FFTs not simulated
            W_k1L4_k2L2=W_k1L2_k2L4 #approximate
            W_k1L4_k2L4=1/9.*W_L0+100/693.*W_k1L2+162/1001.*W_k1L4
            Wc_k1L2_k2L2= conj(W_k1L2_k2L2)
            Wc_k1L2_k2L4=conj(W_k1L2_k2L4); Wc_k1L4_k2L2=Wc_k1L2_k2L4
            Wc_k1L4_k2L4=conj(W_k1L4_k2L4)
            
            k1k2W12=W12xxxx*(k1xh*k2xh)**2+W12yyyy*(k1yh*k2yh)**2+W12zzzz*(k1zh*k2zh)**2 \
                +W12xxxy*(k1xh*k1yh*k2xh**2+k1xh**2*k2xh*k2yh)*2\
                +W12xxxz*(k1xh*k1zh*k2xh**2+k1xh**2*k2xh*k2zh)*2\
                +W12yyyz*(k1yh*k1zh*k2yh**2+k1yh**2*k2yh*k2zh)*2\
                +W12yzzz*(k1zh*k1yh*k2zh**2+k1zh**2*k2zh*k2yh)*2\
                +W12xyyy*(k1yh*k1xh*k2yh**2+k1yh**2*k2yh*k2xh)*2\
                +W12xzzz*(k1zh*k1xh*k2zh**2+k1zh**2*k2zh*k2xh)*2\
                +W12xxyy*(k1xh**2*k2yh**2+k1yh**2*k2xh**2+4.*k1xh*k1yh*k2xh*k2yh)\
                +W12xxzz*(k1xh**2*k2zh**2+k1zh**2*k2xh**2+4.*k1xh*k1zh*k2xh*k2zh)\
                +W12yyzz*(k1yh**2*k2zh**2+k1zh**2*k2yh**2+4.*k1yh*k1zh*k2yh*k2zh)\
                +W12xyyz*(k1xh*k1zh*k2yh**2+k1yh**2*k2xh*k2zh+2.*k1yh*k2yh*(k1zh*k2xh+k1xh*k2zh))*2\
                +W12xxyz*(k1yh*k1zh*k2xh**2+k1xh**2*k2yh*k2zh+2.*k1xh*k2xh*(k1zh*k2yh+k1yh*k2zh))*2\
                +W12xyzz*(k1yh*k1xh*k2zh**2+k1zh**2*k2yh*k2xh+2.*k1zh*k2zh*(k1xh*k2yh+k1yh*k2xh))*2
            
            xxW12=W12xx*k1xh**2+W12yy*k1yh**2+W12zz*k1zh**2+2.*W12xy*k1xh*k1yh+2.*W12yz*k1yh*k1zh+2.*W12xz*k1zh*k1xh
        
            W12_L0 = W12
            W12_k1L2=1.5*xxW12-0.5*W12
            W12_k1L4=35./8.*(W12xxxx*k1xh**4 +W12yyyy*k1yh**4+W12zzzz*k1zh**4 \
                +4.*W12xxxy*k1xh**3*k1yh +4.*W12xxxz*k1xh**3*k1zh +4.*W12xyyy*k1yh**3*k1xh \
                +6.*W12xxyy*k1xh**2*k1yh**2+6.*W12xxzz*k1xh**2*k1zh**2+6.*W12yyzz*k1yh**2*k1zh**2 \
                +12.*W12xxyz*k1xh**2*k1yh*k1zh+12.*W12xyyz*k1yh**2*k1xh*k1zh +12.*W12xyzz*k1zh**2*k1xh*k1yh) \
                -5./2.*W12_k1L2 -7./8.*W12_L0
            W12_k1L4_k2L2=2/7.*W12_k1L2+20/77.*W12_k1L4
            W12_k1L4_k2L4=1/9.*W12_L0+100/693.*W12_k1L2+162/1001.*W12_k1L4
            W12_k2L2=1.5*(W12xx*k2xh**2+W12yy*k2yh**2+W12zz*k2zh**2\
                +2.*W12xy*k2xh*k2yh+2.*W12yz*k2yh*k2zh+2.*W12xz*k2zh*k2xh)-0.5*W12
            W12_k2L4=35./8.*(W12xxxx*k2xh**4 +W12yyyy*k2yh**4+W12zzzz*k2zh**4 \
                +4.*W12xxxy*k2xh**3*k2yh +4.*W12xxxz*k2xh**3*k2zh +4.*W12xyyy*k2yh**3*k2xh \
                +4.*W12yyyz*k2yh**3*k2zh +4.*W12xzzz*k2zh**3*k2xh +4.*W12yzzz*k2zh**3*k2yh \
                +6.*W12xxyy*k2xh**2*k2yh**2+6.*W12xxzz*k2xh**2*k2zh**2+6.*W12yyzz*k2yh**2*k2zh**2 \
                +12.*W12xxyz*k2xh**2*k2yh*k2zh+12.*W12xyyz*k2yh**2*k2xh*k2zh +12.*W12xyzz*k2zh**2*k2xh*k2yh) \
                -5./2.*W12_k2L2 -7./8.*W12_L0
            
            W12_k1L2_k2L2= 9./4.*k1k2W12 -3./4.*xxW12 -1./2.*W12_k2L2
            
            W_k1L2_Sumk2L22=1/5.*W_k1L2+2/7.*W_k1L2_k2L2+18/35.*W_k1L2_k2L4
            W_k1L2_Sumk2L24=2/7.*W_k1L2_k2L2+20/77.*W_k1L2_k2L4
            W_k1L4_Sumk2L22=1/5.*W_k1L4+2/7.*W_k1L4_k2L2+18/35.*W_k1L4_k2L4
            W_k1L4_Sumk2L24=2/7.*W_k1L4_k2L2+20/77.*W_k1L4_k2L4
            W_k1L4_Sumk2L44=1/9.*W_k1L4+100/693.*W_k1L4_k2L2+162/1001.*W_k1L4_k2L4
            
            C00exp = [Wc_L0*W_L0,Wc_L0*W_k2L2,Wc_L0*W_k2L4,\
                    Wc_k1L2*W_L0,Wc_k1L2*W_k2L2,Wc_k1L2*W_k2L4,\
                    Wc_k1L4*W_L0,Wc_k1L4*W_k2L2,Wc_k1L4*W_k2L4]
            
            C00exp += [2.*W_L0*W12_L0,W_k1L2*W12_L0,W_k1L4*W12_L0,\
                    W_k2L2*W12_L0,W_k2L4*W12_L0,conj(W12_L0)*W12_L0]
            
            C22exp = [Wc_k2L2*W_k1L2 + Wc_L0*W_k1L2_k2L2,\
                    Wc_k2L2*W_k1L2_k2L2 + Wc_L0*W_k1L2_Sumk2L22,\
                    Wc_k2L2*W_k1L2_k2L4 + Wc_L0*W_k1L2_Sumk2L24,\
                    Wc_k1L2_k2L2*W_k1L2 + Wc_k1L2*W_k1L2_k2L2,\
                    Wc_k1L2_k2L2*W_k1L2_k2L2 + Wc_k1L2*W_k1L2_Sumk2L22,\
                    Wc_k1L2_k2L2*W_k1L2_k2L4 + Wc_k1L2*W_k1L2_Sumk2L24,\
                    Wc_k1L4_k2L2*W_k1L2 + Wc_k1L4*W_k1L2_k2L2,\
                    Wc_k1L4_k2L2*W_k1L2_k2L2 + Wc_k1L4*W_k1L2_Sumk2L22,\
                    Wc_k1L4_k2L2*W_k1L2_k2L4 + Wc_k1L4*W_k1L2_Sumk2L24]
            
            C22exp += [W_k1L2*W12_k2L2 + W_k2L2*W12_k1L2\
                    +W_k1L2_k2L2*W12_L0+W_L0*W12_k1L2_k2L2,\
                    0.5*((1/5.*W_L0+2/7.*W_k1L2+18/35.*W_k1L4)*W12_k2L2 + W_k1L2_k2L2*W12_k1L2\
                    +(1/5.*W_k2L2+2/7.*W_k1L2_k2L2+18/35.*W_k1L4_k2L2)*W12_L0 + W_k1L2*W12_k1L2_k2L2),\
                        0.5*((2/7.*W_k1L2+20/77.*W_k1L4)*W12_k2L2 + W_k1L4_k2L2*W12_k1L2\
                    +(2/7.*W_k1L2_k2L2+20/77.*W_k1L4_k2L2)*W12_L0 + W_k1L4*W12_k1L2_k2L2),\
                    0.5*(W_k1L2_k2L2*W12_k2L2+(1/5.*W_L0+2/7.*W_k2L2+18/35.*W_k2L4)*W12_k1L2\
                    +(1/5.*W_k1L2+2/7.*W_k1L2_k2L2+18/35.*W_k1L2_k2L4)*W12_L0 + W_k2L2*W12_k1L2_k2L2),\
                    0.5*(W_k1L2_k2L4*W12_k2L2+(2/7.*W_k2L2+20/77.*W_k2L4)*W12_k1L2\
                    +(2/7.*W_k1L2_k2L2+20/77.*W_k1L2_k2L4)*W12_L0 + W_k2L4*W12_k1L2_k2L2),\
                    conj(W12_k1L2_k2L2)*W12_L0+conj(W12_k1L2)*W12_k2L2]
            
            C44exp = [Wc_k2L4*W_k1L4 + Wc_L0*W_k1L4_k2L4,\
                    Wc_k2L4*W_k1L4_k2L2 + Wc_L0*W_k1L4_Sumk2L24,\
                    Wc_k2L4*W_k1L4_k2L4 + Wc_L0*W_k1L4_Sumk2L44,\
                    Wc_k1L2_k2L4*W_k1L4 + Wc_k1L2*W_k1L4_k2L4,\
                    Wc_k1L2_k2L4*W_k1L4_k2L2 + Wc_k1L2*W_k1L4_Sumk2L24,\
                    Wc_k1L2_k2L4*W_k1L4_k2L4 + Wc_k1L2*W_k1L4_Sumk2L44,\
                    Wc_k1L4_k2L4*W_k1L4 + Wc_k1L4*W_k1L4_k2L4,\
                    Wc_k1L4_k2L4*W_k1L4_k2L2 + Wc_k1L4*W_k1L4_Sumk2L24,\
                    Wc_k1L4_k2L4*W_k1L4_k2L4 + Wc_k1L4*W_k1L4_Sumk2L44]
            
            C44exp += [W_k1L4*W12_k2L4 + W_k2L4*W12_k1L4\
                    +W_k1L4_k2L4*W12_L0+W_L0*W12_k1L4_k2L4,\
                    0.5*((2/7.*W_k1L2+20/77.*W_k1L4)*W12_k2L4 + W_k1L2_k2L4*W12_k1L4\
                    +(2/7.*W_k1L2_k2L4+20/77.*W_k1L4_k2L4)*W12_L0 + W_k1L2*W12_k1L4_k2L4),\
                    0.5*((1/9.*W_L0+100/693.*W_k1L2+162/1001.*W_k1L4)*W12_k2L4 + W_k1L4_k2L4*W12_k1L4\
                    +(1/9.*W_k2L4+100/693.*W_k1L2_k2L4+162/1001.*W_k1L4_k2L4)*W12_L0 + W_k1L4*W12_k1L4_k2L4),\
                    0.5*(W_k1L4_k2L2*W12_k2L4+(2/7.*W_k2L2+20/77.*W_k2L4)*W12_k1L4\
                    +(2/7.*W_k1L4_k2L2+20/77.*W_k1L4_k2L4)*W12_L0 + W_k2L2*W12_k1L4_k2L4),\
                    0.5*(W_k1L4_k2L4*W12_k2L4+(1/9.*W_L0+100/693.*W_k2L2+162/1001.*W_k2L4)*W12_k1L4\
                    +(1/9.*W_k1L4+100/693.*W_k1L4_k2L2+162/1001.*W_k1L4_k2L4)*W12_L0 + W_k2L4*W12_k1L4_k2L4),\
                    conj(W12_k1L4_k2L4)*W12_L0+conj(W12_k1L4)*W12_k2L4] #1/(nbar)^2
            
            C20exp = [Wc_L0*W_k1L2,Wc_L0*W_k1L2_k2L2,Wc_L0*W_k1L2_k2L4,\
                    Wc_k1L2*W_k1L2,Wc_k1L2*W_k1L2_k2L2,Wc_k1L2*W_k1L2_k2L4,\
                    Wc_k1L4*W_k1L2,Wc_k1L4*W_k1L2_k2L2,Wc_k1L4*W_k1L2_k2L4]
            
            C20exp += [W_k1L2*W12_L0 + W*W12_k1L2,\
                    0.5*((1/5.*W+2/7.*W_k1L2+18/35.*W_k1L4)*W12_L0 + W_k1L2*W12_k1L2),\
                    0.5*((2/7.*W_k1L2+20/77.*W_k1L4)*W12_L0 + W_k1L4*W12_k1L2),\
                    0.5*(W_k1L2_k2L2*W12_L0 + W_k2L2*W12_k1L2),\
                    0.5*(W_k1L2_k2L4*W12_L0 + W_k2L4*W12_k1L2),\
                    np.conj(W12_k1L2)*W12_L0]
            
            C40exp = [Wc_L0*W_k1L4,Wc_L0*W_k1L4_k2L2,Wc_L0*W_k1L4_k2L4,\
                    Wc_k1L2*W_k1L4,Wc_k1L2*W_k1L4_k2L2,Wc_k1L2*W_k1L4_k2L4,\
                    Wc_k1L4*W_k1L4,Wc_k1L4*W_k1L4_k2L2,Wc_k1L4*W_k1L4_k2L4]
            
            C40exp += [W_k1L4*W12_L0 + W*W12_k1L4,\
                    0.5*((2/7.*W_k1L2+20/77.*W_k1L4)*W12_L0 + W_k1L2*W12_k1L4),\
                    0.5*((1/9.*W+100/693.*W_k1L2+162/1001.*W_k1L4)*W12_L0 + W_k1L4*W12_k1L4),\
                    0.5*(W_k1L4_k2L2*W12_L0 + W_k2L2*W12_k1L4),\
                    0.5*(W_k1L4_k2L4*W12_L0 + W_k2L4*W12_k1L4),\
                    np.conj(W12_k1L4)*W12_L0]
            
            C42exp = [Wc_k2L2*W_k1L4 + Wc_L0*W_k1L4_k2L2,\
                    Wc_k2L2*W_k1L4_k2L2 + Wc_L0*W_k1L4_Sumk2L22,\
                    Wc_k2L2*W_k1L4_k2L4 + Wc_L0*W_k1L4_Sumk2L24,\
                    Wc_k1L2_k2L2*W_k1L4 + Wc_k1L2*W_k1L4_k2L2,\
                    Wc_k1L2_k2L2*W_k1L4_k2L2 + Wc_k1L2*W_k1L4_Sumk2L22,\
                    Wc_k1L2_k2L2*W_k1L4_k2L4 + Wc_k1L2*W_k1L4_Sumk2L24,\
                    Wc_k1L4_k2L2*W_k1L4 + Wc_k1L4*W_k1L4_k2L2,\
                    Wc_k1L4_k2L2*W_k1L4_k2L2 + Wc_k1L4*W_k1L4_Sumk2L22,\
                    Wc_k1L4_k2L2*W_k1L4_k2L4 + Wc_k1L4*W_k1L4_Sumk2L24]
            
            C42exp += [W_k1L4*W12_k2L2 + W_k2L2*W12_k1L4+\
                    W_k1L4_k2L2*W12_L0+W*W12_k1L4_k2L2,\
                    0.5*((2/7.*W_k1L2+20/77.*W_k1L4)*W12_k2L2 + W_k1L2_k2L2*W12_k1L4\
                    +(2/7.*W_k1L2_k2L2+20/77.*W_k1L4_k2L2)*W12_L0 + W_k1L2*W12_k1L4_k2L2),\
                    0.5*((1/9.*W+100/693.*W_k1L2+162/1001.*W_k1L4)*W12_k2L2 + W_k1L4_k2L2*W12_k1L4\
                    +(1/9.*W_k2L2+100/693.*W_k1L2_k2L2+162/1001.*W_k1L4_k2L2)*W12_L0 + W_k1L4*W12_k1L4_k2L2),\
                    0.5*(W_k1L4_k2L2*W12_k2L2+(1/5.*W+2/7.*W_k2L2+18/35.*W_k2L4)*W12_k1L4\
                    +(1/5.*W_k1L4+2/7.*W_k1L4_k2L2+18/35.*W_k1L4_k2L4)*W12_L0 + W_k2L2*W12_k1L4_k2L2),\
                    0.5*(W_k1L4_k2L4*W12_k2L2+(2/7.*W_k2L2+20/77.*W_k2L4)*W12_k1L4\
                    +(2/7.*W_k1L4_k2L2+20/77.*W_k1L4_k2L4)*W12_L0 + W_k2L4*W12_k1L4_k2L2),\
                    conj(W12_k1L4_k2L2)*W12_L0+conj(W12_k1L4)*W12_k2L2] #1/(nbar)^2
            
            for delta_k in range(-self.delta_k_max,self.delta_k_max+1):
                ind=(sort==delta_k)
                # Iterating over terms (m,m') that will multiply P_m(k1)*P_m'(k2) in the sum
                for term in range(15):
                    avgWij[delta_k+self.delta_k_max,term, 0] += np.sum(np.real(C00exp[term][ind]))
                    avgWij[delta_k+self.delta_k_max,term, 1] += np.sum(np.real(C22exp[term][ind]))
                    avgWij[delta_k+self.delta_k_max,term, 2] += np.sum(np.real(C44exp[term][ind]))
                    avgWij[delta_k+self.delta_k_max,term, 3] += np.sum(np.real(C20exp[term][ind]))
                    avgWij[delta_k+self.delta_k_max,term, 4] += np.sum(np.real(C40exp[term][ind]))
                    avgWij[delta_k+self.delta_k_max,term, 5] += np.sum(np.real(C42exp[term][ind]))
        
        # divide by the number of k-modes to get the average
        for i in range(0,2*self.delta_k_max+1):
            if(i+kbin_idx-self.delta_k_max>=self.nBins[zbin_idx] or i+kbin_idx-self.delta_k_max<0): 
                avgWij[i,:,:] = 0
            else:
                avgWij[i,:,:] /= (norm * Nmodes[kbin_idx + i - self.delta_k_max] * self.I22[zbin_idx]**2)
                #avgWij[i]/=(norm*self.Bin_ModeNum[kbin_idx+i-self.delta_k_max]*self.I22**2)
        
        avgWij[:,:,0]*=self.ell_factor(0, 0)
        avgWij[:,:,1]*=self.ell_factor(2, 2)
        avgWij[:,:,2]*=self.ell_factor(4, 4)
        avgWij[:,:,3]*=self.ell_factor(2, 0)
        avgWij[:,:,4]*=self.ell_factor(4, 0)
        avgWij[:,:,5]*=self.ell_factor(4, 2)
        
        return(avgWij)