# This script calculates the window function used by the Gaussian covariance term in CovaPT
# NOTE: It is highly recommended you run this on multiple cpu cores, or on an hpc cluster
# Source, Jay Wadekar: https://github.com/JayWadekar/CovaPT

import time, os, sys
import numpy as np
from itertools import repeat
from multiprocessing import Pool
from covapt_mt import window
from covapt_mt.utils import load_config_file

# ints specifying which redshift and sample bins to generate windows for
# there are 5 sample bins and 11 redshift bins
zbin = 1
sample_bin = 1

def main():
    
    if len(sys.argv) < 2:
        print("USAGE: python scripts/make_window_function.py <config_file>")
        return 0
    
    config_dict = load_config_file(sys.argv[1])

    if config_dict["make_gaussian_window"] == False and config_dict["make_ssc_window"] == False:
        return 0

    print("Calculating Gaussian window functions:", config_dict["make_gaussian_window"])
    print("Calculating SSC window functions:", config_dict["make_ssc_window"])

    survey_kernels = window.Survey_Window_Kernels(config_dict["h"], 
                                                  config_dict["Om0"],
                                                  zbin,
                                                  sample_bin,
                                                  config_dict["input_dir"],
                                                  config_dict["random_file"])
    I22 = survey_kernels.I22
    k_centers = np.loadtxt(config_dict["input_dir"]+config_dict["k_array_file"])

    print("\nStarting FFT calculations...")
    t1 = time.time()
    export = survey_kernels.calc_gaussian_kernels(config_dict["fft_mesh_size"], config_dict["box_size"])
    t2 = time.time()
    print('Done! Run time: {:.0f}m {:.0f}s'.format((t2-t1) // 60, (t2-t1) % 60))

    save_file = config_dict["output_dir"]+'FFTWinFun_sample_'+str(sample_bin)+'_redshift_'+str(zbin)+'.npy'
    np.save(save_file,export)
    print("FFTs saved to", save_file)

    if config_dict["make_gaussian_window"]:
        print("\nStarting Gaussian window function generation with {:0.0f} processes...".format(config_dict["num_processes"]))
        t1 = time.time()
        window_kernels = window.Gaussian_Window_Kernels(config_dict["output_dir"],
                                                        k_centers, 
                                                        zbin, 
                                                        sample_bin, 
                                                        config_dict["box_size"], 
                                                        I22)
        idx = range(len(k_centers))
        nBins = len(k_centers)

        p = Pool(processes=config_dict["num_processes"])
        WinFunAll=p.starmap(window_kernels.calc_gaussian_window_function, 
                            zip(idx, repeat(config_dict["kmodes_sampled"])))
        p.close()
        p.join()

        t2 = time.time()
        print('Done! Run time: {:.0f}m {:.0f}s'.format((t2-t1) // 60, (t2-t1) % 60))

        #save_file = covapt_data_dir+'Wij_k'+str(nBins)+'_sample_'+str(sample_bin)+'_redshift_'+str(zbin)+'.npy'
        save_file = config_dict["output_dir"]+config_dict["window_file"]
        b=np.zeros((len(idx),7,15,6))
        for i in range(len(idx)):
            b[i]=WinFunAll[i]
        np.save(save_file, b)
        print("window function saved to", save_file)

    if config_dict["make_ssc_window"]:
        print("WARNING! This functionality has not been tested!")
        print("\nStarting FFT calculations...")
        t1 = time.time()
        P_W = survey_kernels.calc_SSC_window_function(config_dict["fft_mesh_size"], config_dict["box_size"])
        t2 = time.time()
        print('Done! Run time: {:.0f}m {:.0f}s'.format((t2-t1) // 60, (t2-t1) % 60))

        save_file = config_dict["output_dir"]+'WindowPowers_sample_'+str(sample_bin)+'_redshift_'+str(zbin)+'.npy'
        np.save(save_file,P_W)
        print("SSC window functions saved to", save_file)

if __name__ == "__main__":
    main()