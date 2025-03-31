# This script calculates the window function used by the Gaussian covariance term in CovaPT
# NOTE: It is highly recommended you run this on multiple cpu cores, or on an hpc cluster
# Source, Jay Wadekar: https://github.com/JayWadekar/CovaPT

import time, os, sys, tqdm
import numpy as np
from itertools import repeat
from multiprocessing import Pool
from covapt_mt import window
from covapt_mt.utils import load_config_file

def make_window_function(yaml_file):
    
    config_dict = load_config_file(yaml_file)

    if config_dict["make_gaussian_window"] == False and config_dict["make_ssc_window"] == False:
        return 0

    print("Calculating Gaussian window functions:", config_dict["make_gaussian_window"])
    print("Calculating SSC window functions:", config_dict["make_ssc_window"])

    # TODO: update this once I get a k array file from Chen / Henry
    k_data = np.load(config_dict["input_dir"]+config_dict["k_array_file"])
    num_zbins = int(len(config_dict["zbins"]) / 2)
    num_tracers = int(config_dict["num_tracers"])
    k_centers = []
    for idx in range(num_zbins*num_tracers):
        #key = "k_"+str(idx)
        key = "k"
        k_centers.append(k_data[key])# / 0.7)

    survey_kernels = window.Survey_Geometry_Kernels(config_dict, k_centers)
    
    if not os.path.exists(config_dict["output_dir"]+'FFTWinFun.npy') or \
       config_dict["make_random_ffts"] == True:
        
        print("\nStarting FFT calculations...")
        t1 = time.time()
        export = survey_kernels.calc_gaussian_kernels(config_dict["fft_mesh_size"])
        t2 = time.time()
        print('Done! Run time: {:.0f}m {:.0f}s'.format((t2-t1) // 60, (t2-t1) % 60))

        save_file = config_dict["output_dir"]+'FFTWinFun.npy'
        np.save(save_file,export)
        print("FFTs saved to", save_file)

    if config_dict["make_gaussian_window"]:
        t1 = time.time()
        # TODO: find a faster way to loop thru redshift bins
        # For now, save the window functions in seperate files
        for idx in range(num_zbins*num_tracers):
            print("Sampling {:0.0f} kmodes for bin {:0.0f} with {:0.0f} processes...".format(config_dict["kmodes_sampled"], idx, config_dict["num_processes"]))
            #Wij = np.zeros((len(k_centers[z_idx]), 7, 15, 6))
            k_idx = range(len(k_centers[idx]))

            survey_kernels.load_fft_file(config_dict["output_dir"], idx)
            #for tqdm(range(len(k_centers[z_idx])), desc='Computing window kernels'):
            p = Pool(processes=config_dict["num_processes"])
            Wij=p.starmap(survey_kernels.calc_gaussian_window_function, 
                          zip(k_idx, repeat(idx), repeat(config_dict["kmodes_sampled"])))
            p.close()
            p.join()

            save_file = config_dict["output_dir"]+"Wij"+str(idx)+".npy"
            np.save(save_file, Wij)
            print("window function saved to", save_file)
        t2 = time.time()
        print('Done! Run time: {:.0f}m {:.0f}s'.format((t2-t1) // 60, (t2-t1) % 60))

    if config_dict["make_ssc_window"]:
        print("WARNING! This functionality does not work right now!")
        raise NotImplementedError
        # print("\nStarting FFT calculations...")
        # t1 = time.time()
        # P_W = survey_kernels.calc_SSC_window_function(config_dict["fft_mesh_size"], config_dict["box_size"])
        # t2 = time.time()
        # print('Done! Run time: {:.0f}m {:.0f}s'.format((t2-t1) // 60, (t2-t1) % 60))

        # save_file = config_dict["output_dir"]+'WindowPowers.npy'
        # np.save(save_file,P_W)
        # print("SSC window functions saved to", save_file)

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("USAGE: python -m covapt_mt.scripts.make_window_function <config_file>")
        return 0
    
    yaml_file = sys.argv[1]

    make_window_function(yaml_file)