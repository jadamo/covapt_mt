# This script calculates the window function used by the Gaussian covariance term in CovaPT
# NOTE: It is highly recommended you run this on multiple cpu cores, or on an hpc cluster
# Source, Jay Wadekar: https://github.com/JayWadekar/CovaPT

import time, os, sys, tqdm
import numpy as np
from itertools import repeat
from multiprocessing import Pool
from covapt_mt import window
from covapt_mt.utils import load_config_file

def main():
    
    if len(sys.argv) < 2:
        print("USAGE: python scripts/make_window_function.py <config_file>")
        return 0
    
    config_dict = load_config_file(sys.argv[1])

    if config_dict["make_gaussian_window"] == False and config_dict["make_ssc_window"] == False:
        return 0

    print("Calculating Gaussian window functions:", config_dict["make_gaussian_window"])
    print("Calculating SSC window functions:", config_dict["make_ssc_window"])

    # TODO: update this once I get a k array file from Chen / Henry
    k_data = np.load(config_dict["input_dir"]+config_dict["k_array_file"])
    num_zbins = int(len(config_dict["zbins"]) / 2)
    k_centers = []
    for zbin in range(num_zbins):
        key = "k_"+str(zbin)
        k_centers.append(k_data[key])

    survey_kernels = window.Survey_Geometry_Kernels(config_dict["h"], 
                                                    config_dict["Om0"],
                                                    config_dict["zbins"],
                                                    k_centers,
                                                    config_dict["box_size"],
                                                    config_dict["input_dir"],
                                                    config_dict["random_file_prefix"])

    if not os.path.exists(config_dict["output_dir"]+'FFTWinFun.npy'):
        print("\nStarting FFT calculations...")
        t1 = time.time()
        export = survey_kernels.calc_gaussian_kernels(config_dict["fft_mesh_size"], config_dict["box_size"])
        t2 = time.time()
        print('Done! Run time: {:.0f}m {:.0f}s'.format((t2-t1) // 60, (t2-t1) % 60))

        save_file = config_dict["output_dir"]+'FFTWinFun.npy'
        np.save(save_file,export)
        print("FFTs saved to", save_file)

    if config_dict["make_gaussian_window"]:
        t1 = time.time()
        # TODO: find a faster way to loop thru redshift bins
        # For now, save the window functions in seperate files
        for z_idx in range(num_zbins):
            print("Starting Gaussian window function generation for zbin {:0.0f} with {:0.0f} processes...".format(z_idx, config_dict["num_processes"]))
            #Wij = np.zeros((len(k_centers[z_idx]), 7, 15, 6))
            k_idx = range(len(k_centers[z_idx]))

            survey_kernels.load_fft_file(config_dict["output_dir"], z_idx)
            #for tqdm(range(len(k_centers[z_idx])), desc='Computing window kernels'):
            p = Pool(processes=config_dict["num_processes"])
            Wij=p.starmap(survey_kernels.calc_gaussian_window_function, 
                          zip(k_idx, repeat(z_idx), repeat(config_dict["kmodes_sampled"])))
            p.close()
            p.join()

            save_file = config_dict["output_dir"]+config_dict["window_file_prefix"]+str(z_idx)+".npy"
            np.save(save_file, Wij)
            print("window function saved to", save_file)
        t2 = time.time()
        print('Done! Run time: {:.0f}m {:.0f}s'.format((t2-t1) // 60, (t2-t1) % 60))

    if config_dict["make_ssc_window"]:
        print("WARNING! This functionality does not work right now!")
        print("\nStarting FFT calculations...")
        t1 = time.time()
        P_W = survey_kernels.calc_SSC_window_function(config_dict["fft_mesh_size"], config_dict["box_size"])
        t2 = time.time()
        print('Done! Run time: {:.0f}m {:.0f}s'.format((t2-t1) // 60, (t2-t1) % 60))

        save_file = config_dict["output_dir"]+'WindowPowers.npy'
        np.save(save_file,P_W)
        print("SSC window functions saved to", save_file)

if __name__ == "__main__":
    main()