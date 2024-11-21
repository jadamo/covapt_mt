import os, sys
import numpy as np
from covapt_mt.covapt import covariance_model
from covapt_mt.utils import load_config_file

def main():

    if len(sys.argv) < 2:
        print("USAGE: python scripts/make_window_function.py <config_file>")
        return 0
    config_dict = load_config_file(sys.argv[1])

    if config_dict["make_Gaussian_cov"] == False:
        return 0

    print("\nGenerating Gaussian covariance matrix...")

    if not os.path.exists(config_dict["output_dir"] + config_dict["window_file"]):
        print("ERROR! Can't find window function file! Aborting...")
        return 0

    k_centers = np.loadtxt(config_dict["input_dir"]+config_dict["k_array_file"])

    model = covariance_model(2, 2, 
                             config_dict["n_galaxy"], 
                             config_dict["alpha"], 
                             k_centers, 
                             config_dict["output_dir"] + config_dict["window_file"])
    model.load_power_spectrum(config_dict["input_dir"] + config_dict["pk_galaxy_file"])
    C_G = model.get_mt_gaussian_covariance()

    # test if matrix is positive definite
    for z in range(model.num_zbins):
        try:
            L = np.linalg.cholesky(C_G[z])
            print("Covariance matrix for zbin " + str(z) + " is positive definite! :)")
        except:
            print("ERROR! Covariance matrix for zbin " + str(z) + " is not positive definite!")

    save_file = config_dict["output_dir"] + config_dict["covariance_file"]
    print("Saving to " + save_file)
    np.save(save_file, C_G)

if __name__ == "__main__":
    main()