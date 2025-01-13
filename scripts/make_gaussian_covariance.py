import os, sys
import numpy as np
from covapt_mt.covapt import covariance_model
from covapt_mt.utils import load_config_file, test_matrix

def main():

    if len(sys.argv) < 2:
        print("USAGE: python scripts/make_window_function.py <config_file>")
        return 0
    config_dict = load_config_file(sys.argv[1])

    if config_dict["make_Gaussian_cov"] == False:
        return 0

    print("\nGenerating Gaussian covariance matrix...")
    num_zbins = int(len(config_dict["zbins"]) / 2)
    model = covariance_model(config_dict["num_tracers"],
                             num_zbins, 
                             config_dict["input_dir"] + config_dict["k_array_file"],
                             config_dict["n_galaxy"], 
                             config_dict["alpha"],  
                             window_dir=config_dict["output_dir"] + config_dict["window_file_prefix"])
    model.load_power_spectrum(config_dict["input_dir"] + config_dict["pk_galaxy_file"])
    C_G = model.get_mt_gaussian_covariance()

    # test if matrix (and all sub-matrices) is positive definite
    test_matrix(C_G, model.num_spectra, int(model.num_kbins))

    keys = list(range(model.num_zbins))
    save_file = config_dict["output_dir"] + config_dict["covariance_file"]
    save_data = {}
    for z in range(model.num_zbins):
        save_data["zbin_"+str(keys[z])] = C_G[z]
    print("Saving to " + save_file)
    np.savez(save_file,**save_data)

if __name__ == "__main__":
    main()