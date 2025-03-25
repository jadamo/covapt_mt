import os, sys
import numpy as np
from covapt_mt.covapt import covariance_model
from covapt_mt.utils import load_config_file, test_matrix, flip_axes

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
                             config_dict["alpha"],  
                             window_dir=config_dict["output_dir"])
    model.load_power_spectrum(config_dict["input_dir"] + config_dict["pk_galaxy_file"])
    C_G = model.get_mt_gaussian_covariance()

    # test if matrix (and all sub-matrices) is positive definite
    test_matrix(C_G, model.num_spectra, model.num_kbins)

    # Reformat to the shape Cosmo_Inference expects
    num_spectra = int(config_dict["num_tracers"]*(config_dict["num_tracers"]+1)/2)
    num_ells = 2
    C_G_reshaped = flip_axes(C_G, num_spectra, len(model.get_k_bins()[0]), num_ells)
    print(C_G_reshaped.shape)

    if config_dict["save_inverse"] == True:
        print("Inverting covariance...")
        for z in range(model.num_zbins):
            C_G_reshaped[z] = np.linalg.inv(C_G_reshaped[z])
        save_file = config_dict["output_dir"] + "invcov.npy"

    else:
        save_file = config_dict["output_dir"] + "cov.npy"

    print("Saving to " + save_file)
    np.save(save_file, C_G_reshaped)

if __name__ == "__main__":
    main()