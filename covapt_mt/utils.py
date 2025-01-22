# -----------------------------------------------
# this file contains all necesary file paths to use this repository
import yaml
import numpy as np

def load_config_file(config_file:str):
    """loads in the emulator config file as a dictionary object
    
    Parameters
    ----------
        config_file: Config file path and name to laod

    Returns
    ----------
        config_dict: dictionary
            
    """
    with open(config_file, "r") as file:
        try:
            config_dict = yaml.load(file, Loader=yaml.FullLoader)
        except:
            print("ERROR! Couldn't read yaml file")
            raise IOError
    
    return config_dict

# Functions taken from thecov (https://github.com/cosmodesi/thecov)
def sample_from_shell(rmin : float, rmax : float, discrete=True):
    """Sample a point uniformly from a spherical shell.

    Args:
        rmin : Minimum radius of the shell.
        rmax : Maximum radius of the shell.
        discrete : bool, optional
            If True, the sampled point will be rounded to the nearest integer.
            Default is True.

    Returns:
        x,y,z,r : Coordinates of the sampled point.
    """

    r = rmin + (rmax - rmin) * np.random.rand()
    theta = 2 * np.pi * np.random.rand()
    phi = np.arccos(1 - 2 * np.random.rand())

    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    if(discrete):
        x,y,z = int(np.round(x)), int(np.round(y)), int(np.round(z))
        r = np.sqrt(x**2 + y**2 + z**2)
        if(r < rmin or r > rmax):
            return sample_from_shell(rmin, rmax, discrete)

    return x,y,z,r

def nmodes(volume : float, kmin : float, kmax : float):
    '''Compute the number of modes in a given shell.

    Args:
        volume : Volume of the survey.
        kmin : Minimum k of the shell in h/Mpc.
        kmax : Maximum k of the shell in h/Mpc.

    Returns:
        N: Number of modes.
    '''
    return volume / 3. / (2*np.pi**2) * (kmax**3 - kmin**3)

def flip_axes(cov, nps:int, nk:int, nl:int):
    """flips k and l dimensions of the given input covariance matrix
    
    Args:
        cov: {np array} block covariance matrix with elements ordered as [nps, nl, nk]
        nps: number of (auto + cross) power spectra in the covariance matrix
        nk: number of k bins
        nl: number of ells
    
    Returns:
        new_cov: {np array} block covariance matrix with elements ordered as [nps, nk, nl]
    """
    tmp_cov = cov.reshape(nps, nl, nk, nps, nl, nk)
    tmp_cov = tmp_cov.transpose(0, 2, 1, 3, 5, 4)
    new_cov = tmp_cov.reshape(nps*nk*nl, nps*nk*nl)
    return new_cov

def test_matrix(cov: list, num_spectra: int, num_kbins : list):
    """Tests if the given list of covariacne matrices, and all sub-blocks, are positive-definite
    
    Args:
        cov: list of covariance matrices (each stored as numpy arrays)
        num_spectra: number of (auto + cross) power spectra that make up the covariance matrix
        num_kbins: number of k bins
    """
    
    for z in range(len(cov)):

        try:
            L = np.linalg.cholesky(cov[z])
            print("Covariance matrix for zbin " + str(z) + " is positive definite! :)")
        except:
            print("WARNING! Covariance matrix for zbin " + str(z) + " is not positive definite!")
            eigvals = np.linalg.eigvals(cov[z])
            print("There are {:0.0f} negative eigenvalues, smallest value = {:0.3e}".format(len(eigvals[(eigvals < 0)]), np.amin(eigvals)))

        # test if partial matrices are positive-definite
        sub_test = 0
        for i in range(int(num_spectra)):
            for j in range(int(num_spectra)):
                C_sub = cov[z][i*2*int(num_kbins[z]): (i+1)*2*int(num_kbins[z]),j*2*int(num_kbins[z]): int((j+1)*2*num_kbins[z])]
                try:
                    L = np.linalg.cholesky(C_sub)
                    #print("Partial covariance matrix ({:0.0f}, {:0.0f}) is positive-definite :)".format(i, j))
                except:
                    print("Partial covariance matrix ({:0.0f}, {:0.0f}, {:0.0f}) is NOT positive-definite".format(z, i, j))
                    sub_test = 1

        if sub_test == 0: print("All sub-matrices are positive-definite :)")

        cond = np.linalg.cond(cov[z])
        print("Condition number = {:0.3e}".format(cond))