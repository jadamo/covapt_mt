# -----------------------------------------------
# this file contains all necesary file paths to use this repository
import yaml
import numpy as np

def load_config_file(config_file:str):
    """loads in the emulator config file as a dictionary object
    
    Args:
        config_file: Config file path and name to laod
    """
    with open(config_file, "r") as file:
        try:
            config_dict = yaml.load(file, Loader=yaml.FullLoader)
        except:
            print("ERROR! Couldn't read yaml file")
            raise IOError
    
    return config_dict

# Functions taken from thecov (https://github.com/cosmodesi/thecov)
def sample_from_shell(rmin, rmax, discrete=True):
    """Sample a point uniformly from a spherical shell.

    Parameters
    ----------
    rmin : float
        Minimum radius of the shell.
    rmax : float
        Maximum radius of the shell.
    discrete : bool, optional
        If True, the sampled point will be rounded to the nearest integer.
        Default is True.

    Returns
    -------
    x,y,z,r : float
        Coordinates of the sampled point.
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

def nmodes(volume, kmin, kmax):
    '''Compute the number of modes in a given shell.

    Parameters
    ----------
    volume : float
        Volume of the survey.
    kmin : float
        Minimum k of the shell.
    kmax : float
        Maximum k of the shell.

    Returns
    -------
    float
        Number of modes.
    '''
    return volume / 3. / (2*np.pi**2) * (kmax**3 - kmin**3)

def test_matrix(cov, num_zbins, num_spectra, num_kbins):

    # test if full matrix is positive definite
    for z in range(num_zbins):

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
                C_sub = cov[z][i*2*num_kbins: (i+1)*2*num_kbins,j*2*num_kbins: (j+1)*2*num_kbins]
                try:
                    L = np.linalg.cholesky(C_sub)
                    #print("Partial covariance matrix ({:0.0f}, {:0.0f}) is positive-definite :)".format(i, j))
                except:
                    print("Partial covariance matrix ({:0.0f}, {:0.0f}, {:0.0f}) is NOT positive-definite".format(z, i, j))
                    sub_test = 1

        if sub_test == 0: print("All sub-matrices are positive-definite :)")

        cond = np.linalg.cond(cov[z])
        print("Condition number = {:0.3e}".format(cond))