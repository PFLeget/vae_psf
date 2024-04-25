# Simple psf simulation where psf profile is a 2D Gaussian
# with ellipticity, shift and size parameters. The PSF is not
# on a grid, but is evaluated at a set of points. The PSF is
# normalized to have a total flux of 1.


import numpy as np
import pickle

def get_correlation_length_matrix(size, e1, e2):
    """
    Produce correlation matrix to introduce anisotropy in kernel.
    Used same parametrization as shape measurement in weak-lensing
    because this is mathematicaly equivalent (anistropic kernel
    will have an elliptical shape).

    :param correlation_length: Correlation lenght of the kernel.
    :param g1, g2:             Shear applied to isotropic kernel.
    """
    if abs(e1)>1:
        e1 = 0
    if abs(e2)>1:
        e2 = 0
    e = np.sqrt(e1**2 + e2**2)
    q = (1-e) / (1+e)
    phi = 0.5 * np.arctan2(e2,e1)
    rot = np.array([[np.cos(phi), np.sin(phi)],
                    [-np.sin(phi), np.cos(phi)]])
    ell = np.array([[size**2, 0],
                    [0, (size * q)**2]])
    cov = np.dot(rot.T, ell.dot(rot))
    return cov

def make_2dgaussian(x, y, x0=0, y0=0, size=1, e1=0, e2=0):
    """
    Generate a 2D Gaussian distribution.

    Parameters:
    - x (float or array-like): x-coordinate(s) of the point(s) at which to evaluate the Gaussian.
    - y (float or array-like): y-coordinate(s) of the point(s) at which to evaluate the Gaussian.
    - x0 (float, optional): x-coordinate of the center of the Gaussian. Default is 0.
    - y0 (float, optional): y-coordinate of the center of the Gaussian. Default is 0.
    - size (float, optional): size of the Gaussian. Default is 1.
    - e1 (float, optional): first ellipticity parameter. Default is 0.
    - e2 (float, optional): second ellipticity parameter. Default is 0.

    Returns:
    - gauss (float or array-like): the value(s) of the 2D Gaussian distribution evaluated at the given coordinates.
    """
    cov = get_correlation_length_matrix(size, e1, e2)
    det_c = np.linalg.det(cov)
    w = np.linalg.inv(cov)
    coord = np.array([x-x0, y-y0]).T
    gauss = np.zeros_like(x)
    for i in range(len(gauss)):
        norm = 1. / np.sqrt((2. * np.pi)**2 * det_c)
        gauss[i] = norm * np.exp(-0.5 * coord[i] @ w @ coord[i].T)
    return gauss


def make_mock_gaussian_psf(n_psfs=1000, n_pixel=16,
                           mean_size=3, std_size=0.2,
                           mean_e1=0, std_e1=0.3,
                           mean_e2=0, std_e2=0.3):
    """
    Generate a set of mock Gaussian PSFs.

    Parameters:
    - n_psfs (int): Number of PSFs to generate (default: 1000).
    - n_pixel (int): Number of pixels in each dimension of the PSF (default: 16).
    - mean_size (float): Mean size of the PSFs (default: 3).
    - std_size (float): Standard deviation of the size of the PSFs (default: 0.2).
    - mean_e1 (float): Mean ellipticity component e1 of the PSFs (default: 0).
    - std_e1 (float): Standard deviation of the ellipticity component e1 of the PSFs (default: 0.3).
    - mean_e2 (float): Mean ellipticity component e2 of the PSFs (default: 0).
    - std_e2 (float): Standard deviation of the ellipticity component e2 of the PSFs (default: 0.3).

    Returns:
    - psfs (ndarray): Array of generated PSFs with shape (n_psfs, n_pixel, n_pixel).
    - x (ndarray): Array of x-coordinates of the PSF grid.
    - y (ndarray): Array of y-coordinates of the PSF grid.
    - size (ndarray): Array of sizes of the generated PSFs.
    - e1 (ndarray): Array of ellipticity component e1 of the generated PSFs.
    - e2 (ndarray): Array of ellipticity component e2 of the generated PSFs.
    """
    np.random.seed(42)
    size = np.random.normal(scale=std_size, size=n_psfs) + mean_size
    e1 = np.random.normal(scale=std_e1, size=n_psfs) + mean_e1
    e2 = np.random.normal(scale=std_e2, size=n_psfs) + mean_e2
    psfs = []

    x = np.linspace(-10, 10, n_pixel)
    y = np.linspace(-10, 10, n_pixel)
    x, y = np.meshgrid(x, y)
    x = x.reshape(-1)
    y = y.reshape(-1)

    for i in range(n_psfs):
        psf = make_2dgaussian(x, y, x0=0, y0=0,
                              size=size[i], e1=e1[i], e2=e2[i])
        psfs.append(psf)

    psfs = np.array(psfs)
    psfs = psfs.reshape((n_psfs, n_pixel, n_pixel))

    return psfs, x, y, size, e1, e2

def make_mock(file_out='mock.pkl',
              plot_mock=False,
              **kwargs):
    """
    Generates a mock dataset of PSFs using the specified parameters.

    Parameters:
    - file_out (str): The output file path where the generated mock dataset will be saved. Default is 'mock.pkl'.
    - plot_mock (bool): Whether to plot the generated mock dataset. If True, the PSFs will be displayed using matplotlib. Default is False.
    - **kwargs: Additional keyword arguments that can be passed to the make_mock_gaussian_psf function.

    Returns:
    None

    Example usage:
    make_mock(file_out='mock.pkl', plot_mock=True, n_psfs=1000, n_pixel=16,
              mean_size=3, std_size=0.2, mean_e1=0, std_e1=0.3, mean_e2=0, std_e2=0.3)
    """
    psfs, x, y, size, e1, e2 = make_mock_gaussian_psf(**kwargs)

    if plot_mock:
        import matplotlib.pyplot as plt
        max_value = np.max(psfs) / 3.
        for i in range(30):
            plt.figure()
            plt.imshow(psfs[i], vmin=-max_value, vmax=max_value, cmap=plt.cm.seismic)
            plt.gca().invert_yaxis()
            plt.colorbar()
        plt.show()

    dic_out = {'psfs': psfs,
               'x': x,
               'y': y,
               'size': size,
               'e1': e1,
               'e2': e2}

    f = open(file_out, 'wb')
    pickle.dump(dic_out, f)
    f.close()

if __name__ == "__main__":

   make_mock(file_out='mock.pkl', plot_mock=True, n_psfs=1000, 
             n_pixel=16, mean_size=4, std_size=0.2, 
             mean_e1=0, std_e1=0.1, mean_e2=0, std_e2=0.1)

