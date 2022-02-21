import numpy as np
from skimage.transform import resize
from projection import flow_img

"""Base = np.random.normal(size = (200,200))
Base = gaussian_filter(Base, 5.)

#A, B = Base[:50,:50], Base[5:55,3:53]
A = np.array(Image.open("PhyDNet-master/data/mobotix3-planarRB256/2020-07-02/2020-07-02_11_23.png"))[:,:]/255
B = np.array(Image.open("PhyDNet-master/data/mobotix3-planarRB256/2020-07-02/2020-07-02_11_24.png"))[:,:]/255

#A = gaussian_filter(A,2.)
#B = gaussian_filter(B,2.)

dx, dy = 17, -17

#A, B = A[30:130,30:130], B[30:130,30:130] # A[30+dx:130+dx,30+dy:130+dy]


plt.subplot(1,2,1).imshow(A)
plt.subplot(1,2,2).imshow(B)
plt.show()
"""


def simple_optical_flow(A, B, GradB, W):

    """
    Computes the Lukas-Kanade optical flow
    
    OF = - (GradB^T * W * GradB)^(-1) * GradB^T * W * (B - A)
    
    Parameters
    ----------
    A : np.ndarray
        Reference image
    B : np.ndarray
        Target image
    GradB : np.ndarray
        Gradients used for computations
    W : np.ndarray
        Weights used for computations
    
    Returns
    -------
    result : np.ndarray of size 2
        Computed optical flow (di, dj)
    """

    W = W.flatten()
    try:
        L0 = np.linalg.inv(GradB.T.dot(np.einsum("i,ij->ij", W, GradB))).dot(GradB.T)

        Y0 = W*(B.flatten() - A.flatten())

        theta = -L0.dot(Y0)

        return theta
    except:
        return np.array([0.,0.])


class Pyramid_OF():

    """
    A class used to perform Lukas-Kanade pyramidal iterative optical flow computations.

    Construction::
      pyramid = pyramid_OF(D, levels=3, k=1/2, dphi_calc=None)
    
    Parameters
    ----------
    D : int
        Size of input images
    levels : int, optional
        Number of supplementary levels in the pyramid
    k : float, optional
        Size reduction factor when moving to a higher layer
    dphi_calc : func, optional
        Deprecated
    """

    def __init__(self, D, levels=3, k=1/2, dphi_calc=None):

        self.D = D
        self.levels = levels


        self.dims = [D]
        if dphi_calc is None:
            self.dphis = None
        else:
            self.dphis = [dphi_calc(D//2,(D,D)).reshape((D*D,2,2))]

        for level in range(1,levels+1):
            D = D*k
            int_d = int(D)
            self.dims.append(int_d)
            if dphi_calc is not None:
                self.dphis.append(dphi_calc(int_d//2,(int_d,int_d)).reshape((int_d*int_d,2,2)))

    def compute_resized(self, B, W_B=None):
        """
        Processes an image into the pyramid

        Parameters
        ----------
        B : np.ndarray of shape (D,D)
            Single channel image to be used as target image for the impending optical flow computation           
        W_B : np.ndarray of shape (D,D), optional
            Weights corresponding  to pixels of image B. If None, these weights are set to 1.

        Returns
        -------
        None
        """

        self.Bs = [B]

        if W_B is None:
            self.W_Bs = [np.ones_like(B)]
        else:
            self.W_Bs = [W_B]

        for level in range(1,self.levels+1):
            self.Bs.append(resize(B,(self.dims[level],self.dims[level]), anti_aliasing=True))

            if W_B is None:
                self.W_Bs.append(np.ones_like(self.Bs[-1]))
            else:
                self.W_Bs.append(resize(W_B,(self.dims[level],self.dims[level]), anti_aliasing=True))

    def compute_gradient(self, sigma=1.):
        """
        Computes the (gaussian) gradient of the last processed image

        Parameters
        ----------
        sigma : float, optional
            Standard deviation of gaussian convolution used to compute the gradient

        Returns
        -------
        None
        """

        def gaussian(x, sigma):
            return 1/(sigma*(2*np.pi)**0.5) * np.exp(-(x**2)/(2*sigma**2))

        def der_x_gaussian_filter(x, sigma):
            return -x/sigma**2*gaussian(x,sigma)

        def filter_g(sigma):
            limit = np.ceil(4*sigma)
            filter_values = np.arange(-limit, limit+1, 1)
            return der_x_gaussian_filter(filter_values, sigma)


        def gradient_image_x(img, sigma):
            gradient = np.zeros(img.shape[:2])
            filte = filter_g(sigma)
            for i in range(img.shape[0]):
                convolution = np.convolve(filte, img[i,:], mode="same")
                gradient[i,:] = convolution
            return gradient

        self.Grads = []

        for i, B in enumerate(self.Bs):

            gradient_j = gradient_image_x(B, sigma).flatten()
            gradient_i = gradient_image_x(B.T, sigma).T.flatten()
            grad = np.column_stack([gradient_i,gradient_j])
            if self.dphis is not None:
                grad = np.einsum('...j,...ji', grad, self.dphis[i])
            self.Grads.append(grad)

            limit = int(np.ceil(4*sigma))
            mask = np.ones_like(B, dtype=bool)
            mask[limit:-limit,limit:-limit] = False
            self.W_Bs[i][mask] = 0.

    def compute_windowed_OF(self, windows, iters=2, iters_upper_layer=4):
        """
        Computes the iterative pyramidal optical flow on a list of windows

        Parameters
        ----------
        windows : list of np.ndarrays of shape (D,D) and dtype float
            List of windows where the local optical flow is computed
        iters : int, optional
            Number of iterations performed on each layer of the pyramid except the highest one
        windows : int, optional
            Number of iterations performed on the highest layer of the pyramid

        Returns
        -------
        (flows, window_weights)

        flows : np.ndarray of shape (len(windows), 2)
            Optical flow computed for each window
        window_weights : np.ndarray of size len(windows)
            Sum of weights*values for each window (used for experimental purposes)        
        """

        res = np.zeros((len(windows), 2))
        window_weights = np.zeros(len(windows))

        for k, (center, window) in enumerate(windows):

            window_weights[k] = np.sum(self.W_Bs[0]*window*self.Bs[0])

            theta_tot = np.zeros(2)
            for level in range(self.levels,-1,-1):
                if level == self.levels[-1]:
                    n = iters_upper_layer
                else:
                    n = iters
                for _ in range(n):
                    flownA, flownW = flow_img(self.As[level], theta_tot, self.W_As[level])
                    resized_window = resize(window.astype(np.float32), self.As[level].shape)
                    theta_tot += simple_optical_flow(flownA, self.Bs[level], self.Grads[level], flownW*self.W_Bs[level]*resized_window)

                if level:
                    theta_tot *= self.dims[level-1]/self.dims[level]

            if np.any(np.abs(theta_tot) > 60):
                res[k] = np.array([0.,0.])
            else:
                res[k] = theta_tot
        
        return res, window_weights

    def step(self):
        """
        Replaces the last reference image for calculating the optical flow with the last target image

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        self.As = self.Bs
        self.W_As = self.W_Bs