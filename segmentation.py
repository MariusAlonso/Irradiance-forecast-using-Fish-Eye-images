import cv2
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
from skimage.segmentation import watershed
from skimage.transform import resize

import numpy as np


import matplotlib.pyplot as plt


def segmenter(A, A_RB, plots=False):
    """
    Segment the clouds from a hemispheric sky image

    Parameters
    ----------
    A : np.ndarray of shape (512,512,3)
        Hemispherical sky image to apply segmentation on.
    A_RB : np.ndarray of shape (512,512)
        Channel to apply segmentation on (only tested with (R-B)/(R+B)).
    plots : bool
        Display the steps and results of segmentation

    Returns
    -------
    result : np.ndarray of size 512x512 and type int
        1 if the pixel corresponds to a cloud, 0 if not
    """

    D = A.shape[0]

    # Segmentation domain
    Domain = np.sum(np.square(np.moveaxis(np.indices((D,D)),0,2) - np.array([D//2, D//2])), axis=2) < (D*0.48)**2
    
    # Solar pixels
    sol = (np.sum(A, axis=2) > 1.3)&Domain


    if np.sum(sol) < 20:
        # Case where the sun is not or barely visible : no halo reduction
        print("No sun segmentation")
        return ((A_RB > 0.03) & Domain & (~ sol)).astype(int)

    else:

        ##################
        # Halo reduction #
        ##################

        #We compute the center of the sun
        M = cv2.moments(sol.astype(float))
        x_maxi, y_maxi = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

        # We compute R : Distance from the sun
        Relative = np.moveaxis(np.indices((D,D)),0,2) - np.array([y_maxi, x_maxi])
        R = np.sum(np.square(Relative), axis=2)**0.5

        # We compute Theta : Angle relative to the direction (0,-1), ranging from -pi/2 to 3*pi/2
        null_mask = Relative[:,:,0] == 0
        Theta = np.zeros_like(R)
        Theta[~null_mask] = np.arctan(Relative[:,:,1][~null_mask]/Relative[:,:,0][~null_mask]) + np.pi*(1-np.sign(Relative[:,:,0][~null_mask]))/2
        Theta[null_mask] = np.pi*np.sign(Relative[:,:,1][null_mask])/2


        if plots:
            plt.subplot(1,2,1).imshow(A_RB)
            plt.subplot(1,2,2).imshow(A_Gradient)
            plt.show()

        def regularizer(image, n_radial_disks=50, n_ang_sections=9, minimize_slope=True):
            """
            Reduces the halo surrounding the sun

            Parameters
            ----------
            image : np.ndarray of shape (512,512)
                Channel to apply halo reduction on (only tested with (R-B)/(R+B))
            n_radial_disks : int
                Number of radial disks where 2-means clustering is computed
            n_ang_sections : int
                Number of angular sections where 2-means clustering is computed
            minimize_slope : bool
                Deprecated 

            Returns
            -------
            result : (image, section_averages)
                image : np.ndarray of shape (512,512)
                    Imput image with halo reduced
                section_averages : np.ndarray of shape (n_radial_disks,2,n_ang_sections)
                    Results of 2-means clustering on each section
            """      

            image = np.copy(image)

            bins = np.concatenate([np.linspace(25, 250, n_radial_disks, endpoint=True)]) # np.concatenate([np.linspace(7,90,50, endpoint=True)])

            thetas = np.linspace(-np.pi/2, 3*np.pi/2, n_ang_sections+1)

            current_moys = np.zeros(len(thetas)-1) 
            slopes = np.zeros(len(thetas)-1)

            moys = np.zeros((len(bins)-1, 2, len(thetas)-1))

            for i in range(len(bins)-1):
                new_moys = np.zeros((2,len(thetas)-1))
                computed = np.ones((len(thetas)-1), dtype=bool) 
                masks = np.zeros((len(thetas)-1,)+image.shape, dtype=bool)
                for j in range(len(thetas)-1):
                    masks[j] = (R >= bins[i])&(R < bins[i+1])&(Theta >= thetas[j])&(Theta < thetas[j+1])
                    kmeans = KMeans(2)
                    selection = image[Domain&masks[j]].reshape(-1,1)
                    if len(selection) < 3:
                        computed[j] = False
                        continue
                    if np.all(selection[:,0] == selection[0,0]):
                        new_moys[:,j] = selection[0,0]
                        moys[i,:,j] = selection[0,0]
                        continue
                    kmeans.fit(selection)
                    new_moys[:,j] = kmeans.cluster_centers_.flatten()
                    moys[i,:,j] = kmeans.cluster_centers_.flatten()
                    """
                    if i == 35:
                        plt.hist(image[mask].flatten(), 20)
                        for j in range(2):
                            plt.axvline(kmeans.cluster_centers_[j], c="red")
                        plt.show()
                    """
                
                dist_moys = np.abs(new_moys - current_moys - slopes) # new_moys
                chosen_moys = np.zeros(len(thetas)-1)
                
                for j in range(len(thetas)-1):
                    if dist_moys[0,j] > dist_moys[1,j]:
                        chosen_moys[j] = new_moys[1,j]
                    else:
                        chosen_moys[j] = new_moys[0,j]

                slopes = chosen_moys - current_moys

                if np.any(computed):
                    min_slope = np.min(slopes[computed])
                    max_slope = np.max(slopes[computed])
                    if minimize_slope and i > 0:
                        slopes = np.maximum(np.minimum(min_slope+0.1, slopes),min_slope)
                    else:
                        slopes = np.minimum(np.maximum(max_slope-0.1, slopes),max_slope)
                    chosen_moys = current_moys + slopes

                for j in range(len(thetas)-1):
                    image[masks[j]] -= chosen_moys[j]


            return image, moys


        A_RB_reduced = regularizer(A_RB)[0]
        A_RB_reduced = np.abs(A_RB_reduced)

        if plots:
            plt.subplot(1,2,1).imshow(A_RB)
            plt.subplot(1,2,2).imshow(np.abs(A_RB_reduced))
            plt.show()

        ##################################
        # Thresholding and watershedding #
        ##################################
 
        #We compute the heighmap used for watershedding
        A_Gradient = (gaussian_filter(A[:,:,2], 1.) - gaussian_filter(A[:,:,2], 4.))
        A_Gradient += (gaussian_filter(A[:,:,0], 1.) - gaussian_filter(A[:,:,0], 4.))
        A_Gradient += (gaussian_filter(A[:,:,1], 1.) - gaussian_filter(A[:,:,1], 4.))
        A_Gradient = np.maximum(A_Gradient,-0.05) + 0.05


        markers = np.zeros(A_Gradient.shape, dtype=int)

        # First we mark pixels that are very likely to be clouds
        markers[gaussian_filter(np.maximum(A_RB_reduced,0.2),0.1) > 0.2] = 1
        markers[R < 40] = 0

        # Then we mark pixels that are very likely to be sky
        markers[gaussian_filter(np.maximum(0.03-A_RB_reduced,0.),0.1) > 0.] = 2
        markers[A_RB==0] = 2
        markers[R < 25] = 0

        # Finaly we mark the sun
        markers[sol] = 3

        # The clouds are segmented using watershed technique
        labels = watershed(-A_Gradient, markers, mask=Domain)
        labels[labels > 1] = 0

        if plots:
            fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
            ax = axes.ravel()

            ax[0].imshow(markers, cmap=plt.cm.gray)
            ax[0].set_title('Overlapping objects')
            ax[1].imshow(A_Gradient, cmap=plt.cm.gray)
            ax[1].set_title('Distances')
            ax[2].imshow(labels, cmap=plt.cm.nipy_spectral)
            ax[2].set_title('Separated objects')

            for a in ax:
                a.set_axis_off()

            fig.tight_layout()
            plt.show()

        return labels


if __name__ == "__main__":

    from PIL import Image
        
    initial_crop = 20

    A = np.array(Image.open("PhyDNet-master/data/full_mobotix3/2020-08-06/2020-08-06_15_19.jpg")).astype(np.float32)/225
    A = resize(A[initial_crop:(960-initial_crop),(160+initial_crop):(1120-initial_crop),:], (512,512,3))

    plt.imshow(A[:,:,0]+A[:,:,1]+A[:,:,1])
    plt.show()
    
    R = A[:,:,0]
    B = A[:,:,2]
    A_RB = 2.7*(R-B)/(R+B)
    A_RB[(R+B)==0] = 0
    A_RB += 0.5
    A_RB = np.minimum(np.maximum(A_RB,0),1.)

    plt.imshow(np.minimum(B,0.35))
    plt.show()


    segmenter(A, A_RB, plots=True)

