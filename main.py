import numpy as np
import torch
import pandas as pd
import datetime
import io

from estimator import MLP
from utilities.projections import planar_proj, perisolar_mask, inv_planar_proj
from skimage.transform import resize
from segmentation import segmenter
from pyramid_of import Pyramid_OF
import cv2


import matplotlib.pyplot as plt
from PIL import Image
import PIL.ImageFont as ImageFont
import PIL.ImageDraw as ImageDraw

font_fname = 'utilities/open-sans/OpenSans-Bold.ttf'
font_size = 40
font_size2 = 28
font = ImageFont.truetype(font_fname, font_size)
font2 = ImageFont.truetype(font_fname, font_size2)

import matplotlib.pyplot as plt

def windows_generator(D, w_size, w_step, padding=0., sharpness=2.):
    """
    Generates a list of windows covering a DxD image 

    Parameters
    ----------
    D : int
        Size of the image to be covered with windows
    w_size : int
        Size of the windows
    w_step : int
        Space between the windows
    padding : float, optional
        Size of the added space on the edges of the image, expressed relative to w_size/2
    sharpness : float, optional
        If > 0, the windows generated are gaussian windows of standard deviation w_size/sharpness. If = 0, the windows generated are constant

    Returns
    -------
    result : list of (window_center, window_mask)
        window_center : (i, j) ; i and j of type int
            Coordinates of the center of the window
        window_mask : np.ndarray of shape (D, D) and dtype float
            Mask to apply to image
    """

    if sharpness == 0.:
        base_window = np.ones((w_size,w_size), dtype=np.float32)
    else:
        base_window = np.exp(-np.sum(np.square(np.moveaxis(np.indices((w_size,w_size), dtype=np.float32),0,2)-np.ones(2)*(w_size-1)/2), axis=2)/(2*(w_size/sharpness)**2))

    margin_left = (int(w_size*(1-padding))-1)//2
    margin_right = int(w_size*(1-padding))//2
    res = []
    for i in range(margin_left,D-margin_right,w_step):
        for j in range(margin_left,D-margin_right,w_step):
            i0, j0 = i, j
            window = np.zeros((D,D))
            
            selection = base_window[max(0,-(i0-(w_size-1)//2)):w_size-max(0,i0+w_size//2+1-D),max(0,-(j0-(w_size-1)//2)):w_size-max(0,j0+w_size//2+1-D)]
            window[max(0,i0-(w_size-1)//2):min(D,i0+w_size//2+1),max(0,j0-(w_size-1)//2):min(j0+w_size//2+1,D)] = selection
            res.append(((i,j),window))

    return res

#####################

# the folder contains daily folders (YYYY-MM-DD)
# of hemispherical images (name YYYY-MM-DD_HH_MM | size 1280x960)
folder_raw =  "data/mobotix3_cam"

# Initial projection of raw hemispherical images of size 1280x960 to planar images (see doc) of size 512x512
first_proj = planar_proj(480, 0.1, 512)
# Projection of segmented hemispherical images of size 512x512 to planar images (see doc) of size 512x512
proj_segm = planar_proj(256, 0.1, 512, base_img_center=(256,256))
# Inverse projection of the above projection
inv_proj_segm = inv_planar_proj(256, 0.1, 512, base_img_center=(256,256))

def to_RB(A):
    """
    Converts a RGB image to a mono-channel (R-B)/(R+B) image

    Parameters
    ---------
    A : np.ndarray of shape (D,D,3)
        The image to be converted
    
    Returns
    ---------
    A_RB : np.ndarray of shape (D,D)
        The converted image
    """
    R = A[:,:,0]
    B = A[:,:,2]
    null_mask = (R+B)==0
    A_RB = np.zeros_like(R)
    A_RB[~null_mask] = 2.7*(R-B)[~null_mask]/(R+B)[~null_mask]
    A_RB[null_mask] = 0
    A_RB += 0.5
    A_RB = np.minimum(np.maximum(A_RB,0),1.)
    return A_RB

def shrinken(A_raw, dim, initial_crop = 20):
    """
    Converts raw 1280x960 images into resized dimxdim images, with the 960x960 frame
    tangent to the sky hemisphere having been previously cropped.

    Parameters
    ---------
    A_raw : np.ndarray of shape (960,1280,3)
        The raw image to be converted
    dim : int
        The dimension of the output image
    initial_crop : int
        The crop applied to the frame tangent to the sky hemisphere
    
    Returns
    ---------
    np.ndarray of shape (dim,dim,3)
        The converted image
    """
    A_raw = A_raw[initial_crop:(960-initial_crop),(160+initial_crop):(1120-initial_crop),:]
    return resize(A_raw, (dim, dim, 3), anti_aliasing=True)

# We compute the so-called "base_mask" hiding backgroud elements (roofs, chimneys, cranes, poles ...)
img_mask = np.array(Image.open("data/background_mask.png"))/255
base_mask = (img_mask[:,:,0] != 0.)&(img_mask[:,:,1] != 1.)&(img_mask[:,:,2] != 0.)

#####################


D = 512
w_size = 96

# We compute all the (overlapping) windows where OF will be calculated
windows = windows_generator(D,96,32,0.,3.)

# The weight of a window represents the confidence in the last optical flow calculated on the window in question
windows_weights = np.zeros(len(windows))

"""# We compute the projection of the windows in the zenith plane
windows_proj_viz = []
for (i, j), window in windows:
    windows_proj_viz.append(((proj_viz[0][i,j], proj_viz[1][i,j]), window[proj_inv_viz]))"""

pyramid = Pyramid_OF(D, levels=2)

# Duration of the prediction phase
T = 30

cache_images_flown = np.zeros((T+1,D,D,3))
cache_images_flown_RB = np.zeros((T+1,D,D))

# Blue sky model determined progressively during estimation steps by cloud segmentation
blue_sky = np.zeros((D,D,3))
# Mask describing confidence on current blue sky computation for each pixel
bs_weight = np.zeros((D,D))

image_base = None
i = 0

# Array containing results of windowed optical flow computations - format (dy, dx)
thetas = np.zeros((781,len(windows),2))
# Array containing median over all windows of optical flow computations
global_thetas = np.zeros((781,2))


########################

#M = int(input("Starting month ? "))
#D = int(input("Starting day ? "))


# Available test instance
M = 8
D = 12
h = int(input("Starting hour ? "))
m = int(input("Starting min ? "))


# A negative value of - n means that the prediction will start after n calibration steps (n minutes)
pred_t = -8

start_time = datetime.datetime(year=2020,month=M, day=D, hour=h, minute=m)
end_time = datetime.datetime(year=2020,month=8,day=31, hour=19, minute=0)

##########################

final_df = pd.read_csv("data/data_06_2020_08_2020.csv")
final_df = final_df.set_index("Time")

device = torch.device("cpu")

# CNN model used for solar irradiance estimation
cnn = MLP(64,3,1).to(device)
cnn_state_dict = torch.load('data/cnn.pth', map_location = device)
cnn.load_state_dict(cnn_state_dict)

true_SIs = []
estimated_SIs = []
predicted_SIs = []

#######################

day_range = pd.date_range(start=start_time, end=end_time, freq="min").strftime("%Y-%m-%d").values
min_range = pd.date_range(start=start_time, end=end_time, freq="min").strftime("%Y-%m-%d_%H_%M.jpg").values

for day, img in zip(day_range, min_range):

    print(img)

    # Preprossessing on raw (960,1280) hemispherical sky images
    image_raw = np.array(Image.open(folder_raw + "/" + day + "/"+img))/255
    image_OF2 = image_raw[first_proj]
    image_OF = to_RB(image_OF2)

    # We compute the theoretical position of the sun and of the mask used to hide it from OF algorithm
    W_image_OF, (i_sol, j_sol) = perisolar_mask(time = img[:-4], D = 512, r_mask=64, df=final_df)
    W_image_OF = W_image_OF[proj_segm]
    i_sol, j_sol = inv_proj_segm[0][i_sol, j_sol], inv_proj_segm[1][i_sol, j_sol]

    #################################################################
    # COMPUTATION OF OPTICAL FLOW (Stopped during prediction phase) #
    #################################################################
    if pred_t <= 0:

        # Combination of the solar mask and the background elements mask
        W_image_OF = W_image_OF.astype(np.float32)
        W_image_OF = W_image_OF*base_mask

        """plt.subplot(1,2,1).imshow(image_OF2)
        plt.subplot(1,2,2).imshow(np.einsum("...k,...",image_OF2,W_image_OF))
        plt.show()"""

        pyramid.compute_resized(image_OF, W_image_OF)
        pyramid.compute_gradient()

        if image_base is not None:

            # Update of windowed OF results
            thetas[i], new_weights = pyramid.compute_windowed_OF(windows)
            thetas[i] = np.einsum("...k,...",thetas[i-1],windows_weights) + np.einsum("...k,...",thetas[i],new_weights)
            null_mask = (windows_weights + new_weights) == 0
            thetas[i][~ null_mask] = np.einsum("...k,...",thetas[i][~ null_mask],1/(windows_weights + new_weights)[~ null_mask])
            thetas[i][null_mask] = 0.

            # Update of the confidence weights of windowed OF results
            windows_weights += new_weights/6
            windows_weights /= 7/6

        pyramid.step()

        # As an indication, we evaluate a "median" optical flow on the whole projection
        global_thetas[i] = np.median(thetas[i], axis=0)
        print("Global cloud motion :", global_thetas[i])

    ##################################################################
    # SEGMENTATION OF THE SKY IMAGE (Unused during prediction phase) #
    ##################################################################

    if pred_t <= 0:

        # The segmentation is performed on hemispherical images of size (512,512)
        image_512 = shrinken(image_raw, 512, initial_crop = 0)
        imagsegm_hemi = segmenter(image_512, to_RB(image_512))
        # Its result is then projected in the zenith plane, where most of the computations of the model are performed
        imagsegm = imagsegm_hemi[proj_segm]

        sky = (imagsegm == 0)

        # The blue sky model is updated with sky given by last segmentation results
        blue_sky[sky] = (np.einsum('...,...k',(1-bs_weight),image_OF2) + np.einsum('...,...k',bs_weight,blue_sky))[sky]
        bs_weight += (1-imagsegm)*(1-bs_weight)/4
        bs_weight *= 3/4

        print("Blue-sky completion rate :", np.mean(bs_weight))

    ############################################################################
    # COMPUTATION OF THE FINAL BLUE SKY MODEL (At the start of the prediction) #
    ############################################################################

    if pred_t == 0:

        print("START OF PREDICTION")

        # We fill the holes that remain in the last iterated blue sky
        last_blue_sky = np.copy(blue_sky)
        lbs_null = (last_blue_sky[:,:,0] == 0)&(last_blue_sky[:,:,1] == 0)&(last_blue_sky[:,:,2] == 0)
        mean_lbs = np.mean(last_blue_sky[(np.sum(last_blue_sky, axis=2) < 0.9)&(~lbs_null)], axis = 0)
        last_blue_sky[lbs_null] = mean_lbs

        # If the sun is not present in the last iterated blue sky (it has remained hidden), we manually add it
        i0_sol, j0_sol = i_sol, j_sol
        if np.sum(np.sum(last_blue_sky, axis=2) > 1.2) < 40:
            print("Sun manually added")
            sun = np.array(Image.open("data/generic_sun.png"))/255
            last_blue_sky[i_sol-48:i_sol+48, j_sol-48:j_sol+48] = sun


        #lbs_blurred = gaussian(last_blue_sky, sigma=10.)
        #last_blue_sky[np.sum(lbs_blurred, axis=2) <0.7] = lbs_blurred[np.sum(lbs_blurred, axis=2) <0.7]

        cache_images_flown[0] = np.copy(image_OF2)
        cache_images_flown_RB[0] = imagsegm

    ####################################################
    # PROPAGATION OF THE SKY IMAGE (During prediction) #
    ####################################################

    if pred_t > 0:

        # RGB image to propagate
        current_img = cache_images_flown[pred_t%(T+1)-1]
        # Corresponding cloud mask to propagate
        current_RB = cache_images_flown_RB[pred_t%(T+1)-1]

        # Propagation is done by adding every shifted window of "current_img/current_RB" to "new_img/new_RB"
        new_img = np.zeros_like(current_img)
        new_RB = np.zeros_like(current_RB)
        # Since shifted windows overlap, we need to normalize the propagated images "new_img" and "new_RB"
        depth = np.zeros_like(current_RB)

        for k, (center, window) in enumerate(windows):

            # The motion vector used to propagate the clouds is the last optical flow computed on the window
            # before the prediction phase
            dx, dy = int(np.round(thetas[i-pred_t%(T+1),k,0])), int(np.round(thetas[i-pred_t%(T+1),k,1]))
  
            new_RB += np.roll(current_RB*window, (dx, dy), axis=(0,1))
            new_img += np.roll(np.einsum("ij...,ij->ij...", current_img, window),(dx, dy), axis=(0,1))
            depth += np.roll(window,(dx, dy), axis=(0,1))

        # The propagated images are normalized
        null_depth = (depth == 0)
        new_img[~null_depth] = np.einsum("ij,i->ij",new_img[~null_depth],1/depth[~null_depth])
        new_RB[~null_depth] = new_RB[~null_depth]/depth[~null_depth]

        """plt.subplot(1,3,1).imshow(new_img)
        plt.subplot(1,3,1).imshow(new_RB < 0.1)
        plt.subplot(1,3,3).imshow(new_RB)
        plt.show()"""

        # The blue sky model is translated to adjust for the movement of the sun
        translated_lbs = np.roll(last_blue_sky, (i_sol-i0_sol, j_sol-j0_sol), axis=(0,1))

        """plt.subplot(1,3,1).imshow(new_img)
        plt.subplot(1,3,3).imshow(new_RB)
        plt.show()"""

        # The part of new image that is not overlapped by cloud mask is refreshed with blue sky model
        new_img[(new_RB < 0.3)] = translated_lbs[(new_RB < 0.3)]       
        
        cache_images_flown[pred_t%(T+1)] = new_img
        cache_images_flown_RB[pred_t%(T+1)] = new_RB

        # If the cloud cover is not thick enough, the sun can break through it
        new_img[(new_RB < 0.7)&(np.sum(translated_lbs, axis=2) > 1.2)] = translated_lbs[(new_RB < 0.7)&(np.sum(translated_lbs, axis=2) > 1.2)]

        # We save the propagated cloudmask at each step of the prediction
        new_RB_image = Image.fromarray((new_RB*255).astype(np.uint8))
        new_RB_image.save("results/cloudmask_propagated/" + img[:-4]+".png")

    ##################################
    # ESTIMATION OF SOLAR IRRADIANCE #
    ##################################

    # Real value
    true_SIs.append(final_df.loc[img[:-4], "Irradiance"])

    # Value given by the estimator on REAL sky image
    image_cnn = resize(np.moveaxis(image_OF2,2,0)[None,:,128:384,128:384], (1,3,64,64))
    estimated_SIs.append(cnn(torch.FloatTensor(image_cnn)).detach().numpy()[0,0]*100)

    # Value given by the estimator on PREDICTED sky image (only if t >= 0)
    if pred_t == 0:
        predicted_SIs.append(estimated_SIs[-1])
    elif pred_t > 0:
        image_cnn_flowed = resize(np.moveaxis(new_img,2,0)[None,:,128:384,128:384], (1,3,64,64))
        predicted_SIs.append(cnn(torch.FloatTensor(image_cnn_flowed)).detach().numpy()[0,0]*100)

    ############################
    # VISUALIZATION OF RESULTS #
    ############################

    # We save the blue sky model iterations

    if pred_t <= 0:
        bs_img = Image.fromarray((blue_sky*255).astype(np.uint8))
        bs_img.save("results/bluesky_model/" + img[:-4] + ".png")

    if pred_t == 0:
        bs_img = Image.fromarray((last_blue_sky*255).astype(np.uint8))
        bs_img.save("results/bluesky_model/last_" + img[:-4] + ".png")    

    ############################

    # We build a basic visualization of the results of the windowed calculation of the optical flow,
    # superimposed as white directional lines on the projected image

    if pred_t > 0:
        image_flow = np.copy(new_img)
    else:
        image_flow = np.copy(image_OF2)
    for k, (center, _) in enumerate(windows):
        if pred_t > 0:
            theta = thetas[i - pred_t,k]
        else:
            theta = thetas[i,k]     
        for t in np.linspace(0,1,100):
            P = (t*theta + np.array(center, dtype=np.float64)).astype(int)
            image_flow[P[0],P[1]] = np.array([1.,0,0])

    Image.fromarray((image_flow*255).astype(np.uint8)).save("results/optical_flow/" + img[:-4] + ".png")

    ############################

    # We then build a visualization of the results of irradiance prediction, compared to true irradiance value,
    # and value obtained by estimation on the true sky images

    new_img_RB = Image.fromarray((to_RB(image_OF2[128:384,128:384,:])*255).astype(np.uint8))
    img_flow_RB = Image.fromarray((to_RB(image_flow[128:384,128:384,:])*255).astype(np.uint8))

    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True

    plt.figure()

    plt.plot(range(0,i+1), true_SIs, label="real_SI", marker="o", c="orange")
    plt.plot(range(0,i+1), estimated_SIs, label="estimation", marker="o", linestyle="dotted", c="blue")
    plt.plot(range(i-pred_t,i+1),predicted_SIs, label="prediction with OF", marker="o", c="blue")
    if pred_t > 0:
        plt.axvline(i-pred_t+1/2)
    plt.legend()

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')

    plt.close()

    graph = Image.open(img_buf)
    graph = graph.resize((288,162),Image.ANTIALIAS)

    big_img = Image.new('RGB', (1024, 512))
    big_img.paste(new_img_RB.resize((512, 512),Image.ANTIALIAS), (0, 0))
    big_img.paste(img_flow_RB.resize((512, 512),Image.ANTIALIAS), (512, 0))
    big_img.paste(graph, (368, 340))
    draw = ImageDraw.Draw(big_img)
    draw.text((395, 12), img[:-4], font=font2, fill='rgb(255, 255, 255)')
    if pred_t == 0:
        draw.text((300, 220), "Start of prediction", font=font, fill='rgb(255, 255, 255)')
    big_img.save("results/prediction/" + img[:-4] + ".png")

    ####################################

    image_base = image_OF
    image_base2 = image_OF2

    i += 1
    pred_t += 1

    if pred_t > T:
        break