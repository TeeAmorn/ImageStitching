# Import libraries
import cv2 as cv
import numpy as np
import os
from itertools import product

# =============== Load Images ===============

def load_images(file_path):
    """
    Load all images in file_path. The images must be sorted and numbered 000,
    001, 002, etc. 

    Parameters
    ----------
    (str) file_path : Path to the directory in which the images are stored

    Returns
    -------
    (list) images : List of images (stored as np.array) inside the directory
    """
    images = []
    for filename in sorted(os.listdir(file_path)):
        if (filename.endswith(".jpg") or filename.endswith(".png") \
            or filename.endswith(".bmp") or filename.endswith(".jpeg")):
            image = cv.imread(os.path.join(file_path, filename))
            images.append(image)
    return images

# =============== Keypoints Detection ===============

def detect_keypoints(orig_img, warp_img):
    """
    Detect matching keypoints between orig_img and warp_img

    Parameters
    ----------
    (np.array) orig_img : The base persepctive image
    (np.array) warp_img : The image whose persepctive has to be converted

    Returns
    -------
    (np.array) matched_points : 3D array of size (n x 2 x 2) containing
        the posiitons of the matching points
    """

    # Initialize sift object
    sift = cv.SIFT_create()

    # Convert images to grayscale
    gray_orig = cv.cvtColor(orig_img, cv.COLOR_BGR2GRAY)
    gray_warp = cv.cvtColor(warp_img, cv.COLOR_BGR2GRAY)

    # Find the keypoints and descriptors with SIFT
    kp_orig, des_orig = sift.detectAndCompute(gray_orig, None)
    kp_warp, des_warp = sift.detectAndCompute(gray_warp, None)

    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des_orig, des_warp, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.3*n.distance:
            good.append([m])

    # Obtain positions of the matched points
    matched_points = []
    for m in good:
        x_orig, y_orig = kp_orig[m[0].queryIdx].pt
        x_warp, y_warp = kp_warp[m[0].trainIdx].pt
        matched_points.append(np.array([[round(y_orig), round(y_warp)], 
                                        [round(x_orig), round(x_warp)]]))

    # ================ Draw images with the matching points ================
    # ======================== (for report purpose) ========================

    # Draw matching lines onto the colored images
    matched_img = cv.drawMatchesKnn(orig_img, kp_orig, 
                                    warp_img, kp_warp,
                                    good, None, 
                                    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                                    
    # Draw detected matching points on each image individually
    curr = np.copy(orig_img)
    curr2 = np.copy(warp_img)
    for m in matched_points:
        xo, yo = m[1][0], m[0][0]
        xw, yw = m[1][1], m[0][1]
        curr = cv.circle(curr, (xo, yo), 5, (255, 0, 0), 2)
        curr2 = cv.circle(curr2, (xw, yw), 5, (255, 0, 0), 2)
    orig_img_matched, warped_img_matched = curr, curr2

    return matched_img, orig_img_matched, warped_img_matched

    # ======================================================================
    # ======================================================================

    return np.array(matched_points)

# ================== Visualize the matched keypoints ==================
# ======================= (for report purpose) ========================
orig_img = cv.imread("results_7/crop_mat_2.jpg")
warp_img = cv.imread("test_images_7/004.jpeg")
img, img2, img3 = detect_keypoints(orig_img, warp_img)
# cv.imwrite("results_3/matched_keypoints.png", img)
cv.imshow("img", img)
cv.waitKey(0)
cv.imshow("img2", img2)
cv.waitKey(0)
cv.imshow("img3", img3)
cv.waitKey(0)
cv.destroyAllWindows()
# =====================================================================
# =====================================================================


# =============== Homography Matrix ===============

def compute_Hmatrix(keypoints):
    """
    Compute the homography matrix given the keypoints in orig_img and warp_img

    Parameters
    ----------
    (np.array) orig_img : The image whose perspective is treated as the base
    (np.array) warp_img : The image whose perspective must be changed to match
        that of the orig_img
    (np.array) keypoints: (n x 2 x 2) array containing n pairs of corresponding
        points. For example, the subarray [0, :, :] contains 
        [[ y , y' ]    - [0, 0, :] : y coordinate of keypoint (y' for warp_img)
         [ x , x' ]]   - [0, 1, :] : x coordinate of keypoint (x' for warp_img)

    Returns
    -------
    (np.array) Hmatrix : The homography matrix which is a (3 x 3) array
    """
    
    # Populate the b vector, which are the (x'-x) and (y'-y)
    # [x' - x]
    # [y' - y]
    # Populate the A matrix, which are given by
    # [x  y  1  0  0  0  -x'x  -x'y]
    # [0  0  0  x  y  1  -y'x  -y'y]
    A = []
    b = []
    curr_row = 0
    while (curr_row < keypoints.shape[0]):
        curr_xy = keypoints[curr_row, :, :]
        curr_b = np.array([[curr_xy[1, 1]-curr_xy[1, 0]],
                           [curr_xy[0, 1]-curr_xy[0, 0]]])
        curr_A = np.array(([[curr_xy[1, 0], curr_xy[0, 0], 1, 0, 0, 0,
                             -curr_xy[1, 1]*curr_xy[1, 0], 
                             -curr_xy[1, 1]*curr_xy[0, 0]],
                            [0, 0, 0, curr_xy[1, 0], curr_xy[0, 0], 1,
                             -curr_xy[0, 1]*curr_xy[1, 0], 
                             -curr_xy[0, 1]*curr_xy[0, 0]]]))
        b.append(curr_b)
        A.append(curr_A)
        curr_row += 1
    b = np.array(b).reshape(-1, 1)
    A = np.array(A).reshape((-1, 8))
    
    # Compute the homography matrix
    h = np.matmul(np.linalg.inv(np.matmul(A.T, A)), np.matmul(A.T, b))[:,0]
    Hmatrix = np.array([[1+h[0], h[1], h[2]],
                        [h[3], 1+h[4], h[5]],
                        [h[6], h[7], 1]])
    return Hmatrix

# =============== Image Warping ===============

def warp_image(orig_img, warp_img, Hmatrix):
    """
    Warp warp_img, which is changing the perspective of the warp_img from its
    current perspective to that of the base perspective (this information is
    stored within the homography matrix Hmatrix)

    Parameters
    ----------
    (np.array) orig_img: the origin image with the preserved perspective
    (np.array) warp_img : The image to be warped
    (np.array) Hmatrix : The homography matrix to use in warping the image

    Returns
    -------
    (np.array) orig_mat : The original image in the background frame
    (np.array) warp_mat : The warped image in the background frame
    (np.array) both_mat : The overlapped image (without proper blending)
    (string) pos_orig : Position of the original image relative to the
        warped image
    (np.array) translated_warp_vertices: four vertices of warpped image 
        after translation, in the order (upper left, upper right,
        lower left,lower right)
    (np.array) translated_orig_vertices: four vertices of original image 
        after translation, same order as translated_warp_vertices
    """

    # Find the warp_img vertices in the base perspective
    warp_img_height, warp_img_width = warp_img.shape[0], warp_img.shape[1]
    orig_img_height, orig_img_width = orig_img.shape[0], orig_img.shape[1]
    
    # Store vertices in the warp/wrong perspective in warp_vertices
    warp_vertices = np.array([[0, 0],
                              [0, warp_img_width-1],
                              [warp_img_height-1, 0],
                              [warp_img_height-1, warp_img_width-1]]) 
    
    # Compute warpped vertices in the base perspective 
    # and store them in base_vertices
    base_vertices = []
    for v in warp_vertices:
        v_tmp = np.array([[v[1]], [v[0]], [1]]) 
        base_v = np.matmul(np.linalg.inv(Hmatrix), v_tmp)[:,0]
        base_vertices.append(np.array([base_v[1], base_v[0]])/base_v[2]) 
    base_vertices = np.array(base_vertices) 
    
    # Find the size of background mat initializations
    mat_height = orig_img_height
    mat_width = orig_img_width
    row_translate = 0
    col_translate = 0
    orig_mat = [] 
    warp_mat = [] 
    # Find out whether original image is on the right or left to warpped image
    pos_orig = 'left' 

    if np.argwhere(base_vertices<0).shape[0] == 0: # no negative verticies 
        row_translate = 0
        col_translate = 0

    # Only has negative row values but no negative col values (origin at left 
    # most corner of images)
    elif np.argwhere(base_vertices[:,0]<0).shape[0] != 0 and \
        np.argwhere(base_vertices[:,1]<0).shape[0] == 0:
        row_translate = abs(int(round(min(min(base_vertices[:,0]),0))))
        col_translate = 0

    # Only has negative col values but no negative row values
    elif np.argwhere(base_vertices[:,0]<0).shape[0] == 0 and \
        np.argwhere(base_vertices[:,1]<0).shape[0] != 0:
        row_translate = 0
        col_translate = abs(int(round(min(min(base_vertices[:,1]),0))))
        pos_orig = 'right'

    # Have both negative col and row values
    else:
        row_translate = abs(int(round(min(min(base_vertices[:,0]),0))))
        col_translate = abs(int(round(min(min(base_vertices[:,1]),0))))
        pos_orig = 'right'

    # Calculate the size of matrix
    mat_height = int(round(max(max(base_vertices[:,0]),orig_img_height))) \
        + row_translate + 1
    mat_width = int(round(max(max(base_vertices[:,1]),orig_img_width))) \
        + col_translate + 1

    # Store the original image with the background mat
    orig_mat = np.zeros((mat_height, mat_width,3))
    orig_mat[row_translate:orig_img_height+row_translate, 
             col_translate:orig_img_width+col_translate,:] += orig_img
    
    warp_mat = np.zeros((mat_height, mat_width,3)) 
    
    # Translate warp_img vertices (in the base-perspective) so that the 
    # top-left corner of warp_img is at the origin
    translate_amount = np.array([row_translate,col_translate])

    # Start filling in the color density of the warped image
    for (y, x) in product(range(mat_height), range(mat_width)):
        base_xy = np.array([[x-translate_amount[1]],
                            [y-translate_amount[0]],
                            [1]])
        warp_xy = np.matmul(Hmatrix, base_xy)[:,0]
        warp_y = int(round(warp_xy[1]/warp_xy[2]))
        warp_x = int(round(warp_xy[0]/warp_xy[2]))

        if (warp_y in range(warp_img.shape[0]) and \
            warp_x in range(warp_img.shape[1])):
            warp_mat[y, x, :] = warp_img[warp_y, warp_x, :]

    # Put both images into the same matrix
    both_mat = np.zeros((mat_height, mat_width,3)) 
    for r in range(mat_height):
        for c in range(mat_width):
            if np.any(warp_mat[r,c,:]!=0):
                both_mat[r,c,:] = warp_mat[r,c,:]
            if np.any(orig_mat[r,c,:]!=0):
                both_mat[r,c,:] = orig_mat[r,c,:]
    
    translated_orig_vertices = np.array([[0,0],[0,orig_img_width-1],
                                         [orig_img_height-1,0],
                                         [orig_img_height-1, 
                                          orig_img_width-1]]) + \
                                              translate_amount
    translated_warp_vertices = base_vertices + translate_amount

    orig_mat = orig_mat.astype(np.uint8)
    warp_mat = warp_mat.astype(np.uint8)
    both_mat = both_mat.astype(np.uint8)
    return orig_mat, warp_mat, both_mat, pos_orig, \
        translated_warp_vertices,translated_orig_vertices

#    # =============== Translated version of the warped image ===============
#    # ======================== (for report purpose) ========================
#
#    # Translate warp_img vertices (in the base-perspective) so that the 
#    # top-left corner of warp_img is at the origin
#    translate_amount = np.amin(base_vertices, axis=0)
#    translated_base_vertices = base_vertices - translate_amount
#
#    # Create warped_image object to store the warp_img after warped
#    warped_image_shape = np.ceil(np.amax(translated_base_vertices, 
#                                 axis=0)).astype(np.uint32)
#    warped_image = np.zeros((warped_image_shape[0], warped_image_shape[1], 3))
#
#    # Start filling in the color density of the warped image
#    height, width = warped_image_shape
#    for (y, x) in product(range(height), range(width)):
#        base_xy = np.array([[x+translate_amount[1]],
#                            [y+translate_amount[0]],
#                            [1]])
#        warp_xy = np.matmul(Hmatrix, base_xy)[:,0]
#        warp_y = round(warp_xy[1]/warp_xy[2])
#        warp_x = round(warp_xy[0]/warp_xy[2])
#
#        if (warp_y in range(warp_img.shape[0]) and \
#            warp_x in range(warp_img.shape[1])):
#            warped_image[y, x, :] = warp_img[warp_y, warp_x, :]
#
#    return warped_image.astype(np.uint8)
#    
#    # ======================================================================
#    # ======================================================================

# =============== Image Blending ===============

def blend_image(orig_mat, warp_mat, both_mat, pos_orig):
    """
    Blend two images, orig_mat and warp_mat, together. The position of the two
    images are determined by pos_orig.

    Parameters
    ----------
    (np.array) orig_mat : The original image in the background frame
    (np.array) warp_mat : The warped image in the background frame
    (np.array) both_mat : The overlapped image (without proper blending)
    (string) pos_orig : Position of the original image relative to the
        warped image

    Returns
    -------
    (np.array) blend_mat : The final blended image
    """
    blend_mat = np.zeros((orig_mat.shape))
    mat_height = orig_mat.shape[0]
    mat_width = orig_mat.shape[1]
    # Store all overlap vertex indices as dictionary 
    # (key = row number, value = list of col numbers)
    overlap = {} 
    for r in range(mat_height):
        for c in range(mat_width):
            if np.any(warp_mat[r,c,:]!=0) and np.any(orig_mat[r,c,:]!=0):
                if r in overlap.keys():
                    overlap[r].append(c)
                else:
                    overlap[r] = [c]
    boundary = {}
    for k in overlap.keys():
        val = overlap[k]
        boundary[k] = [min(val),max(val)]
    
    for r in range(mat_height):
        for c in range(mat_width):
            if np.any(warp_mat[r,c,:]!=0):
                blend_mat[r,c,:] = warp_mat[r,c,:]
            if np.any(orig_mat[r,c,:]!=0):
                blend_mat[r,c,:] = orig_mat[r,c,:]
            if np.any(warp_mat[r,c,:]!=0) and np.any(orig_mat[r,c,:]!=0):
                total_weight = boundary[r][1] - boundary[r][0]
                if total_weight == 0:
                    orig_weight = 0.5
                    warp_weight = 0.5
                elif pos_orig == 'left':
                    orig_weight =  (boundary[r][1]-c)/total_weight
                    warp_weight = (c-boundary[r][0])/total_weight
                else:
                    orig_weight = (c-boundary[r][0])/total_weight
                    warp_weight = (boundary[r][1]-c)/total_weight
                blend_mat[r,c,:] = orig_weight * orig_mat[r,c,:] + \
                    warp_weight * warp_mat[r,c,:]
    return blend_mat.astype(np.uint8)

# =============== Image Cropping ===============

def crop(translated_warp_vertices, translated_orig_vertices, blend_mat, pos_orig):
    """
    Crop the blended image to make it a rectangular-shaped image
    
    Parameters
    ----------
    (np.array) translated_warp_vertices: four vertices of warpped image 
                    after translation, in the order (upper left, upper right, 
                    lower left,lower right)
    (np.array) translated_orig_vertices: four vertices of original image 
                    after translation, same order as translated_warp_vertices
    (np.image) blend_mat : The blended image without cropping
    
    Returns
    -------
    (np.image) crop_mat: The cropped image 
    """
    # Find max of the upper 4 coordinates, min of the lower 4
    upper4row = np.concatenate((translated_warp_vertices[:2,0],
                                translated_orig_vertices[:2,0]),axis=0)
    row_up = int(round(np.amax(upper4row,0)))
    lower4row = np.concatenate((translated_warp_vertices[2:,0],
                                translated_orig_vertices[2:,0]),axis=0)
    row_low = int(round(np.amin(lower4row,0)))
    
    # Depending on where origin image is located, derive the column 
    # value closest to center
    if pos_orig =='left':
        col_left = 0
        col_right = int(round(np.amin(translated_warp_vertices[[1,3],1])))
    else: # pos_orig = right
        col_left = int(round(np.amax(translated_warp_vertices[[0,2],1])))
        col_right = translated_orig_vertices[1,1]
        
    # generating the cropped mat
    crop_height = row_low-row_up
    crop_width = col_right-col_left
    crop_mat = np.zeros((crop_height,crop_width,3))
    for i in range(crop_height):
        for j in range(crop_width):
            crop_mat[i,j,:] = blend_mat[i+row_up,j+col_left,:]
    return crop_mat.astype(np.uint8)

# =============== Stitch ===============

def stitch(file_path, center_idx):
    """
    Stitch all images in "file_path". The perspective of all images will be 
    projected to that of the image whose number is center_idx.

    Parameters
    ----------
    (str) file_path: Path to the directory in which the images are stored
    (int) center_idx: The image number (as written in the image's filename)
    
    Returns
    -------
    (np.array) stitched_image: Final stitched image
    """
    
    # Load all images to be stitched
    images = load_images(file_path)

    # Return None or original image if images only has one image (or none)
    if (len(images) == 0):
        return None
    if (len(images) == 1):
        return images[0]

    # Start processing the images iteratively, starting from the center image
    curr_img = images[center_idx]   # Stores the current blended image
    warp_idx = center_idx+1         # Process from center and out to the right
    while (warp_idx < len(images)): 
        warp_img = images[warp_idx]
        keypoints = detect_keypoints(curr_img, warp_img)
        Hmatrix = compute_Hmatrix(keypoints)
        orig_mat, warp_mat, both_mat, pos_orig, translated_warp_vertices, \
            translated_orig_vertices = warp_image(curr_img, warp_img, Hmatrix)
        blend_mat = blend_image(orig_mat, warp_mat, both_mat, pos_orig)
        crop_mat = crop(translated_warp_vertices, translated_orig_vertices, \
            blend_mat, pos_orig)
        curr_img = crop_mat
        warp_idx += 1

    warp_idx = center_idx-1         # Process from center and out to the left
    while (warp_idx >= 0):
        warp_img = images[warp_idx]
        keypoints = detect_keypoints(curr_img, warp_img)
        Hmatrix = compute_Hmatrix(keypoints)
        orig_mat, warp_mat, both_mat, pos_orig, translated_warp_vertices, \
            translated_orig_vertices = warp_image(curr_img, warp_img, Hmatrix)
        blend_mat = blend_image(orig_mat, warp_mat, both_mat, pos_orig)
        crop_mat = crop(translated_warp_vertices, translated_orig_vertices, \
            blend_mat, pos_orig)
        curr_img = crop_mat
        warp_idx -= 1

    return curr_img

# =============== Test Stitch ===============

# final_img = stitch("test_images_7", 1)
# cv.imwrite("final_img_7.png", final_img)

# =============== Test 1 - Nichada ===============

# orig_img = cv.imread("test_images/001.jpg")
# warp_img = cv.imread("test_images/002.jpg")

# # keypoints = np.array([[[169, 184], [501, 71]],
# #                       [[325, 342], [588, 161]],
# #                       [[542, 548], [569, 138]],
# #                       [[467, 485], [476, 53]]])

# keypoints = detect_keypoints(orig_img, warp_img)
# Hmatrix = compute_Hmatrix(keypoints)

# keypoints = detect_keypoints(orig_img, warp_img)
# Hmatrix = compute_Hmatrix(keypoints)

# orig_mat, warp_mat, both_mat, pos_orig, translated_warp_vertices, \
#     translated_orig_vertices = warp_image(orig_img, warp_img, Hmatrix)
# blend_mat = blend_image(orig_mat, warp_mat, both_mat, pos_orig)
# crop_mat = crop(translated_warp_vertices, translated_orig_vertices, blend_mat)

# cv.imshow('orig_mat.jpg', orig_mat)
# cv.waitKey(0)
# cv.imshow('warp_mat.jpg', warp_mat)
# cv.waitKey(0)
# cv.imshow('both_mat.jpg', both_mat)
# cv.waitKey(0)
# cv.imshow('blend_mat.jpg', blend_mat)
# cv.waitKey(0)
# cv.imshow('crop_mat.jpg', crop_mat)
# cv.waitKey(0)

# cv.destroyAllWindows()

# =============== Test 2 - Mountains ===============

# orig_img = cv.imread("test_images_2/001.png")
# warp_img = cv.imread("test_images_2/002.png")

# keypoints = detect_keypoints(orig_img, warp_img)
# Hmatrix = compute_Hmatrix(keypoints)

# orig_mat, warp_mat, both_mat, pos_orig, translated_warp_vertices, \
#     translated_orig_vertices = warp_image(orig_img, warp_img, Hmatrix)
# blend_mat = blend_image(orig_mat, warp_mat, both_mat, pos_orig)
# crop_mat = crop(translated_warp_vertices, translated_orig_vertices, blend_mat)

# cv.imshow('orig_mat.jpg', orig_mat)
# cv.waitKey(0)
# cv.imshow('warp_mat.jpg', warp_mat)
# cv.waitKey(0)
# cv.imshow('both_mat.jpg', both_mat)
# cv.waitKey(0)
# cv.imshow('blend_mat.jpg', blend_mat)
# cv.waitKey(0)
# cv.imshow('crop_mat.jpg', crop_mat)
# cv.waitKey(0)

# cv.destroyAllWindows()

# =============== Test 3 - Lake ===============

# orig_img = cv.imread("test_images_3/002.jpg")
# warp_img = cv.imread("test_images_3/001.jpg")

# keypoints = detect_keypoints(orig_img, warp_img)
# Hmatrix = compute_Hmatrix(keypoints)

# orig_mat, warp_mat, both_mat, pos_orig, translated_warp_vertices, \
#     translated_orig_vertices = warp_image(orig_img, warp_img, Hmatrix)
# blend_mat = blend_image(orig_mat, warp_mat, both_mat, pos_orig)
# crop_mat = crop(translated_warp_vertices, translated_orig_vertices, blend_mat, pos_orig)

# cv.imwrite(os.path.join("results_3", '2_orig_mat.jpg'), orig_mat)
# cv.imwrite(os.path.join("results_3", '2_warp_mat.jpg'), warp_mat)
# cv.imwrite(os.path.join("results_3", '2_both_mat.jpg'), both_mat)
# cv.imwrite(os.path.join("results_3", '2_blend_mat.jpg'), blend_mat)
# cv.imwrite(os.path.join("results_3", '2_crop_mat.jpg'), crop_mat)

# cv.imshow('orig_mat.jpg', orig_mat)
# cv.waitKey(0)
# cv.imshow('warp_mat.jpg', warp_mat)
# cv.waitKey(0)
# cv.imshow('both_mat.jpg', both_mat)
# cv.waitKey(0)
# cv.imshow('blend_mat.jpg', blend_mat)
# cv.waitKey(0)
# cv.imshow('crop_mat.jpg', crop_mat)
# cv.waitKey(0)

# cv.destroyAllWindows()

# =============== Test 4 - Mountains ===============

# orig_img = cv.imread("results_7/crop_mat_1.jpg")
# warp_img = cv.imread("test_images_7/003.jpeg")

# keypoints = detect_keypoints(orig_img, warp_img)
# Hmatrix = compute_Hmatrix(keypoints)

# orig_mat, warp_mat, both_mat, pos_orig, translated_warp_vertices, \
#     translated_orig_vertices = warp_image(orig_img, warp_img, Hmatrix)
# cv.imwrite('results_7/orig_mat_2.jpg', orig_mat)
# cv.imwrite('results_7/warp_mat_2.jpg', warp_mat)
# cv.imwrite('results_7/both_mat_2.jpg', both_mat)
# blend_mat = blend_image(orig_mat, warp_mat, both_mat, pos_orig)
# cv.imwrite('results_7/blend_mat_2.jpg', blend_mat)
# crop_mat = crop(translated_warp_vertices, translated_orig_vertices, blend_mat, pos_orig)
# cv.imwrite('results_7/crop_mat_2.jpg', crop_mat)

# =============== See if homography matrix is accurate ===============

# import math

# orig_img = cv.imread("test_images_2/001.jpg")
# warp_img = cv.imread("test_images_2/002.jpg")
# keypoints = detect_keypoints(orig_img, warp_img)
# Hmatrix = compute_Hmatrix(keypoints)

# diff = []
# for m in keypoints:
#     xo, yo = m[1, 0], m[0, 0]
#     xw, yw = m[1, 1], m[0, 1]
#     base_xy = np.array([[xo],
#                         [yo],
#                         [1]])
#     warp_xy = np.matmul(Hmatrix, base_xy)[:,0]
#     pred_xw = round(warp_xy[0]/warp_xy[2])
#     pred_yw = round(warp_xy[1]/warp_xy[2])
#     diff.append(round(math.sqrt((pred_xw-xw)**2+(pred_yw-yw)**2)))
# diff = np.array(diff)
# print(np.mean(diff))