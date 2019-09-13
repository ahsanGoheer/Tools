import cv2 
import numpy as np
import numpy.matlib as matty
import os


def binarize (img):
    Use_MedianFilter = True
    # Getting the un-binarized image.
    raw_image=img   
    
    
    #Check if the image is GrayScale or not.
    if(len(raw_image)!=2):
        grayScale=cv2.cvtColor(raw_image,cv2.COLOR_BGR2GRAY)
    else:
        grayScale=raw_image

    #Getting the height and width of the Image.
    m,n = grayScale.shape

	# apply automatic Canny edge detection using the computed median
    sigma=0.33
    v = np.median(grayScale)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
	
    #Applying Canny Edge Detector to the Image.
    
    canny_Image = cv2.Canny(grayScale,lower,upper,3)
    
    
    kernel = np.ones((5,5),np.uint8)
    closing = cv2.morphologyEx(canny_Image, cv2.MORPH_CLOSE, kernel)

    
    #Finding Index Values in Image Where Pixels are White.  
    inds = np.where(closing==255)
    #Finding Index Values in Image Where Pixels are Black.  
    inds2 = np.where(closing==0)
    #Comparing  with original Image to obtain the pixel values for text.  
    pix_val = grayScale[inds]
    #Comparing  with original Image to obtain the pixel values for back.  
    pix_val_back = grayScale[inds2]
    
    #Taking Median of Text and back and then comparing them.
    median_text=np.median(pix_val,axis=0)
    median_back=np.median(pix_val_back,axis=0)
    
    if median_text>median_back:
        pre_binarize = cv2.bitwise_not(grayScale)
    else:
        pre_binarize=grayScale
 
    new_img = cv2.copyMakeBorder(pre_binarize,9,9,9,9,cv2.BORDER_CONSTANT,value=255)

    img= apply_threshold(new_img,wolf_threshold(new_img))

    if Use_MedianFilter:
        img = cv2.medianBlur(img,3)
  
    img=cv2.flip(img,1)
    
    x, y= img.shape
    new_y = int((y / x) * 90)
    img = cv2.resize(img, (new_y, 90), interpolation = cv2.INTER_CUBIC)
    return img
    


def wolf_threshold(img, w_size=10, k=0.5):
    # Obtaining rows and cols
    rows, cols = img.shape
    i_rows, i_cols = rows + 1, cols + 1

    # Computing integral images
    integ = np.zeros((i_rows, i_cols), np.float)
    sqr_integral = np.zeros((i_rows, i_cols), np.float)

    integ[1:, 1:] = np.cumsum(np.cumsum(img.astype(np.float), axis=0), axis=1)
    sqr_img = np.square(img.astype(np.float))
    sqr_integral[1:, 1:] = np.cumsum(np.cumsum(sqr_img, axis=0), axis=1)

    # Defining grid
    x, y = np.meshgrid(np.arange(1, i_cols), np.arange(1, i_rows))

    # Obtaining local coordinates
    hw_size = w_size // 2
    x1 = (x - hw_size).clip(1, cols)
    x2 = (x + hw_size).clip(1, cols)
    y1 = (y - hw_size).clip(1, rows)
    y2 = (y + hw_size).clip(1, rows)

    # Obtaining local areas size
    l_size = (y2 - y1 + 1) * (x2 - x1 + 1)

    # Computing sums
    sums = (integ[y2, x2] - integ[y2, x1 - 1] -
            integ[y1 - 1, x2] + integ[y1 - 1, x1 - 1])
    sqr_sums = (sqr_integral[y2, x2] - sqr_integral[y2, x1 - 1] -
                sqr_integral[y1 - 1, x2] + sqr_integral[y1 - 1, x1 - 1])

    # Computing local means
    means = sums / l_size

    # Computing local standard deviation
    stds = np.sqrt(sqr_sums / l_size - np.square(means))

    # Computing min and max values
    max_std = np.max(stds)
    min_v = np.min(img)

    # Computing thresholds
    thresholds = ((1.0 - k) * means + k * min_v + k * stds /
                  max_std * (means - min_v))

    return thresholds
    
    


def apply_threshold(img, threshold=128, wp_val=255):
    #wp_val is the white pixel value.
    return ((img >= threshold) * wp_val).astype(np.uint8)


