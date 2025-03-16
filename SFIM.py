import numpy as np
from scipy import ndimage
from scipy import signal
from skimage.transform import resize



def upsample_interp23(image, ratio):
    
    r,c = image.shape

    CDF23 = 2*np.array([0.5, 0.305334091185, 0, -0.072698593239, 0,
                        0.021809577942, 0, -0.005192756653, 0, 
                        0.000807762146, 0, -0.000060081482])
    d = CDF23[::-1] 
    CDF23 = np.insert(CDF23, 0, d[:-1])
    BaseCoeff = CDF23
    # CDF23 = np.insert(CDF23, 0, CDF23[::-1][:-1])  
    # BaseCoeff = 2 * CDF23
    
    first = True
    for z in range(1,int(np.log2(ratio))+1):
        # I1LRU = np.zeros((b, 2**z*r, 2**z*c))
        I1LRU = np.zeros((2**z * r, 2**z * c), dtype= image.dtype)
        if first:
            I1LRU[1::2, 1::2]=image
            first = False
        else:
            I1LRU[::2,::2]=image
        
        for j in range(I1LRU.shape[0]):
            I1LRU[j,:]=ndimage.correlate(I1LRU[j,:],BaseCoeff,mode='wrap', cval=0)
        for k in range(I1LRU.shape[1]):
            I1LRU[:,k]=ndimage.correlate(I1LRU[:,k],BaseCoeff,mode='wrap', cval = 0)
        image = I1LRU
        
    # re_image=np.transpose(I1LRU, (1, 2, 0))
    # re_image = I1LRU
        
    resized_image = resize(image, (9, 9), anti_aliasing=True)
    return resized_image



def SFIM(pan, hs, ratio):

    # M, N = pan.shape
    # m, n = hs.shape
    
    #upsample
    u_hs = upsample_interp23(hs, ratio)
    
    if np.mod(ratio, 2)==0:
        ratio = ratio + 1
        
    # Create averaging kernel
    kernel = np.ones((ratio, ratio)) / (ratio**2)
    
    # Nornamlize pan
    pan = (pan - (-50.0))/51
    # pan = (pan - np.mean(pan))*(np.std(u_hs, ddof=1)/np.std(pan, ddof=1)) + np.mean(u_hs)
    # Create smoothed version of PAN
    lrpan = signal.convolve2d(pan, kernel, mode='same', boundary='wrap')
    
    
    # Calculate SFIM
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_map = np.divide(pan, lrpan + 1e-8, 
                            where=(lrpan + 1e-8) != 0)
        
        I_SFIM = u_hs * ratio_map

    
    # Handle edge cases
    I_SFIM = np.nan_to_num(I_SFIM, nan=0.0, posinf=0.0, neginf=0.0)
    # I_SFIM = np.clip(I_SFIM, 0, 1)
    
    return I_SFIM.astype(np.float32)

