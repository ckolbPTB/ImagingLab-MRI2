import numpy as np
import pywt
import sigpy
from pynufft import NUFFT

def k2i(kdat):
    if len(kdat.shape) == 1: # Carry out 1D FFT: k-space -> image space
       im = (np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(kdat,(0)),
                         (kdat.shape[0],),(0,), norm=None), (0,)))
    else: # Carry  out 2D FFT: k-space -> image space
       im = (np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(kdat,(0,1)),
                         (kdat.shape[0], kdat.shape[1]),(0,1), norm=None), (0,1)))

    return(im)
    
    
def i2k(im):
   if len(im.shape) == 1: # Carry out 1D FFT: image space -> k-space
       kdat = (np.fft.fftshift(np.fft.fftn(np.fft.fftshift(im,(0)),
                      (im.shape[0],),(0,), norm=None), (0,)))
   else: # Carry out 2D FFT: image space -> k-space
       kdat = (np.fft.fftshift(np.fft.fftn(np.fft.fftshift(im,(0,1)),
                      (im.shape[0], im.shape[1]),(0,1), norm=None), (0,1)))

   return(kdat)
   
   
def gauss_filt(dim, sigma):
    if sigma > 1 or sigma <= 0:
        raise KeyError('Sigma is normalised and should be between 0 and 1')
        
    # Create meshgrid based on image dimensions
    gridx, gridy = np.meshgrid(np.linspace(-1., 1., dim[0]), np.linspace(-1., 1., dim[1]))
    gridx = np.array(gridx)
    gridy = np.array(gridy)
    
    # Calculate 2D-Gaussian function
    filt_weights = np.exp(-(gridx**2 + gridy**2)/(2*sigma**2))
    
    return(filt_weights)
    
    
def get_lowres(kdat, low_res_factor):
    if low_res_factor > 1 or low_res_factor <= 0:
        raise KeyError('The factor should be between 0 and 1')
        
    low_res_factor = np.sqrt(low_res_factor)
    
    # Calculate low-res data extend
    kdim = np.round(np.array(kdat.shape) * np.array(low_res_factor))
    
    # Ensure dimensions are even
    kdim = kdim + kdim % 2
    
    # Calculate offset
    koff = np.round((np.array(kdat.shape) - kdim)/2)
    
    # Calculate extend of low-res k-space center
    kx0 = int(koff[0])
    kx1 = int(koff[0]+kdim[0])
    ky0 = int(koff[1])
    ky1 = int(koff[1]+kdim[1])
    
    kdat_lowres = np.zeros(kdat.shape, dtype=kdat.dtype)
    kdat_lowres[kx0:kx1, ky0:ky1] = kdat[kx0:kx1, ky0:ky1]
    
    return(kdat_lowres)
    
    
def create_us_mask(kdim, us_factor):
    if len(us_factor) != 2:
        raise KeyError('The undersampling factors should be a vector of length 2')
    if us_factor[0] < 1 or us_factor[1] < 1:
        raise KeyError('The undersampling factors should be larger than 1')
        
    # Ensure the undersampling factors are integers
    us_factor_kx = int(us_factor[0])
    us_factor_ky = int(us_factor[1])
    
    # Create a binary mask of the undersampling pattern
    us_mask = np.zeros(kdim)
    
    # Undersampling pattern
    us_mask[0::us_factor_kx, 0::us_factor_ky] = 1
    
    return(us_mask)    
    
   
def rec_2d_rad(kdat, ktraj, kdcf, idim):
   # Check size of input and reshape
   kdim = kdat.shape
   if len(kdim) == 2:
       kdat = np.reshape(kdat, (kdim[0]*kdim[1],))
       ktraj = np.reshape(ktraj, (kdim[0]*kdim[1],2))
       if len(kdcf) > 0:
           kdcf = np.reshape(kdcf, (kdim[0]*kdim[1],))
       
   # Set gridding parameters
   oversamp_factor = (1.6, 1.6)
   interp_size = (6, 6)
   
   # Check input parameters
   if len(kdat.shape) != 1:
        raise KeyError('kdat should be 1D is %dD' % (len(kdat.shape,)))

   if len(ktraj.shape) != 2:
        raise KeyError('ktraj should be 2D is %dD' % (len(ktraj.shape,))) 
       
   if ktraj.shape[1] != 2 or ktraj.shape[0] != kdat.shape[0]:
       raise KeyError('ktraj should be [%d x 2] but is [%d x %d]' % (kdat.shape[0], ktraj.shape[0], ktraj.shape[1]))
       
   if ktraj.max() > 0.5 or ktraj.min() < -0.5:
       raise KeyError('ktraj should be between [-0.5 0.5] but is between [%3.1f %3.1f]' % (ktraj.max(), ktraj.min()))
       
   if len(kdcf) > 0 and len(kdcf.shape) != 1:
       raise KeyError('kdcf should be 1D is %dD' % (len(kdcf.shape),)) 
       

   # Calculate oversampled size for gridding and ensure it is even
   kdim_os = [round(idim[0]*oversamp_factor[0]),
                      round(idim[1]*oversamp_factor[1])]
   kdim_os[0]+= kdim_os[0] % 2 
   kdim_os[1]+= kdim_os[1] % 2 
   kdim_os = tuple(kdim_os)
   
   A = NUFFT()
   A.plan(om=ktraj*2*np.pi, Nd=idim, Kd=kdim_os, Jd=interp_size)
   if len(kdcf) > 0:
       kdat = np.multiply(kdat, kdcf)
   return A.adjoint(kdat)


def sense_unfold(im_us, csm, us_factor):
    if len(us_factor) != 2:
        raise KeyError('us_factor should be 1D vector with two entries for each dimension')
    if us_factor[0] < 1 or us_factor[1] < 1:
        raise KeyError('us_factor should be larger or equal to 1')
    if us_factor[0] > 1 and us_factor[1] > 1:   
        raise KeyError('Only one entry in us_factor should larger than 1')
        
    im = np.zeros([im_us.shape[0], im_us.shape[1]], dtype=im_us.dtype)    
    if us_factor[0] > 1:    
        u_step = int(im_us.shape[0]/us_factor[0])
        for ynd in range(im_us.shape[1]):
            for xnd in range(u_step):
                A = np.transpose(csm[xnd::u_step, ynd, :])
                unfolded = np.dot(np.linalg.pinv(A, 1e-4), im_us[xnd, ynd, :, np.newaxis])
                
                im[xnd::u_step, ynd] = unfolded[:,0] 
                
    else:
        u_step = int(im_us.shape[1]/us_factor[1])
        for xnd in range(im_us.shape[0]):
            for ynd in range(u_step):
                A = np.transpose(csm[xnd, ynd::u_step, :])
                unfolded = np.dot(np.linalg.pinv(A, 1e-4), im_us[xnd, ynd, :, np.newaxis])
                
                im[xnd, ynd::u_step] = unfolded[:,0]
    return(im)


def create_sparse_sig(dim, noise_scale=0):
    sparse_sig = np.zeros((dim,1))
    sparse_sig[10,0] = 0.3
    sparse_sig[20,0] = 0.8
    sparse_sig[40,0] = 0.1
    sparse_sig[50,0] = 0.4
    sparse_sig[80,0] = 1.0
    sparse_sig = sparse_sig + np.random.randn(128,1)*noise_scale
    
    return(sparse_sig)


def soft_thresh(data, thresh):
    data[np.absolute(data) < thresh] = 0
    data[data > 0] = data[data > 0] - thresh
    data[data < 0] = data[data < 0] + thresh
    return(data)
    
    
def create_1d_us_mask(dim, us_factor, type='reg'):
    us_mask = np.zeros((dim,1))
    if type == 'reg':
        us_mask[::us_factor] = 1
    else:
        rnd_perm = np.random.permutation(np.linspace(0,dim-1,dim,dtype=np.int32))
        us_mask[rnd_perm[:int(dim/us_factor)]] = 1
        
    return(us_mask)
        
        
def it_soft_thresh(fdat, us_mask, num_it, thresh, red_thresh):
    fdat_it = np.zeros((fdat.shape[0], num_it+1), dtype=fdat.dtype)
    idat_it = np.zeros((fdat.shape[0], num_it+1), dtype=fdat.dtype)
    res_it = np.zeros((num_it,1))
    
    fdat_it[:,0] = fdat[:,0]
    idat_it[:,0] = k2i(fdat[:,0])
    
    for ind in range(num_it):
        # Threshhold current signal estimate
        idat_it[:,ind] = soft_thresh(idat_it[:,ind], thresh)
        
        # Transform signal to frequency space
        fdat_it[:, ind+1] = i2k(idat_it[:,ind])

        # Enforce data consistency
        fdat_it[us_mask[:,0]==1,ind+1] = fdat[us_mask[:,0]==1,0]

        # Reconstruct signal
        idat_it[:,ind+1] = k2i(fdat_it[:, ind+1])

        # Calculate residual
        res_it[ind,0] = np.absolute(np.sqrt(np.sum((idat_it[:,ind+1] - idat_it[:,ind])**2)))
        
        # Reduce threshhold
        thresh = thresh*red_thresh
        
    return(idat_it)
    
    
def it_soft_thresh_2d(kdat, dcf, us_mask, num_it, thresh, red_thresh, wave_type='haar'):
    kdat_it = np.zeros((kdat.shape[0], kdat.shape[1], num_it+1), dtype=kdat.dtype)
    idat_it = np.zeros((kdat.shape[0], kdat.shape[1], num_it+1), dtype=kdat.dtype)
    res_it = np.zeros((num_it,1))
    
    kdat_it[:,:,0] = kdat
    idat_it[:,:,0] = k2i(np.multiply(kdat, dcf))
    
    for ind in range(num_it):
        # Apply wavelet transform
        coeffs = pywt.wavedec2(idat_it[:,:,ind], wave_type, level=5)
        idat_wl = pywt.coeffs_to_array(coeffs)
        
        # Threshhold current signal estimate
        idat_wl_thresh = soft_thresh(idat_wl[0], thresh)

        # Transform back to image space
        coeffs = pywt.array_to_coeffs(idat_wl_thresh, idat_wl[1], 'wavedec2')
        idat_it[:,:,ind] = pywt.waverec2(coeffs, wave_type)

        # Transform signal to frequency space
        kdat_it[:,:,ind+1] = i2k(idat_it[:,:,ind])

        # Enforce data consistency
        kdat_it[us_mask==1,ind+1] = kdat[us_mask==1]
        
        # Reconstruct signal
        idat_it[:,:,ind+1] = k2i(kdat_it[:,:,ind+1])
        
        # Calculate residual
        res_it[ind,0] = np.absolute(np.sqrt(np.sum((idat_it[:,:,ind+1] - idat_it[:,:,ind])**2)))
        
        # Reduce threshhold
        thresh = thresh*red_thresh
        
    return(idat_it)

   
   
def phantom (n = 256):
    return sigpy.shepp_logan(shape=(n, n), dtype=float)

