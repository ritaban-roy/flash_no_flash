from skimage import io
from scipy.ndimage import gaussian_filter
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import binary_erosion, binary_dilation, binary_fill_holes
import gc
from opts import get_opts

def gamma_correction(x):
    x = np.where(x <= 0.0404482, x/12.92, ((x+0.055)/1.055)**2.4)
    return x

def gaussian(x, sig):
    return (
        1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x) / sig, 2.0) / 2)
    )
def gaussian_numpy(x, sig):
   return (
        1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power(x / sig, 2.0) / 2)
    ) 

def piecewise_bilateral_per_channel(I, F, minI, maxI, nb_segments, sigma_s=32, sigma_r=0.05):
    i = minI + np.arange(nb_segments+1) * ((maxI - minI)/nb_segments)
    J = None
    for j in range(nb_segments+1):
        G_j = gaussian_numpy(F - i[j], sig=sigma_r)
        K_j = gaussian_filter(G_j, sigma=sigma_s)
        H_j = G_j * I
        H_j_ = gaussian_filter(H_j, sigma=sigma_s)
        J_j = H_j_ / K_j
        if(j == 0):
            J = J_j
        else:
            J = np.dstack((J, J_j))
        del G_j, K_j, H_j, J_j
        gc.collect()
    final_single_channel_image = np.zeros_like(I)
    
    for j in range(nb_segments):
        mask = ((F >= i[j]) * (F < i[j+1])).astype(np.float16)
        masked_I = F * mask
        I_j = np.clip(masked_I - (i[j]*mask), a_max=None, a_min=0)
        I_j_1 = np.clip(-masked_I + (i[j+1]*mask), a_max=None, a_min=0)
        final_single_channel_image += ((J[:, :, j]*I_j_1 + J[:, :, j+1]*I_j)/(I_j + I_j_1 + 0.001))
        del mask, masked_I
        gc.collect()
        
    return final_single_channel_image

    

def piecewise_bilateral_filter(I, F, sigma_s=40, sigma_r=0.4):
    minI = np.min(F) - 0.01
    maxI = np.max(F) + 0.01
    nb_segments = np.ceil((maxI - minI)/sigma_r).astype(np.int8)
    
    #apply on all channel
    r = piecewise_bilateral_per_channel(I[:, :, 0], F[:, :, 0], minI, maxI, nb_segments ,sigma_s, sigma_r)
    g = piecewise_bilateral_per_channel(I[:, :, 1], F[:, :, 1], minI, maxI, nb_segments ,sigma_s, sigma_r)
    b = piecewise_bilateral_per_channel(I[:, :, 2], F[:, :, 2], minI, maxI, nb_segments ,sigma_s, sigma_r)
    return np.dstack((r,g,b))


def read_tif(img_path):
    raw_data = io.imread(img_path).astype(np.double)
    return raw_data

def detail_transfer(I, F):
    ISO_A = 1600
    ISO_F = 100
    tau_shad = 0.001
    
    
    #Bilateral filtering on ambient image
    A_base = piecewise_bilateral_filter(I, I, sigma_s=32, sigma_r=0.05)
    #Joint biltateral filtering
    A_nr = piecewise_bilateral_filter(I, F, sigma_s=2, sigma_r=0.05)
    #Biltaeral filtering on flash image
    F_base = piecewise_bilateral_filter(F, F, sigma_s=2, sigma_r=0.05)
    #Detail trasnfer
    A_detail = A_nr * ((F + 0.02)/(F_base + 0.02))
    
    
    F_lin = gamma_correction(F)
    A_lin = gamma_correction(I) * (ISO_F/ISO_A)
    
    #Use Y from xyY for mask calculation
    Y_F = 0.2126 * F_lin[:, :, 0] + 0.7152 * F_lin[:, :, 1] + 0.0722 * F_lin[:, :, 2]
    Y_A = 0.2126 * A_lin[:, :, 0] + 0.7152 * A_lin[:, :, 1] + 0.0722 * A_lin[:, :, 2]
    
    #Shadow mask
    mask_shad = np.where((Y_F - Y_A) <= tau_shad, 1, 0)
    mask_shad = binary_erosion(mask_shad)
    mask_shad = binary_fill_holes(mask_shad)
    mask_shad = binary_dilation(mask_shad)
    mask_shad = mask_shad.reshape((mask_shad.shape[0], mask_shad.shape[1], 1))
    
    #Specularity mask
    mask_spec = np.where(Y_F < 0.95, 0, 1)
    mask_spec = binary_erosion(mask_spec)
    mask_spec = binary_fill_holes(mask_spec)
    mask_spec = binary_dilation(mask_spec)
    mask_spec = mask_spec.reshape((mask_spec.shape[0], mask_spec.shape[1], 1))
    
    M = np.logical_or(mask_shad, mask_spec)
    M = np.clip(np.sum(M, axis = 2), a_min=0, a_max=1)
    M = np.reshape(M, (M.shape[0], M.shape[1], 1))
    #Final image with detail transfer and mask applied
    A_Final = (1 - M) * A_detail + M * A_base
    
    del mask_shad, mask_spec, M, F_lin, A_lin
    gc.collect()
    A_Final = np.clip(A_Final, a_min=0, a_max=1)
    A_nr = np.clip(A_nr, a_min=0, a_max=1)
    A_detail = np.clip(A_detail, a_min=0, a_max=1)
    A_base = np.clip(A_base, a_min=0, a_max=1)
    
    io.imsave(f'a_base.png', (A_base*255).astype(np.uint8))
    io.imsave(f'a_nr.png', (A_nr*255).astype(np.uint8))
    io.imsave(f'a_detail.png', (A_detail*255).astype(np.uint8))
    io.imsave(f'a_final.png', (A_Final*255).astype(np.uint8))
    
def test_bilateral():
    img = read_tif('../data/lamp/lamp_ambient.tif')/255.0
    fimg = read_tif('../data/lamp/lamp_flash.tif')/255.0
    
    bf_img = piecewise_bilateral_filter(img, fimg, sigma_s=4, sigma_r=0.05)
    bf_img = np.clip(bf_img, a_min=0, a_max=1)
    io.imsave(f'a_nr.png', (bf_img*255).astype(np.uint8))

def main(opts):
    I = read_tif(opts.i_path)/255.0
    if(opts.f_path == ''):
        F = I
        bf_img = piecewise_bilateral_filter(I, I, sigma_s=4, sigma_r=0.05)
        bf_img = np.clip(bf_img, a_min=0, a_max=1)
        io.imsave(f'a_base.png', (bf_img*255).astype(np.uint8))
    else:
        F = read_tif(opts.f_path)/255.0
        detail_transfer(I, F)

def do_diff():
    I2 = read_tif('../data/lamp/lamp_ambient.tif')
    I1 = read_tif('exp1_nr.png')
    I2 = np.mean(I2, axis=2)
    I1 = np.mean(I1, axis=2)
    plt.imshow((I1-I2), cmap='gray')
    plt.show()
    
if __name__ == '__main__':
    opts = get_opts()
    main(opts)