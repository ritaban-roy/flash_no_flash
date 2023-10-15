from skimage import io
from scipy.ndimage import gaussian_filter
import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt
from functools import partial
from scipy import signal

def gradient(I):
    I_y = np.diff(I, axis=0, prepend=0)
    I_x = np.diff(I, axis=1, prepend=0)
    return np.dstack((I_x, I_y))

def divergence_of_grad(I):
    I_y = np.diff(I, axis=0, prepend=0)
    #I_y = I[:, :, 1]
    I_yy = np.diff(I_y, axis=0, append=0)
    I_x = np.diff(I, axis=1, prepend=0)
    #I_x = I[:, :, 0]
    I_xx = np.diff(I_x, axis=1, append=0)
    div_grad_I = I_xx + I_yy
    return div_grad_I

def divergence(I):
    #I_y = np.diff(I, axis=0, prepend=0)
    I_y = I[:, :, 1]
    I_yy = np.diff(I_y, axis=0, append=0)
    #I_x = np.diff(I, axis=1, prepend=0)
    I_x = I[:, :, 0]
    I_xx = np.diff(I_x, axis=1, append=0) 
    
    div_grad_I = I_xx + I_yy
    return div_grad_I

def laplacian(I):
    laplacian_filter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    I_lap = signal.convolve2d(I, laplacian_filter, mode='same', boundary='fill', fillvalue='0')
    return I_lap

def cgd(D, I_init, B, I_boundary, eps, N):
    '''
    Conjugate gradient descent poisson solver
    '''
    I_ = B * I_init + (1-B)*I_boundary
    r = B * (D - laplacian(I_))
    d = r
    delta_new = np.sum(r * r)
    #print(delta_new.shape)
    n = 0
    while np.sum(r * r) > eps**2 and n < N:
        q = laplacian(d)
        eta = delta_new/np.sum(d * q)
        I_ = I_ + B * (eta * d)
        r = B*(r - eta*q)
        delta_old = delta_new.copy()
        delta_new = np.sum(r * r)
        beta = delta_new/delta_old
        d = r + beta * d
        n+=1
    return I_

def grad_field_integration():
    '''
    Test for poisson solver
    '''
    img = read_png('../data/museum/museum_ambient.png')/255.0
    img = img[:, :, :-1]
    #img = (read_png('../data/my_captures/window_flash.jpg')/255.0)[::4, ::4, :]
    I_star = np.zeros_like(img)
    for channel in range(3):
        I = img[:, :, channel]
        #print(I.shape)
        #exit()
        I_init = np.zeros_like(I)
        
        B = np.ones_like(I)
        B[0, :] = 0
        B[-1, :] = 0
        B[:, 0] = 0
        B[:, -1] = 0
        
        I_boundary = np.ones_like(I)
        I_boundary[0, :] = I[0, :]
        I_boundary[-1, :] = I[-1, :]
        I_boundary[:, 0] = I[:, 0]
        I_boundary[:, -1] = I[:, -1]
        
        I_ = cgd(divergence(gradient(I)), I_init, B, I_boundary, eps=0.001, N=1000)
        I_star[:, :, channel] = I_
    plt.imshow(I_star, cmap='gray')
    plt.show()

def fused_grad_field():
    '''
    Fused gradient field for Question 2
    '''
    #I =  (read_png('../data/my_captures/window_ambient.jpg')/255.0)#[::10, ::10, :]#[:, :, :-1]
    #F = (read_png('../data/my_captures/window_flash.jpg')/255.0)#[::10, ::10, :]#[:, :, :-1]
    I = (read_png('../data/museum/museum_ambient.png')/255.0)[:, :, :-1]
    F = (read_png('../data/museum/museum_flash.png')/255.0)[:, :, :-1]
    sigma = 40 #sigma_s from the paper
    tau_s = 0.7 #tau_s from the paper
    
    I_star = np.zeros_like(I)
    for channel in range(3):
        a = I[:, :, channel]
        phi_ = F[:, :, channel]
        grad_a = gradient(a)
        grad_phi_ = gradient(phi_)
        
        numerator = np.abs(np.sum(grad_a * grad_phi_, axis=2))
        denominator = np.sqrt(np.sum(np.square(grad_phi_), axis=2))*np.sqrt(np.sum(np.square(grad_a), axis=2))+1e-8
        M = numerator/denominator
        M = M.reshape(M.shape[0], M.shape[1], 1)
        w_s = np.tanh(sigma * (phi_ - tau_s))
        w_s = (w_s - np.min(w_s))/np.max(w_s)
        w_s = w_s.reshape(w_s.shape[0], w_s.shape[1], 1)
        
        grad_phi_star = w_s * grad_a + (1-w_s)*(M * grad_phi_ + (1-M)*grad_a)
        I_init = np.zeros_like(a)
        
        B = np.ones_like(a)
        B[0, :] = 0
        B[-1, :] = 0
        B[:, 0] = 0
        B[:, -1] = 0
        
        I_boundary = np.ones_like(a)
        I_boundary[0, :] = phi_[0, :] #(phi_[0, :] + a[0, :])/2
        I_boundary[-1, :] = phi_[-1, :]#(phi_[-1, :] + a[-1, :])/2
        I_boundary[:, 0] = phi_[:, 0] #(phi_[:, 0] + a[:, 0])/2
        I_boundary[:, -1] = phi_[:, -1] #(phi_[:, -1] + a[:, -1])/2
        #plt.imshow(divergence(grad_phi_star), cmap='gray')
        #plt.show()
        I_ = cgd(divergence(grad_phi_star), I_init, B, I_boundary, eps=0.001, N=1000)
        I_star[:, :, channel] = I_
    
    plt.imshow(I_star, cmap='gray')
    plt.show()
    io.imsave(f'museum_fused.png', (I_star*255).astype(np.uint8))
    
def remove_reflection():
    '''
    Reflection removal for Question 4
    '''
    I =  (read_png('../data/Answer_4_Bonus/captured_images/spray_ambient.JPG')/255.0)[::2, ::2, :]
    F = (read_png('../data/Answer_4_Bonus/captured_images/spray_flash.JPG')/255.0)[::2, ::2, :]
    sigma = 40
    tau_ue = 0.1
    plt.imshow(I + F, cmap='gray')
    plt.show()
    I_star = np.zeros_like(I)
    for channel in range(3):
        a = I[:, :, channel]
        phi_ = F[:, :, channel]
        H = (a + phi_)
        w_ue = 1 - np.tanh(sigma * (a - tau_ue))
        w_ue = w_ue.reshape(w_ue.shape[0], w_ue.shape[1], 1)
        grad_H = gradient(H)
        grad_a = gradient(a)
        
        numerator = grad_a * np.sum(grad_a * grad_H, axis=2).reshape((grad_a.shape[0], grad_a.shape[1], 1))
        denominator = (np.sum(np.square(grad_a), axis=2) + 1e-6).reshape((grad_a.shape[0], grad_a.shape[1], 1))
        grad_phi_star = w_ue * grad_H + (1 - w_ue) * (numerator/denominator)
        

        
        I_init = np.zeros_like(a)
        
        B = np.ones_like(a)
        B[0, :] = 0
        B[-1, :] = 0
        B[:, 0] = 0
        B[:, -1] = 0
        
        I_boundary = np.ones_like(a)
        I_boundary[0, :] = (phi_[0, :] + a[0, :])
        I_boundary[-1, :] = (phi_[-1, :] + a[-1, :])
        I_boundary[:, 0] = (phi_[:, 0] + a[:, 0])
        I_boundary[:, -1] = (phi_[:, -1] + a[:, -1])
        #plt.imshow(divergence(grad_phi_star), cmap='gray')
        #plt.show()
        I_ = cgd(divergence(grad_phi_star), I_init, B, I_boundary, eps=0.001, N=3000)
        I_star[:, :, channel] = I_
    
    I_star = np.clip(I_star, a_min=0, a_max=1)
    io.imsave(f'reflection_removed.png', (I_star*255).astype(np.uint8))


def read_png(img_path):
    raw_data = io.imread(img_path)
    #print(raw_data.shape, np.max(raw_data))
    return raw_data

def main():
    '''
    choose from fused_grad_field() ; remove_reflection()
    '''
    #fused_grad_field()
    remove_reflection()
    
if __name__ == '__main__':
    main()
    