%matplotlib inline
from scipy import misc
import matplotlib.pyplot as plt
# import dft_fft as df
import numpy as np
import math
import cmath
import huffman as hf
import imageio
def dct(img_mat):
    N = 8
    coeff_mat = np.zeros((N, N), dtype=int)
    C = [1.0/math.sqrt(2.0)] + [1.0] * (N-1)
    for r in range(N):
        for c in range(N):
            coeff_mat[r,c] = int(1/4 * C[r] * C[c]*
                                 sum([img_mat[x,y] *
                                      math.cos(math.pi*(2*x+1)*r/16) *
                                      math.cos(math.pi*(2*y+1)*c/16)
                                      for x in range(N) for y in range(N)]))
    return coeff_mat
    
def idct(icoeff_mat):
    N = 8
    reconstructed_mat = np.zeros((N, N), dtype=int)
    C = [1.0/math.sqrt(2.0)] + [1.0] * (N-1)
    for r in range(N):
        for c in range(N):
            reconstructed_mat[r,c] = int(1/4 *
                                 sum([C[x] * C[y]*
                                      icoeff_mat[x,y] *
                                      math.cos(math.pi*(2*r+1)*x/16) *
                                      math.cos(math.pi*(2*c+1)*y/16)
                                      for x in range(N) for y in range(N)]))
    return reconstructed_mat

def q(coeff_mat, q_mat):
    N = 8
    q_coeff_mat = coeff_mat[:, :]
    for r in range(N):
        for c in range(N):
            q_coeff_mat[r, c] = round(coeff_mat[r, c] / q_mat[r, c])
    return q_coeff_mat

def iq(iq_coeff_mat, q_mat):
    return iq_coeff_mat * q_mat

def zigzag(q_coeff_mat):
    N = 8
    q_coeff_list = []
    for i in range(N):
        if i%2 == 0:
            x, y = i, 0
            for k in range(i+1):
                q_coeff_list.append(q_coeff_mat[x,y])
                x, y = x-1, y+1
        else:
            x, y = 0, i
            for k in range(i+1):
                q_coeff_list.append(q_coeff_mat[x, y])
                x, y = x + 1, y - 1
    for j in range(1, N):
        if j%2 != 0:
            x, y = N-1, j
            for k in range(N-j):
                q_coeff_list.append(q_coeff_mat[x, y])
                x, y = x-1, y+1
        else:
            x, y = j, N-1
            for k in range(N-j):
                q_coeff_list.append(q_coeff_mat[x, y])
                x, y = x + 1, y - 1
    return q_coeff_list

def izigzag(iq_coeff_list):
    N = 8
    working_list = iq_coeff_list[:]
    iq_coeff_mat = np.zeros((N, N), dtype = iq_coeff_list[0].dtype)
    for i in range(N):
        if i%2 == 0:
            x, y = i, 0
            for k in range(i+1):
                iq_coeff_mat[x, y] = working_list.pop(0)
                x, y = x-1, y+1
        else:
            x, y = 0, i
            for k in range(i+1):
                iq_coeff_mat[x, y] = working_list.pop(0)
                x, y = x + 1, y - 1
    for j in range(1, N):
        if j%2 != 0:
            x, y = N-1, j
            for k in range(N-j):
                iq_coeff_mat[x, y] = working_list.pop(0)
                x, y = x-1, y+1
        else:
            x, y = j, N-1
            for k in range(N-j):
                iq_coeff_mat[x, y] = working_list.pop(0)
                x, y = x + 1, y - 1
    return iq_coeff_mat

def rl_enc(coeff_list):
    rl_list = []
    i = 0
    zero_count = 0
    while i < len(coeff_list):
        if coeff_list[i] != 0:
            rl_list.append((zero_count, coeff_list[i]))
            i += 1
            zero_count = 0
        else:
            i += 1
            zero_count += 1
    rl_list.append((0, 0))
    return rl_list

def rl_dec(rl_list):
    coeff_list = []
    for t in rl_list[:-1]:
        for i in range(t[0]):
            coeff_list.append(0)
        coeff_list.append(t[1])
    coeff_list.extend([0]* (N*N-1 - len(coeff_list)))
    
    return coeff_list


import huffman as hf

q_mat = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                  [12, 12, 14, 19, 26, 58, 60, 55],
                  [14, 13, 16, 24, 40, 57, 69, 56],
                  [14, 17, 22, 29, 51, 87, 80, 62],
                  [18, 22, 37, 56, 68, 109, 103, 77],
                  [24, 35, 55, 64, 81, 104, 113, 92],
                  [49, 64, 78, 87, 103, 121, 120, 101],
                  [72, 92, 95, 98, 112, 100, 103, 99]])

image_src0 = imageio.imread('lenna_grey.png')
H, W = image_src0.shape
image_src = image_src0[:H//2, :W//2]
H = H // 2
W = W //2
N = 8
decode_dict = {}
dc_list = []
ac_run_collections = []

for r in range(0, H//N):
    for c in range(0, W//N):
        img_mat = image_src[r*N:(r+1)*N, c*N:(c+1)*N]
        coeff_mat = dct(img_mat)
        q_coeff_mat = q(coeff_mat, q_mat)
        zz = zigzag(q_coeff_mat)
        dc_list.append(zz[0])
        ac_run = rl_enc(zz[1:])
        ac_run_collections.extend(ac_run)
        decode_dict[(r, c)] = {'img_mat':img_mat,
                              'coeff_mat':coeff_mat,
                              'q_coeff_mat':q_coeff_mat,
                              'zz':zz,
                              'ac_run':ac_run
                              }

#DC
dc_diff_list = [dc_list[0]] + [dc_list[i]-dc_list[i-1] for i in range(1, len(dc_list))]
dc_tree = hf.gen_huffman_tree(dc_diff_list)
dc_dict = {}
hf.gen_huffman_dict(dc_tree[0], '', dc_dict)
dc_code = hf.huffman_enc(dc_dict, dc_diff_list)
dc_diff_decoded = hf.huffman_dec_list(dc_tree, dc_code)
dc_list_decoded = [dc_diff_decoded[0]]
for i in range(1, len(dc_diff_decoded)):
    dc_list_decoded.append(dc_list_decoded[-1]+dc_diff_decoded[i])

# Build Huff Tree for AC runs
rl_tree = hf.gen_huffman_tree(ac_run_collections)
rl_dict = {}
hf.gen_huffman_dict(rl_tree[0], '', rl_dict)
        
for r in range(0, H//N):
    for c in range(0, W//N):
        ac_run_encoded = hf.huffman_enc(rl_dict, decode_dict[(r, c)]['ac_run'])
        ac_run_decoded = hf.huffman_dec_list(rl_tree, ac_run_encoded)       
        izz = rl_dec(ac_run_decoded) # 63 elements
        izz.insert(0, dc_list_decoded[r*(W//N) + c])
        iq_coeff_mat = izigzag(izz)
        icoeff_mat = iq(iq_coeff_mat, q_mat)
        reconstructed = idct(icoeff_mat)
        
        decode_dict[(r, c)]['ac_run_encoded'] = ac_run_encoded
        decode_dict[(r, c)]['ac_run_decoded'] = ac_run_decoded 
        decode_dict[(r, c)]['izz'] = izz
        decode_dict[(r, c)]['iq_coeff_mat'] = iq_coeff_mat
        decode_dict[(r, c)]['icoeff_mat'] = icoeff_mat
        decode_dict[(r, c)]['reconstructed'] = reconstructed
        

# Look insaide a block's tranformation

print (decode_dict[(11, 19)])
# Compare NUmerical Difference between an original block and its reconstruction

print (decode_dict[(6, 6)]["reconstructed"] - decode_dict[(6, 6)]["img_mat"])
# Plot Original image
plt.imshow(image_src)
# Plot Reconstructed Image
rec_img = np.zeros((H, W), dtype=image_src.dtype)
for r in range(0, H//N):
    for c in range(0, W//N):
        rec_img[r*N:(r+1)*N, c*N:(c+1)*N] = decode_dict[(r, c)]['reconstructed']
plt.imshow(rec_img)

# sum[ len(decode_dict[k]['ac_run_encoded']) for k in decode_dict.keys()]
decode_dict.keys()

for k, v in decode_dict[(25, 15)].items():
    print (k)
    print (v, len(v))
    
rec_img = np.zeros((H, W), dtype=image_src.dtype)
for r in range(0, H//N):
    for c in range(0, W//N):
        rec_img[r*N:(r+1)*N, c*N:(c+1)*N] = decode_dict[(r, c)]['reconstructed']
plt.imshow(rec_img)
import cmath
import math

def fft(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)
    half_N = N//2
    if N == 1:
        X[0] = x[0]
    else:
        x_even = np.array([x[2*k] for k in range(half_N)], dtype=complex)
        x_odd = np.array([x[2*l + 1] for l in range(half_N)], dtype=complex)
        X_even = fft(x_even)
        X_odd = fft(x_odd)
        W = cmath.exp(-1j*2*math.pi/N)
        X_odd_mult = np.array([X_odd[m] * W**m for m in range(half_N)])
        X[:half_N] = X_even + X_odd_mult
        X[half_N:] = X_even - X_odd_mult
    return X

def fft2d(f):
    '''2D Forward FFT'''
    (Nr, Nc) = f.shape
    F = np.zeros((Nr, Nc), dtype=complex)
    for m in range(Nr):
        F[m,:] = fft(f[m, :])
    for n in range(Nc):
        F[:, n] = fft(F[:, n])
    return(F)

decode_dict[(60, 50)]
F = fft2d(image_src[128:128+32, 128:128+32])
plt.imshow(np.log10(np.abs(F)))
plt.colorbar()
a = np.zeros((8, 8), dtype=int)
a[7, 7] = 100
print (dct(idct(a)))

a = np.zeros((8, 8), dtype=int)
a[7, 7] = 100
plt.imshow(idct(a))

