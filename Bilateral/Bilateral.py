import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import time
import cv2
import sys

mod = SourceModule("""
texture<unsigned int, 2, cudaReadModeElementType> tex;
__global__ void filt_gpu(unsigned int* result, const int M, const int N, const float sigma_d, const float sigma_r)
{
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    const int j = threadIdx.y + blockDim.y * blockIdx.y;
    if ((i < M) && (j < N)) {
        float s = 0;
        float c = 0;
        for (int l = i - 1; l <= i + 1; l++){
            for (int k = j - 1; k <= j + 1; k++){
                float img1 = tex2D(tex, k, l) / 255;
                float img2 = tex2D(tex, i, j) / 255;
                float g = exp(-(pow(k - i, 2) + pow(l - j, 2)) / pow(sigma_d, 2));
                float r = exp(-pow((img1 - img2) * 255, 2) / pow(sigma_r, 2));
                c += g * r;
                s += g * r * tex2D(tex, k, l);
            }
        }
        result[i * N + j] = s / c;
    }
}
""")

def filt_cpu(im, sigma_r, sigma_d):
    result = np.zeros(im.shape)
    for i in range(1, im.shape[0] - 1):
        for j in range(1, im.shape[1] - 1):
            c = 0
            s = 0
            for k in range(i-1, i+2):
                for l in range(j-1, j+2):
                    g = np.exp(-((k - i) ** 2 + (l - j) ** 2) / sigma_d ** 2)
                    r = np.exp(-(im[k, l] - im[i, j]) ** 2 / sigma_r ** 2)
                    c += g*r
                    s += g*r*im[k, l]
            result[i, j] = s / c
    return result


path_image = sys.argv[1]
sigma_r = float(sys.argv[2])
sigma_d = float(sys.argv[3])

image = cv2.imread(path_image, cv2.IMREAD_GRAYSCALE)

N = image.shape[0]
M = image.shape[1]
block_size = (8, 8, 1)
grid_size = (int(np.ceil(N/block_size[0])),int(np.ceil(M/block_size[1])))

start_cpu = time.time()
result = filt_cpu(image, sigma_r, sigma_d)
end_cpu = time.time()

result_gpu = np.zeros((N, M), dtype = np.uint32)

filt_gpu = mod.get_function("filt_gpu")

start_gpu = time.time()
tex = mod.get_texref("tex")
tex.set_filter_mode(drv.filter_mode.LINEAR)
tex.set_address_mode(0, drv.address_mode.MIRROR)
tex.set_address_mode(1, drv.address_mode.MIRROR)
drv.matrix_to_texref(image.astype(np.uint32), tex, order="C")
filt_gpu(drv.Out(result_gpu), np.int32(N), np.int32(M), np.float32(sigma_d), np.float32(sigma_r), block=block_size, grid=grid_size, texrefs=[tex])
drv.Context.synchronize()
end_gpu = time.time()

cv2.imwrite('res_gpu.bmp', result_gpu.astype(np.uint8))
cv2.imwrite('res_cpu.bmp', result)

print('CPU Time {}'.format(end_cpu - start_cpu))
print('GPU Time {}'.format(end_gpu - start_gpu))
print('GPU/CPU {}'.format((end_cpu - start_cpu)/(end_gpu - start_gpu)))
                                                                        
