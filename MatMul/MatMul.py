import numpy as np
import time
import pycuda.autoinit
from pycuda import driver
from pycuda.compiler import SourceModule


module = SourceModule("""
        __global__ void mult(double* A, double* B, double* C){
                const int row = blockIdx.y * blockDim.y + threadIdx.y;
                const int column = blockIdx.x * blockDim.x + threadIdx.x;
                const int N = 1024;
                for(int i = 0; i < N; i++){
                        C[row * N + column] += A[row * N + i] * B[i * N + column];              
                }       
        }
""")

def multiplication(A, B):
    C  = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i, j] += A[i, k] * B[k, j]
    return C

N = 1024
A = np.random.randn(N, N)
B = np.random.randn(N, N)
C = np.zeros((N, N))

block_size = (2, 2, 1)
grid_size = (int((N + block_size[0] - 1) / 2), int((N + block_size[1] - 1) / 2))
mult = module.get_function("mult")

start_cpu = time.time()
res_cpu = multiplication(A, B)
end_cpu = time.time()

start_gpu = time.time()
mult(driver.In(A), driver.In(B), driver.Out(C), block = block_size, grid = grid_size)
driver.Context.synchronize()
end_gpu = time.time()

print('Time of GPU: ',end_gpu - start_gpu, '\nTime of CPU: ',end_cpu - start_cpu)

if np.allclose(C, res_cpu):
        print('Results converge')
else:
        print('Results diverge')