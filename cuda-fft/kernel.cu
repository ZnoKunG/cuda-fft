#ifndef __CUDACC__  
	#define __CUDACC__
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include <complex>
#include <cuComplex.h>
#include <map>
#include <vector>
#include <fstream>

typedef std::map<uint32_t, double> ComputeTimeList;

# define M_PI 3.14159265358979323846  /* pi */

#define BLOCK_SIZE 32

uint32_t reverse_bits(uint32_t n)
{
	n = ((n & 0xffff0000) >> 16 | (n & 0x0000ffff) << 16);
	n = ((n & 0xff00ff00) >> 8 | (n & 0x00ff00ff) << 8);
	n = ((n & 0xf0f0f0f0) >> 4 | (n & 0x0f0f0f0f) << 4);
	n = ((n & 0xcccccccc) >> 2 | (n & 0x33333333) << 2);
	n = ((n & 0xaaaaaaaa) >> 1 | (n & 0x55555555) << 1);
	return n;
}

__device__ uint32_t reverse_bits_gpu(uint32_t n)
{
	n = ((n & 0xffff0000) >> 16 | (n & 0x0000ffff) << 16);
	n = ((n & 0xff00ff00) >> 8 | (n & 0x00ff00ff) << 8);
	n = ((n & 0xf0f0f0f0) >> 4 | (n & 0x0f0f0f0f) << 4);
	n = ((n & 0xcccccccc) >> 2 | (n & 0x33333333) << 2);
	n = ((n & 0xaaaaaaaa) >> 1 | (n & 0x55555555) << 1);
	return n;
}

void dft(std::complex<float>* a, std::complex<float>* A, uint32_t N)
{
	for (size_t k = 0; k < N; k++)
	{
		std::complex<float> w_k(0, -2 * (float)M_PI * k / N);

		std::complex<float> sum_a = 0;

		for (size_t n = 0; n < N; n++)
		{
			sum_a += a[n] * std::exp(std::complex<float>(0, w_k.imag() * n));
		}

		A[k] = sum_a;
	}
}

__global__ void dft_kernel(cuFloatComplex* a, cuFloatComplex* A, uint32_t N)
{
	int k = blockIdx.x * blockDim.x + threadIdx.x;

	float w_k = -2 * M_PI * k / N;

	cuFloatComplex sum_a = make_cuFloatComplex(0, 0);

	for (size_t n = 0; n < N; n++)
	{
		sum_a = cuCaddf(sum_a, cuCmulf(a[n], make_cuFloatComplex(cosf(w_k * n), sinf(w_k * n))));
	}

	A[k] = sum_a;
}

void computeDft_CPU(std::complex<float>* a, std::complex<float>* A, uint32_t N)
{
	dft(a, A, N);
}

void computeDft_GPU(cuComplex* a, cuComplex* A, uint32_t N)
{
	// GPU DFT Execution
	// -----------------
	cuComplex* cuda_a;
	cuComplex* cuda_A;

	cudaMalloc(&cuda_a, sizeof(a) * N);
	cudaMalloc(&cuda_A, sizeof(A) * N);

	cudaMemcpy(cuda_a, a, sizeof(a) * N, cudaMemcpyHostToDevice);

	int size = N;
	int blockSize = min(size, BLOCK_SIZE);
	dim3 threadsPerBlock(blockSize, 1);
	dim3 blocksPerGrid((size + blockSize - 1) / blockSize, 1);
	dft_kernel<<<blocksPerGrid, threadsPerBlock >>>(cuda_a, cuda_A, N);

	cudaMemcpy(A, cuda_A, sizeof(A) * N, cudaMemcpyDeviceToHost);

	cudaFree(cuda_a);
	cudaFree(cuda_A);
	// -----------------
}

void fft(std::complex<float>* a, std::complex<float>* A, uint32_t N)
{
	int logN = (int)log2f((float)N);

	// Rearrange input in the reverse order
	for (uint32_t i = 0; i < N; i++)
	{
		uint32_t rev_i = reverse_bits(i);

		// Only retrieve first logN bits of the reverse bit
		rev_i = rev_i >> (32 - logN);

		A[i] = a[rev_i];
	}

	// Loop through each stage
	for (int s = 1; s <= logN; s++)
	{
		// Optimized power of two 2^s
		int m = 1 << s;
		int mh = 1 << (s - 1);

		// Calculate step up W of this stage ex. First stage W0 -> W4, Second stage W0 -> W2 -> W4 -> W6
		std::complex<float> twiddle = std::exp(std::complex<float>(0, -2 * (float)M_PI / m));

		// Loop through each pair of butterfly in this stage (step up each pair by 0, m, 2m, ..., N - m)
		for (uint32_t k = 0; k < N; k += m)
		{
			std::complex<float> twiddle_factor = 1;

			// Loop through to calculate inside the pair
			for (int j = 0; j < mh; j++)
			{
				std::complex<float> x = A[k + j];
				std::complex<float> y = twiddle_factor * A[k + j + mh];

				// Update next W by the twiddle (depending on the stage twiddle = W^n)
				// ex. for N = 8, first stage twiddle = W^4, second stage twiddle = W^2 and so on
				twiddle_factor *= twiddle;

				// Assign the element of butterfly
				A[k + j] = x + y;
				A[k + j + mh] = x - y;
			}
		}
	}
}

__global__ void fft_kernel(cuComplex* a, cuComplex* A, uint32_t N, int logN)
{
	// This thread will run and calculate two element in the array at a time
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

	// Reverse bit
	uint32_t rev_i = reverse_bits_gpu(2 * i);
	rev_i = rev_i >> (32 - logN);
	A[2 * i] = a[rev_i];

	rev_i = reverse_bits_gpu(2 * i + 1);
	rev_i = rev_i >> (32 - logN);
	A[2 * i + 1] = a[rev_i];

	// Wait until all threads is set its pair of the initial reverse-order input above
	__syncthreads();

	for (int s = 1; s <= logN; s++)
	{
		int mh = 1 << (s - 1); // mh = 2^(s - 1)
		int m = 1 << s;

		// k = start index of the set of butterflies (increment by m)
		int k = (i / mh) * m;

		// j = child index of element in the set
		int j = i % mh;
		
		int kj = k + j;

		cuFloatComplex twiddle = make_cuFloatComplex(cosf(-2 * M_PI * j / m), sinf(-2 * M_PI * j / m));

		cuFloatComplex x = A[kj];
		cuFloatComplex y = cuCmulf(twiddle, A[kj + mh]);

		A[kj] = cuCaddf(x, y);
		A[kj + mh] = cuCsubf(x, y);

		// Wait all threads to calculate on their stage
		__syncthreads();
	}
}

void computeFFT_CPU(std::complex<float>* a, std::complex<float>* A, uint32_t N)
{
	if (N & (N - 1))
	{
		std::cout << "N must be a power of 2" << std::endl;
		return;
	}

	fft(a, A, N);
}

void computeFFT_GPU(cuComplex* a, cuComplex* A, uint32_t N)
{
	if (N & (N - 1))
	{
		std::cout << "N must be a power of 2" << std::endl;
		return;
	}

	// GPU FFT Execution
	// -----------------
	int logN = (int)log2(N);
	cuComplex* cuda_a;
	cuComplex* cuda_A;

	cudaMalloc(&cuda_a, sizeof(a) * N);
	cudaMalloc(&cuda_A, sizeof(A) * N);

	cudaMemcpy(cuda_a, a, sizeof(a) * N, cudaMemcpyHostToDevice);

	int size = N >> 1; // size = N / 2
	int blockSize = min(size, BLOCK_SIZE);
	dim3 threadsPerBlock(blockSize, 1);
	dim3 blocksPerGrid((size + blockSize - 1) / blockSize, 1);
	fft_kernel << <blocksPerGrid, threadsPerBlock >> > (cuda_a, cuda_A, N, logN);

	cudaMemcpy(A, cuda_A, sizeof(A) * N, cudaMemcpyDeviceToHost);

	cudaFree(cuda_a);
	cudaFree(cuda_A);
	// --------------
}

uint32_t ToPower2_N(uint32_t N)
{
	// if N is power of 2
	if (!(N & (N - 1)))
	{
		return N;
	}

	int exponent = (int)(log2(N)) + 1;
	return 1 << exponent;
}

std::complex<float>* ToPower2_Matrix_CPU(std::complex<float>* in, uint32_t N, int fill_with)
{
	uint32_t N_p2 = ToPower2_N(N);
	std::complex<float>* out = new std::complex<float>[N_p2];
	for (uint32_t i = 0; i < N_p2; i++)
	{
		if (i < N)
		{
			out[i] = in[i];
		}
		else out[i] = std::complex<float>((float)fill_with, (float)fill_with);
	}

	return out;
}

cuComplex* ToPower2_Matrix_GPU(cuComplex* in, uint32_t N, int fill_with)
{
	uint32_t N_p2 = ToPower2_N(N);
	cuComplex* out = new cuComplex[N_p2];
	for (uint32_t i = 0; i < N_p2; i++)
	{
		if (i < N)
		{
			out[i] = in[i];
		}
		else out[i] = make_cuComplex((float)fill_with, (float)fill_with);
	}

	return out;
}

double compute_dft_cpu(uint32_t N)
{
	std::complex<float>* a_cpu = new std::complex<float>[N];

	for (uint32_t i = 0; i < N; i++)
	{
		a_cpu[i] = std::complex<float>((float)(i + 1), 0);
	}

	std::complex<float>* A_cpu = new std::complex<float>[N];

	clock_t start, end;
	double time_used;

	start = clock();
	computeDft_CPU(a_cpu, A_cpu, N);
	end = clock();
	time_used = ((double)(end - start) * 1000) / CLOCKS_PER_SEC;

	return time_used;
}

double compute_dft_gpu(uint32_t N)
{
	cuComplex* a_gpu = new cuComplex[N];

	for (uint32_t i = 0; i < N; i++)
	{
		a_gpu[i] = make_cuFloatComplex((float)(i + 1), 0);
	}

	cuComplex* A_gpu = new cuComplex[N];

	clock_t start, end;
	double time_used;

	start = clock();
	computeDft_GPU(a_gpu, A_gpu, N);
	end = clock();
	time_used = ((double)(end - start) * 1000) / CLOCKS_PER_SEC;

	return time_used;
}

double compute_fft_cpu(uint32_t nSample)
{
	uint32_t N = ToPower2_N(nSample);

	std::complex<float>* a_cpu = new std::complex<float>[nSample];

	for (uint32_t i = 0; i < nSample; i++)
	{
		a_cpu[i] = std::complex<float>((float)(i + 1), 0);
	}

	std::complex<float>* a_cpu_2;
	a_cpu_2 = ToPower2_Matrix_CPU(a_cpu, nSample, 0);

	std::complex<float>* A_cpu = new std::complex<float>[N];

	clock_t start, end;
	double time_used;

	start = clock();
	computeFFT_CPU(a_cpu_2, A_cpu, N);
	end = clock();
	time_used = ((double)(end - start) * 1000) / CLOCKS_PER_SEC;
	
	return time_used;
}

double compute_fft_gpu(uint32_t nSample)
{
	uint32_t N = ToPower2_N(nSample);

	cuComplex* a_gpu = new cuComplex[nSample];

	for (uint32_t i = 0; i < nSample; i++)
	{
		a_gpu[i] = make_cuFloatComplex((float)(i + 1), 0);
	}

	cuComplex* a_gpu_2;
	a_gpu_2 = ToPower2_Matrix_GPU(a_gpu, nSample, 0);

	cuComplex* A_gpu = new cuComplex[N];

	clock_t start, end;
	double time_used;

	start = clock();
	computeFFT_GPU(a_gpu_2, A_gpu, N);
	end = clock();
	time_used = ((double)(end - start) * 1000) / CLOCKS_PER_SEC;

	return time_used;
}

void write_csv(std::string file_name, ComputeTimeList list)
{
	std::ofstream result(file_name);

	for (const auto& pair : list)
	{
		result << pair.first << "," << pair.second << "\n";
	}

	result.close();
}

int main()
{
	ComputeTimeList timeComputeDFT_cpu;
	ComputeTimeList timeComputeDFT_gpu;
	ComputeTimeList timeComputeFFT_cpu;
	ComputeTimeList timeComputeFFT_gpu;
	
	int repeat_dft = 10;
	int sample_dft = 2000;
	int step_dft = 20;
	for (uint32_t i = 20; i <= sample_dft; i += step_dft)
	{
		double avg_cpu_time = 0;
		double avg_gpu_time = 0;
		
		for (int k = 0; k < repeat_dft; k++)
		{
			double cpu_time = compute_dft_cpu(i);
			double gpu_time = compute_dft_gpu(i);

			avg_cpu_time += cpu_time;
			avg_gpu_time += gpu_time;
		}

		avg_cpu_time = avg_cpu_time / step_dft;
		avg_gpu_time = avg_gpu_time / step_dft;

		timeComputeDFT_cpu.insert(std::make_pair(i, avg_cpu_time));
		timeComputeDFT_gpu.insert(std::make_pair(i, avg_gpu_time));

		std::cout << "\rDFT Computation progress: " << round((float)i * 100 / sample_dft) << "%";
	}

	write_csv("result_dft_cpu.csv", timeComputeDFT_cpu);
	write_csv("result_dft_gpu.csv", timeComputeDFT_gpu);

	int repeat_fft = 10;
	int sample_fft = 50000;
	int step_fft = 20;
	for (uint32_t i = 20; i <= sample_fft; i += step_fft)
	{
		double avg_cpu_time = 0;
		double avg_gpu_time = 0;

		for (int k = 0; k < repeat_fft; k++)
		{
			double cpu_time = compute_fft_cpu(i);
			double gpu_time = compute_fft_gpu(i);

			avg_cpu_time += cpu_time;
			avg_gpu_time += gpu_time;
		}

		avg_cpu_time = avg_cpu_time / repeat_fft;
		avg_gpu_time = avg_gpu_time / repeat_fft;

		timeComputeFFT_cpu.insert(std::make_pair(i, avg_cpu_time));
		timeComputeFFT_gpu.insert(std::make_pair(i, avg_gpu_time));

		std::cout << "\rFFT Computation progress: " << round((float)i * 100 / sample_fft) << "%";
	}

	write_csv("result_fft_cpu.csv", timeComputeFFT_cpu);
	write_csv("result_fft_gpu.csv", timeComputeFFT_gpu);

	std::cout << "Computation success!" << std::endl;

	int i;
	std::cin >> i;
	return 0;
}
