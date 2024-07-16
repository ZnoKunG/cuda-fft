﻿#ifndef __CUDACC__  
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

typedef std::map<uint32_t, std::vector<double>> ComputeTimeList;

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

std::vector<double> compute_dft(uint32_t N)
{
	std::cout << "Computing DFT N = " << N << std::endl;

	std::vector<double> timeUseList;

	std::complex<float>* a_cpu = new std::complex<float>[N];
	cuComplex* a_gpu = new cuComplex[N];

	for (uint32_t i = 0; i < N; i++)
	{
		a_cpu[i] = std::complex<float>((float)(i + 1), 0);
		a_gpu[i] = make_cuFloatComplex((float)(i + 1), 0);
	}

	std::complex<float>* A_cpu = new std::complex<float>[N];
	cuComplex* A_gpu = new cuComplex[N];

	clock_t start, end;
	double time_used;

	start = clock();
	computeDft_CPU(a_cpu, A_cpu, N);
	end = clock();
	time_used = ((double)(end - start) * 1000) / CLOCKS_PER_SEC;

	timeUseList.push_back(time_used);

	start = clock();
	computeDft_GPU(a_gpu, A_gpu, N);
	end = clock();
	time_used = ((double)(end - start) * 1000) / CLOCKS_PER_SEC;

	timeUseList.push_back(time_used);

	return timeUseList;
}

std::vector<double> compute_fft(uint32_t nSample)
{
	uint32_t N = ToPower2_N(nSample);

	std::cout << "Computing FFT N = " << N << std::endl;

	std::vector<double> timeUseList;

	std::complex<float>* a_cpu = new std::complex<float>[nSample];
	cuComplex* a_gpu = new cuComplex[nSample];

	for (uint32_t i = 0; i < nSample; i++)
	{
		a_cpu[i] = std::complex<float>((float)(i + 1), 0);
		a_gpu[i] = make_cuFloatComplex((float)(i + 1), 0);
	}

	std::complex<float>* a_cpu_2;
	cuComplex* a_gpu_2;
	a_cpu_2 = ToPower2_Matrix_CPU(a_cpu, nSample, 0);
	a_gpu_2 = ToPower2_Matrix_GPU(a_gpu, nSample, 0);

	std::complex<float>* A_cpu = new std::complex<float>[N];
	cuComplex* A_gpu = new cuComplex[N];

	clock_t start, end;
	double time_used;

	start = clock();
	computeFFT_CPU(a_cpu_2, A_cpu, N);
	end = clock();
	time_used = ((double)(end - start) * 1000) / CLOCKS_PER_SEC;

	timeUseList.push_back(time_used);

	start = clock();
	computeFFT_GPU(a_gpu_2, A_gpu, N);
	end = clock();
	time_used = ((double)(end - start) * 1000) / CLOCKS_PER_SEC;

	timeUseList.push_back(time_used);

	return timeUseList;
}

void write_csv_dft(std::string file_name, ComputeTimeList list)
{
	std::ofstream result(file_name);

	result << "N,dft_cpu,dft_gpu\n";

	for (const auto& pair : list)
	{
		result << pair.first << "," << pair.second[0] << "," << pair.second[1] << "\n";
	}

	result.close();
}

void write_csv_fft(std::string file_name, ComputeTimeList list)
{
	std::ofstream result(file_name);

	result << "N,fft_cpu,fft_gpu\n";

	for (const auto& pair : list)
	{
		result << pair.first << "," << pair.second[0] << "," << pair.second[1] << "\n";
	}

	result.close();
}

int main()
{
	ComputeTimeList timeComputeDFT;
	ComputeTimeList timeComputeFFT;
	
	for (uint32_t i = 20; i <= 5000; i += 20)
	{
		std::vector<double> timeList = compute_dft(i);
		timeComputeDFT.insert(std::make_pair(i, timeList));
	}

	write_csv_dft("result_dft.csv", timeComputeDFT);

	for (uint32_t i = 20; i <= 50000; i += 20)
	{
		std::vector<double> timeList = compute_fft(i);
		timeComputeFFT.insert(std::make_pair(i, timeList));
	}

	write_csv_fft("result_fft.csv", timeComputeFFT);

	for (const auto& pair : timeComputeDFT)
	{
		std::cout << "Time use to compute for N = " << pair.first << std::endl;
		std::cout << "DFT CPU (ms): " << pair.second[0];
		std::cout << ", DFT GPU (ms): " << pair.second[1] << std::endl;
	}

	for (const auto& pair : timeComputeFFT)
	{
		std::cout << "Time use to compute for N = " << pair.first << std::endl;
		std::cout << ", FFT CPU (ms): " << pair.second[0];
		std::cout << ", FFT GPU (ms): " << pair.second[1] << std::endl;
	}

	int i;
	std::cin >> i;
	return 0;
}
