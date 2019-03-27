//************************************************************************************
//Kalman filter using CUDA
//
//Check https://en.wikipedia.org/wiki/Kalman_filter for more details about Kalman
//
//Created by Junkai Cheng, 2019/01/15
//***********************************************************************************

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <cstring>
#include <math.h>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "point.h"
#include "kalman_cuda.h"



__device__ float ele_multi(float* A, float* B, int Awidth, int Bwidth, int tx, int ty);

//some constant matrices to be used in this part

__device__ float H[8] = {
	1.0, 0, 0, 0,
	0, 1.0, 0, 0
};

__device__ float HT[8] = {
	1.0, 0,
	0, 1.0,
	0, 0,
	0, 0
};
__device__ float A[16] = {
	1.0, 0, Time, 0,
	0, 1.0, 0, Time,
	0, 0, 1.0, 0,
	0, 0, 0, 1.0
};
__device__ float AT[16] = {
	1.0, 0, 0, 0,
	0, 1.0, 0, 0,
	Time, 0, 1.0, 0,
	0, Time, 0, 1.0
};
__device__ float Q[16] = {
	0, 0.01, 0, 0,
	0.01, 0, 0, 0,
	0, 0, 0.002, 0.01,
	0, 0, 0.01, 0.001
};
__device__ float R[4] = {
	0.01, 0.01,
	0.01, 0.01
};
__device__ float I[16] = {
	1.0, 0, 0, 0,
	0, 1.0, 0, 0,
	0, 0, 1.0, 0,
	0, 0, 0, 1.0
};


__global__ void PredictKernel(float* predictD, float* covD, float* new_predictD, float* new_covD, int point_num){
	//Kernel function for the first two steps of Kalman Filter
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;

	__shared__ float temp[CovSize];
	float value;

	//caculate x_k' = A * x_{k-1}
	if (tx < 1){
		value = ele_multi(A, predictD + bx * PredictSize, 4, 1, tx, ty);
		new_predictD[bx * PredictSize + ty] = value;
	}

	//calculate P_k' = A * P_{k-1} * A^T + Q
	value = ele_multi(A, covD + bx * CovSize, 4, 4, tx, ty);
	temp[ty * 4 + tx] = value;

	__syncthreads();

	value = ele_multi(temp, AT, 4, 4, tx, ty);
	if (bx < point_num)
		new_covD[bx * CovSize + ty * 4 + tx] = value + Q[ty * 4 + tx];

	__syncthreads();
}

__global__ void UpdateKernel(float* dataD, float* predictD, float* covD, float* new_predictD, float* new_covD, int point_num, int ite_num){
	//kernel functino for the left three steps of Kalman Filter
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	
	float value;
	
	__shared__ float temp[CovSize];

	//calculate H*P
	if (ty < 2){
		value = ele_multi(H, covD + bx*CovSize, 4, 4, tx, ty);
		temp[ty * 4 + tx] = value;
	}

	__syncthreads();

	//calculate H*P_k*H^T + R
	__shared__ float temp2[PredictSize];
	if (ty < 2 && tx < 2){
		value = ele_multi(temp, HT, 4, 2, tx, ty);
		temp2[ty * 2 + tx] = value + R[ty * 2 + tx];
	}

	//calculate P_k* H^T
	__shared__ float temp3[8];
	if (tx < 2){
		value = ele_multi(covD + bx*CovSize, HT, 4, 2, tx, ty);
		temp3[ty * 2 + tx] = value;
	}
	
	__syncthreads();

	//calculate K
	__shared__ float K[8];
	float det = temp2[0] * temp2[3] - temp2[2] * temp2[1];
	__shared__ float temp2_inv[4];
	temp2_inv[0] = 1.0f / det * temp2[3];
	temp2_inv[1] = -1.0f / det * temp2[1];
	temp2_inv[2] = -1.0f / det * temp2[2];
	temp2_inv[3] = 1.0f / det * temp2[0];
	if (tx < 2){
		value = ele_multi(temp3, temp2_inv, 2, 2, tx, ty);
		K[ty * 2 + tx] = value;
	}

	//calculate z_k - H*x_k'
	__shared__ float temp4[8];
	if (tx < 1 && ty < 2){
		value = ele_multi(H, predictD + bx * PredictSize, 4, 1, tx, ty);
		temp4[ty] = dataD[MeasureSize * bx + ty] - value;
	}
	
	__syncthreads();
	//calculate x_k
	if (tx < 1){
		value = ele_multi(K, temp4, 2, 1, tx, ty);
		new_predictD[bx * PredictSize + ty] = predictD[bx * PredictSize + ty] + value;
	}
	
	//caculate I-K*H
	__shared__ float temp5[CovSize];
	value = ele_multi(K, H, 2, 4, tx, ty);
	temp5[ty * 4 + tx] = I[ty * 4 + tx] - value;
	__syncthreads();

	//calculate P_k
	value = ele_multi(temp5, covD + bx*CovSize, 4, 4, tx, ty);
	new_covD[bx * PredictSize + ty * 4 + tx] = value;

	__syncthreads();
}

void predict_single(float* predict, float* covD, float* new_predict, float* new_covD, int point_num, float delta_t){
	//the first two steps of Kalman Filter

	float* predictD, *new_predictD;

	cudaMalloc(&predictD, point_num* PredictSize* sizeof(float));
	cudaMalloc(&new_predictD, point_num* PredictSize* sizeof(float));

	cudaMemcpy(predictD, predict, point_num*PredictSize*sizeof(float), cudaMemcpyHostToDevice);
	dim3 dimBlock(4, 4);
	dim3 dimGrid(point_num, 1);

	PredictKernel << <dimGrid, dimBlock >> >(predictD, covD, new_predictD, new_covD, point_num);

	// After this step, data in PredictData is x', data in Covariance is P' 
	cudaMemcpy(new_predict, new_predictD, point_num*PredictSize*sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(predictD);
	cudaFree(new_predictD);
}

void update_single(float* data, float* predict, float* covD, float* new_predict, float* new_covD, int point_num, float delta_t, int ite_num){
	//the left three steps of Kalman Filter

	float* predictD,  *new_predictD, *dataD;

	cudaMalloc(&predictD, point_num* PredictSize* sizeof(float));
	cudaMalloc(&new_predictD, point_num* PredictSize* sizeof(float));
	cudaMalloc(&dataD, point_num * 2 * sizeof(float));

	cudaMemcpy(predictD, predict, point_num*PredictSize*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dataD, data, point_num * 2 * sizeof(float), cudaMemcpyHostToDevice);

	dim3 dimBlock(4, 4);
	dim3 dimGrid(point_num, 1);
	UpdateKernel << <dimGrid, dimBlock >> >(dataD, predictD, covD, new_predictD, new_covD, point_num, ite_num);

	// After this step, data in PredictData is x, data in Covariance is P 
	cudaMemcpy(new_predict, new_predictD, point_num*PredictSize*sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(predictD);
	cudaFree(new_predictD);
	cudaFree(dataD);
}

__device__ float ele_multi(float* A, float* B, int Awidth,  int Bwidth, int tx, int ty){
	//calculate one element of the product of two matrices
	float Pvalue = 0;
	for (int k = 0; k < Awidth; ++k){
		float Melement = A[ty * Awidth + k];
		float Nelement = B[k * Bwidth + tx];
		Pvalue += Melement * Nelement;
	}
	return Pvalue;
}