//**********************************************************************
//To test the Kalman filter by using CUDA
// 
//Created by Junkai Cheng, 2019/01/15
//**********************************************************************


#include "point.h"
#include "kalman_cuda.h"
#include <cstring>
#include <ctime>
#include <math.h>
#include <iostream>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"


void get_measure_data(float* data, int k, int point_num){
	//to generate and return the data measured
	float param_r = 1.9;
	float param_R = 1;
	float scale = 130;
	for (int i = 0; i < 2 * point_num; ++i){
		float U = rand() * (0.03 / RAND_MAX);
		float V = rand() * (0.03 / RAND_MAX);
		float e = sqrt(-2.0f * log(U)) * sin(2.0f * 3.14f * V);
		//std::cout << e;
		if (i % 2 == 0)
			data[i] = e + scale * ((param_R - param_r) * cos(param_r / param_R * (1 + Time*k)) + param_r / param_R * cos((1 + Time*k) - param_r / param_R * (1 + Time*k)));
		else
			data[i] = e + scale * ((param_R - param_r) * sin(param_r / param_R * (1 + Time*k)) - param_r / param_R * sin((1 + Time*k) - param_r / param_R * (1 + Time*k)));
	}
}

float error_calc(float x, float y, int k){
	//to calculate the error rate
	float param_r = 1.9f;
	float param_R = 1.0f;
	float scale = 130.0f;
	float actual_x = scale * ((param_R - param_r) * cos(param_r / param_R * (1 + Time*k)) + param_r / param_R * cos((1 + Time*k) - param_r / param_R * (1 + Time*k)));
	float actual_y = scale * ((param_R - param_r) * sin(param_r / param_R * (1 + Time*k)) - param_r / param_R * sin((1 + Time*k) - param_r / param_R * (1 + Time*k)));
	float x_e = (actual_x - x) / actual_x;
	float y_e = (actual_y - y) / actual_y;
	return sqrt(x_e * x_e + y_e * y_e);
}


void predict_loop(int iteration_num, float* predict, float* cov, float* new_predict, float* new_cov, int point_num, float delta_t){
	//iteration to predict
	float error = 0;
	for (int i = 0; i < iteration_num; ++i){

		float *data = new float[2 * point_num]();
		
		predict_single(predict, cov, new_predict, new_cov, point_num, delta_t);
		
		get_measure_data(data, i, point_num);
		update_single(data, new_predict, new_cov, predict, cov, point_num, delta_t, i);	
		delete[] data;
		error += error_calc(predict[0], predict[1], i);

	}
	std::cout << error*100.0/(19.0/Time +1) <<"%"<< std::endl;
}


void printMatrix(float* a, int Width, int length){
	// to print the matrix with given size
	for (int i = 0; i < length; ++i){
		for (int j = 0; j < Width; ++j){
			std::cout << a[i * Width + j]<<' ';
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

int main(){
	int ite_num = 19/Time;
	int points_num = 1;

	clock_t start, end;


	float *pred = new float[PredictSize*points_num]();
	float *new_predict = new float[PredictSize*points_num];
	float *cov = new float[CovSize*points_num]();
	//float *new_cov = new float[CovSize*points_num];

	float* covD, *new_predictD, *new_covD;

	cudaMalloc(&covD, points_num* CovSize* sizeof(float));
	cudaMalloc(&new_covD, points_num* CovSize* sizeof(float));
	cudaMemcpy(covD, cov, points_num*CovSize*sizeof(float), cudaMemcpyHostToDevice);

	start = clock();

	predict_loop(ite_num, pred, covD, new_predict, new_covD, points_num, Time);

	end = clock(); 

	std::cout << (double)(end - start) / CLOCKS_PER_SEC;
	delete[] pred;
	delete[] new_predict;
	delete[] cov;
	//delete[] new_cov;

	cudaFree(covD);
	cudaFree(new_covD);

	return 0;
}




