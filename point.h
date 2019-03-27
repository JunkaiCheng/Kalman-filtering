#pragma once
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#define MeasureSize 2 * sizeof(float)
#define PredictSize  4 * sizeof(float)
#define CovSize 16 * sizeof(float)
#define Time 0.007


void predict_loop(int iteration_num, float* predict, float* cov, float* new_predict, float* new_cov, int point_num, float delta_t);

void printMatrix(float* a, int Width, int length);

