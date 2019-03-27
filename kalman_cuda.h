#pragma once
#include "point.h"
extern "C" void predict_single(float* predict, float* cov, float* new_predict, float* new_cov, int point_num, float delta_t);

extern "C" void update_single(float* data, float* predict, float* cov, float* new_predict, float* new_cov, int point_num, float delta_t, int ite_nem);
