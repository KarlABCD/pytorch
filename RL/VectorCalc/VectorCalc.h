#ifndef VECTOR_CALC_H_
#define VECTOR_CALC_H_
#include "DataType.h"
#include <cmath>

extern tReturnType VectorSum(const vector<float32> * InputArr, float32 & ArrSum);
extern tReturnType VectorSum(const vector<vector<float32>> * InputMat, 
                             float32 & ArrSum);
extern tReturnType VectorMax(const vector<float32> * InputArr, float32 & Max);
extern tReturnType VectorCountEq(vector<float32> * InputArr, 
                            float32 const EqValue, uint16 & u16Eqcnt);
#endif