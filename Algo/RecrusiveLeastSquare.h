#ifndef RECRUSIVELEASTSQUARE_H_
#define RECRUSIVELEASTSQUARE_H_

#include "Basic/DataType.h"
//#include "MatrixCalc.h"
#include <string>
#include <iostream>
#include <cmath>

void DataInit(float32 * InputArrPtr, float32 * OutputArrPtr, const float32 * OriParaArrPtr, uint8 u8ArraySize);
void DisplayArray(const float32 * Array, const uint8 u8ArraySize, 
                  std::string cArrayName);

#endif