#ifndef RECRUSIVELEASTSQUARE_H_
#define RECRUSIVELEASTSQUARE_H_

#include "DataType.h"
#include "MatrixCalc.h"
#include <string>
#include <iostream>
#include <cmath>

void DataInit(float32 InputArr[], float32 OutputArr[], 
              const float32 OriParaArr[], uint8 u8ArraySize);

void DisplayArray(const float32 Array[], const uint8 u8ArraySize, 
                  std::string cArrayName);

void Fitting2DRLS(const float32 * InputPtr, const float32 * OutputPtr,
                float32 ParaArr[], float32 * VarXPtr, float32 * ErrorMaxPtr);

void CoreProcess(const float32 * phiPtr, const float32 * phiTranPtr, 
                 float32 * APtr, float32 * ParaArr, const float32 * OutputPtr, 
                 float32 * VarXPtr, float32 * ErrorMaxPtr);
#endif