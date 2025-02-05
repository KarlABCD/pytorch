#include "VectorCalc.h"
#include <iostream>
using std::cout;
using std::endl;

tReturnType VectorSum(const vector<float32> * InputArr, float32 & ArrSum)
{

    if(InputArr->size() > size_t(0))
    {
        for(float32 input : *InputArr)
        {
            ArrSum += input;
        }
        return FuncNormal;
    }
    else
    {
        return FuncAbNormal;
    }
}

tReturnType VectorSum(const vector<vector<float32>> * InputMat, 
                             float32 & ArrSum)
{
    for(vector<float32> InputVector : *InputMat)
    {
        for(float32 Input : InputVector)
        {
            ArrSum += Input;
        }
    }
    return FuncNormal;
}

tReturnType VectorMax(const vector<float32> * InputArr, float32 & Max)
{
    
    if( InputArr == nullptr)
        return FuncAbNormal;
    else
    {
        Max = (*InputArr)[0];
        for(float32 Input: *InputArr)
        {
            if(Input > Max)
            {
                Max = Input;
            }
            else
            {
                continue;
            }
        }
        return FuncNormal;
    }
}

tReturnType VectorCountEq(vector<float32> * InputArr, float32 const EqValue,
                            uint16 & u16Eqcnt)
{
    u16Eqcnt = 0U;
    if (InputArr == nullptr)
    {
        return FuncAbNormal;
    }
    else
    {
        for (float32 Input:*InputArr)
        {
            if(fabs(Input - EqValue) < FlOAT32_EQ_TOLERANCE)
            {
                u16Eqcnt++;
            }
        }
        return FuncNormal;
    }
}

tReturnType VectorMaxIdx(const vector<float32> * InputArr,
                        uint16 & MaxIdx)
{
    float32 Max;
    MaxIdx = 0U;
    if(InputArr == nullptr)
    {
        return FuncAbNormal;
    }
    else
    {
        Max = (*InputArr)[0];
        for (uint16 i = 0; i < (*InputArr).size(); i++)
        {
            if ((*InputArr)[i] > Max)
            {
                Max = (*InputArr)[i];
                MaxIdx = i;
            }
            else
            {
                continue;
            }
        }
        return FuncNormal;
    }
}