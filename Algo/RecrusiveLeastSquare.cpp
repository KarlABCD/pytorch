#include "RecrusiveLeastSquare.h"
using namespace std;

const uint8 u8ArraySize = 11;
const uint8 u8PloyIndex = 2;
const float32 lanmbda = 0.5;
const float32 OriParaArray[u8PloyIndex + 1] = {2, 3, 4};

int main()
{   
    float32 XArr[u8ArraySize] = {0};
    float32 YArr[u8ArraySize] = {0};

    string strXName = "X Value";
    string strYName = "Y Value";

    float32 ParaArr[u8PloyIndex + 1] = {0};
    float32 VarX;
    float32 ErrorMax;

    DataInit(XArr, YArr, OriParaArray, u8ArraySize);
    DisplayArray(XArr, u8ArraySize, strXName);
    DisplayArray(YArr, u8ArraySize, strYName);
};


void DataInit(float32 * InputArrPtr, float32 * OutputArrPtr, const float32 * OriParaArrPtr, uint8 u8ArraySize)
{
    for (uint8 i = 0; i < u8ArraySize; i++)
    {
        *(InputArrPtr + i) = i * 0.1F;
        for (uint8 j = 0; i < u8PloyIndex + 1; j++)
        {
            *(OutputArrPtr + i) = *(OutputArrPtr + i) + *(OriParaArrPtr + j) * pow(*(InputArrPtr + i), float32(j));
        }
        
    }
};

void DisplayArray(const float32 * Array, const uint8 u8ArraySize, 
                  string cArrayName)
{
    for (uint8 i = 0; i < u8ArraySize; i++)
    {    
        cout << cArrayName << ": " << *(Array + i) << "\n";
    }
};

/*void Fitting2DRLS(const float32 * InputArrPtr, const float32 * OutputArrPtr, const uint8 u8ArraySize, 
                const uint8 u8PloyIndex, const float32 lanmbda,
                float32 * ParaArrPtr, float32 * VarXPtr, float32 * ErrorMaxPtr)
{
    float32 B[u8ArraySize][u8PloyIndex + 1] = {0};

    for (uint8 i = 0; i < u8ArraySize; i++)
    {
        for (uint8 j = 0; j < u8PloyIndex + 1; j++)
            B[i][0] = ;
    }

};*/