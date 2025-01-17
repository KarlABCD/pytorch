#include "RecrusiveLeastSquare.h"
using namespace std;

const uint8 u8ArraySize = 25;
const uint8 u8PloyIndex = 2;
const float32 lanmbda = 0.1;
const float32 OriParaArray[u8PloyIndex + 1] = {4, 3, 2};
static float32 A[u8PloyIndex + 1][u8PloyIndex + 1] = {0};
static float32 k[u8PloyIndex + 1][1] = {0};

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
    MatCreateEye(&A[0][0], u8PloyIndex + 1);
    for (uint8 i = 0; i < u8ArraySize; i++)
    {
        Fitting2DRLS(&XArr[i], &YArr[i], ParaArr, &VarX, &ErrorMax);
    }
}

void DataInit(float32 InputArr[], float32 OutputArr[], 
              const float32 OriParaArr[], uint8 u8ArraySize)
{
    for (uint8 i = 0; i < u8ArraySize; i++)
    {
        InputArr[i] = i * 0.1F;
        for (uint8 j = 0; j < u8PloyIndex + 1; j++)
        {
            OutputArr [i] = OutputArr[i] + 
                            OriParaArr[j] * pow(InputArr[i], float32(j));
        }
    }
}

void DisplayArray(const float32 Array[], const uint8 u8ArraySize, 
                  string cArrayName)
{
    for (uint8 i = 0; i < u8ArraySize; i++)
    {    
        cout << cArrayName << ": " << Array[i] << "\n";
    }
}

void Fitting2DRLS(const float32 * InputPtr, const float32 * OutputPtr, 
                float32 ParaArr[], float32 * VarXPtr, float32 * ErrorMaxPtr)
{
    float32 phi[u8PloyIndex + 1][1] = {0};
    float32 phiTran[1][u8PloyIndex + 1] = {0};
    //MatPrint(&A[0][0], uint8(3), uint8(3), "A");
    for (uint8 j = 0; j < u8PloyIndex + 1; j++)
    {
        phi[j][0] = pow(*InputPtr, float32(j));
    }
    MatTranspose(&phi[0][0], &phiTran[0][0], uint8(3), uint8(1));
    //MatPrint(&phi[0][0], uint8(3), uint8(1), "phi");
    //MatPrint(&phiTran[0][0], uint8(1), uint8(3), "phiTran");
    CoreProcess(&phi[0][0], &phiTran[0][0], &A[0][0], ParaArr, OutputPtr,
                VarXPtr, ErrorMaxPtr);
}

void CoreProcess(const float32 * phiPtr, const float32 * phiTranPtr, 
                 float32 * APtr, float32 * ParaArrPtr, const float32 * OutputPtr, 
                 float32 * VarXPtr, float32 * ErrorMaxPtr)
{
    float32 AmutPhi[u8PloyIndex + 1][1] = {0};
    float32 phiTranmutA[1][u8PloyIndex + 1] = {0};
    float32 temp = 0;
    float32 KmutPhiPtr[u8PloyIndex + 1][u8PloyIndex + 1] = {0};
    float32 Temp2[u8PloyIndex + 1][u8PloyIndex + 1] = {{0, 0, 0}, 
                                                       {0, 0, 0},{0, 0, 0}};
    MatPrint(phiPtr, uint8(1), uint8(3), "phiTranPtr");
    float32 Temp3[u8PloyIndex + 1][1] = {0};
    MatPrint(&AmutPhi[0][0], uint8(3), uint8(1), "AmutPhi-");
    MatMultiply(APtr, uint8(3), uint8(3),
                phiPtr, uint8(1), 
                &AmutPhi[0][0]);
    MatPrint(APtr, uint8(3), uint8(3), "A---");
    MatPrint(phiPtr, uint8(3), uint8(1), "phi");
    MatPrint(&AmutPhi[0][0], uint8(3), uint8(1), "AmutPhi");
    

    MatMultiply(phiTranPtr, uint8(1), uint8(3),
                APtr, uint8(3),
                &phiTranmutA[0][0]);
    MatPrint(&phiTranmutA[0][0], uint8(1), uint8(3), "phiTranmutA");
    MatMultiply(&phiTranmutA[0][0], uint8(1), uint8(3),
                phiPtr, uint8(1),
                &temp);
    //MatPrint(phiPtr, uint8(3), uint8(1), "phi");
    //MatPrint(&temp[0][0], uint8(1), uint8(1), "temp");
    float32 temp2 = lanmbda + temp;

    MatEleWiseDiv(&AmutPhi[0][0], &k[0][0], uint8(3), uint8(1), temp2);
    MatPrint(&k[0][0], uint8(3), uint8(1), "k");

    MatMultiply(&k[0][0], uint8(3), uint8(1),
                phiTranPtr, uint8(3),
                &KmutPhiPtr[0][0]);
    MatPrint(&KmutPhiPtr[0][0], uint8(3), uint8(3), "KmutPhiTran");
    MatMultiply(&KmutPhiPtr[0][0], uint8(3), uint8(3),
                APtr, uint8(3),
                &Temp2[0][0]);
    MatPrint(&Temp2[0][0], uint8(3), uint8(3), "Temp2");
    MatPrint(APtr, uint8(3), uint8(3), "A--");

    MatSubtract(APtr, &Temp2[0][0], APtr, uint8(3), uint8(3));
    MatPrint(APtr, uint8(3), uint8(3), "A-");
    
    MatEleWiseDiv(APtr, APtr, uint8(3), uint8(3), lanmbda);
    MatPrint(APtr, uint8(3), uint8(3), "A");

    temp = 0;
    MatMultiply(phiTranPtr, uint8(1), uint8(3),
                ParaArrPtr, uint8(1),
                &temp);
    MatPrint(phiTranPtr, uint8(1), uint8(3), "phiTranPtr");
    MatPrint(ParaArrPtr, uint8(3), uint8(1), "Para");
    MatPrint(&temp, uint8(1), uint8(1), "temp");
    temp2 = *OutputPtr - temp;
    MatEleWiseDiv(&k[0][0], &Temp3[0][0], uint8(3), uint8(3), 1/temp2);
    MatAdd(ParaArrPtr, &Temp3[0][0], ParaArrPtr, uint8(3), uint8(1));
    MatPrint(ParaArrPtr, uint8(3), uint8(1), "Para");
}
