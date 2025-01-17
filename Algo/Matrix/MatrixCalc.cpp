#include "DataType.h"
#include "MatrixCalc.h"

//template void MatTranspose(const float32 (&InputMat) [uint8(3)][uint8(3)], 
//               float32 (&OutputMat) [uint8(3)][uint8(3)]);

//template void MatPrint(const float32 (&InputMat) [uint8(3)][uint8(3)]);


void MatTranspose(const float32 * InputMatPtr, float32 * OutputMatPtr,
                  uint8 Row, uint8 Col)
{
    for(uint8 i = 0; i < Row; i++)
    {
        for(uint8 j = 0; j < Col; j++ )
        {
            *(OutputMatPtr + j * Row + i) = *(InputMatPtr + i * Col + j);
        }
    }
}

void MatPrint(const float32 * InputMatPtr, uint8 Row,
              uint8 Col, const char cName[])
{
    std::cout << cName << std::endl;
    for(uint8 i = 0; i < Row; i++)
    {
        for(uint8 j = 0; j < Col; j++ )
        {
            std::cout << std::fixed << std::setprecision(5) 
                    << *(InputMatPtr + i * Col + j) << " ";
        }
        std::cout << std::endl;
    }
}

void MatMultiply(const float32 * InputArr1Ptr, uint8 Row1, uint8 Col1,
                 const float32 * InputArr2Ptr, uint8 Col2,
                 float32 * OutputArrPtr)
{
    float32 temp = 0.0F;
    for (int i = 0; i < Row1; i++)
    {
        for (int j = 0; j < Col2; j++)
        {
            for (int k = 0; k < Col1; k++)
            {
                // OutputArr[i][j] += Input1Arr[i][k] * Input2Arr[k][j];
                 temp += *(InputArr1Ptr + i * Col1 + k)
                         *(*(InputArr2Ptr + k * Col2 + j)); 
            }
            *(OutputArrPtr + i * Col2 +j) = temp;
            temp = 0;
        }
    }
}

void MatCreateEye(float32 * ArrPtr, uint8 Size)
{
    for (uint8 i = 0; i < Size; i++)
    {
        for (uint8 j = 0; j < Size; j++)
        {
            if(i == j)
                *(ArrPtr + i * Size + j) = 1.0F;
            else
                *(ArrPtr + i * Size + j) = 0.0F;
        }
    }
}

void MatEleWiseDiv(const float32 * InputArrPtr, float32 * OutputArrPtr, uint8 row, 
                   uint8 col, const float32 Scalar)
{
    for (uint8 i = 0; i < row; i++) 
    {
        for (uint8 j = 0; j < col; j++) 
        {
            *(OutputArrPtr + i * col + j) = *(InputArrPtr + i * col + j) / Scalar;
        }
    }
}

void MatSubtract(const float32 * Input1ArrPtr, const float32 * Input2ArrPtr, 
                 float32 * OutputArrPtr, uint8 row, uint8 col) 
{
    for (uint8 i = 0; i < row; ++i) 
    {
        for (uint8 j = 0; j < col; ++j) 
        {
            *(OutputArrPtr + i*col + j) = *(Input1ArrPtr + i * col + j)
                                            - *(Input2ArrPtr + i * col + j);
        }
    }
}

void MatAdd(const float32 * Input1ArrPtr, const float32 * Input2ArrPtr, 
                 float32 * OutputArrPtr, uint8 row, uint8 col) 
{
    for (uint8 i = 0; i < row; ++i) 
    {
        for (uint8 j = 0; j < col; ++j) 
        {
            *(OutputArrPtr + i*col + j) = *(Input1ArrPtr + i * col + j)
                                            + *(Input2ArrPtr + i * col + j);
        }
    }
}