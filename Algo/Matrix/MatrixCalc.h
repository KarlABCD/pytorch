#ifndef MATRIXCALC_H_
#define MATRIXCALC_H_

#include "DataType.h"
#include <iostream>
#include <iomanip>

/*template <uint8 Row1, uint8 Col1, uint8 Col2>
void MatrixMultiply(const float32 Input1Arr[Row1][Col1],
                    const float32 Input2Arr[Col1][Col2],
                    float32 OutputArr[Row1][Col2]);

template <uint8 Row, uint8 Col>
void MatTranspose(const float32 (&InputMat) [Row][Col], 
               float32 (&OutputMat) [Col][Row])
{
    for(uint8 i = 0; i < Row; i++)
    {
        for(uint8 j = 0; j < Col; j++ )
        {
            OutputMat[j][i] = InputMat[i][j];
        }
    }
}

template <uint8 Row, uint8 Col>
void MatPrint(const float32 (&InputMat) [Row][Col], const char cName[])
{
    std::cout << cName << std::endl;
    for(uint8 i = 0; i < Row; i++)
    {
        for(uint8 j = 0; j < Col; j++ )
        {
            std::cout << InputMat[i][j] << " ";
        }
        std::cout << std::endl;
    }
}*/
void MatTranspose(const float32 * InputMatPtr, float32 * OutputMatPtr,
                  uint8 Row, uint8 Col);


void MatPrint(const float32 * InputMatPtr, uint8 Row,
              uint8 Col, const char cName[]);

void MatMultiply(const float32 * InputArray1Ptr, uint8 Row1, uint8 Col1,
                 const float32 * InputArray2Ptr, uint8 Col2,
                 float32 * OutputArray);

void MatCreateEye(float32 * ArrPtr, uint8 Size);

void MatEleWiseDiv(const float32 * InputArrPtr, float32 * OutputArrPtr, 
                    uint8 row, uint8 col, float32 Scalar); 

void MatSubtract(const float32 * Input1ArrPtr, const float32 * Input2ArrPtr, 
                 float32 * OutputArrPtr, uint8 row, uint8 col);

void MatAdd(const float32 * Input1ArrPtr, const float32 * Input2ArrPtr, 
                 float32 * OutputArrPtr, uint8 row, uint8 col);

#endif