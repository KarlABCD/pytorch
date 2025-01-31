#ifndef DATA_TYPE_H_
#define DATA_TYPE_H_
#include <vector>
using std::vector;
typedef unsigned char uint8;
typedef unsigned short uint16;
typedef short int16;
typedef float float32;
typedef unsigned char boolean;

typedef enum
{
    FuncNormal = 0,
    FuncAbNormal = 1
}tReturnType;

#define MAX_UINT8_VALUE 255
#define MAX_UINT16_VALUE 65535
#define FlOAT32_EQ_TOLERANCE 10e-9

#endif