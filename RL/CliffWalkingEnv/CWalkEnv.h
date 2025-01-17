#include "DataType.h"
#include <vector>

#ifndef CLIFFWALKINGEVN_H_
#define CLIFFWALKINGEVH_H_

class CWalkEnv
{
private:

    uint16 u16NCol;
    uint16 u16NRow;
    uint16 u16NumStates;

public:
    CWalkEnv();
    CWalkEnv(uint16 nCol, uint16 nRow);
    ~CWalkEnv();
    void CreateP();
    void PrintValue();

};
#endif