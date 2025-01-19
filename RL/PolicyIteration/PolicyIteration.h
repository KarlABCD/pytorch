#ifndef POLICYITERATION_H_
#define POLICYITERATION_H_

#include <vector>
#include <iostream>
#include <cmath>
#include "DataType.h"
#include "CWalkEnv.h"

using std::vector;

class PolicyIteration
{
private:

    vector<float32> v;
    vector<vector<float32>> pi;
    vector<vector<float32>> P;
    CWalkEnv env;
    uint16 u16EnvRows;
    uint16 u16EnvCols;

public:
    
    PolicyIteration();
    PolicyIteration(CWalkEnv ENV, uint16 u16Row, uint16 u16Col);
    ~PolicyIteration();
    void PrintStateValues();
    void PrintEnvValues();
    void PrintPolicy();
    void PrintActionValues();
};


#endif