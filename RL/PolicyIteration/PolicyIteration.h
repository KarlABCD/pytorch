#ifndef POLICYITERATION_H_
#define POLICYITERATION_H_

#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#include "DataType.h"
#include "CWalkEnv.h"

using std::vector;


struct tActionInfo
{
    float32 Possibility;
    float32 Reward;
    uint16 u16NextState;
    boolean bEpisodeDone;
};

class PolicyIteration
{
private:

    vector<float32> v;
    vector<vector<float32>> pi;
    vector<vector<tActionInfo>> P;
    CWalkEnv env;
    size_t iEnvRows;
    size_t iEnvCols;

public:
    
    PolicyIteration();
    PolicyIteration(CWalkEnv ENV, uint16 u16Row, uint16 u16Col);
    ~PolicyIteration();
    void PrintStateValues();
    void PrintEnvValues();
    void PrintPolicy();
    void PrintActionValues();
    void createP();
};


#endif