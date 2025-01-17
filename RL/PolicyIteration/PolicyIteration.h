#ifndef POLICYITERATION_H_
#define POLICYITERATION_H_

#include <vector>
#include <iostream>
#include "DataType.h"
#include "CWalkEnv.h"

using std::vector;

class PolicyIteration
{
private:

    vector<float32> v;
    vector<float32> pi;
    CWalkEnv env;

public:
    
    PolicyIteration();
    PolicyIteration(CWalkEnv ENV, uint16 u16Row, uint16 u16Col);
    ~PolicyIteration();
    void PrintStateValues();
    void PrintEnvValues();
    void PrintPolicy();
};


#endif