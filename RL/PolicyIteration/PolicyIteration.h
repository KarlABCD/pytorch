#ifndef POLICYITERATION_H_
#define POLICYITERATION_H_

#include <vector>
#include <array>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cstring>
#include "DataType.h"
#include "CWalkEnv.h"
#include "VectorCalc.h"
#include "StringOper.h"

using namespace::std;

class PolicyIteration
{
private:

    vector<float32> v;
    vector<vector<float32>> pi;
    float32 Potheta;
    float32 Pogamma;

public:
    CWalkEnv env;
    PolicyIteration();
    PolicyIteration(uint16 u16Col, uint16 u16Row, float32 theta, float32 gamma);
    ~PolicyIteration();
    void PrintStateValues() const;
    void PrintPolicy() const;
    void PrintActionValues() const;
    void PolicyEvaluation();
    void PolicyIterationMain();
    void PolicyImprovement();
    void PrintAgent();
};


#endif