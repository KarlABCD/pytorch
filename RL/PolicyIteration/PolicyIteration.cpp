#include "PolicyIteration.h"
using namespace std;

PolicyIteration::PolicyIteration()
{
    cout << "PolicyIteration Default Init" << endl;
};

PolicyIteration::PolicyIteration(CWalkEnv ENV, uint16 u16Row, uint16 u16Col)
{
    cout << "PolicyIteration Row Col Init" << endl;
    if (u16Row * u16Col > MAX_UINT16_VALUE)
        cout << "input error" <<endl;
    else
        v = vector<float32>(u16Row * u16Col, 0);
    env = ENV;
    pi = vector<vector<float32>>(u16Row * u16Col, vector<float32>(4, 0.25));
    P = vector<vector<float32>>(u16Row * u16Col, vector<float32>(4, 0));
    u16EnvRows = u16Row;
    u16EnvCols = u16Col;
};

PolicyIteration::~PolicyIteration()
{
    cout << "PolicyInteration Destory" << endl;
};

void PolicyIteration::PrintStateValues()
{
    for (uint8 i = 0; i < v.size(); i++ )
    {
        cout << "v: " << v[i] << endl;
    }
};

void PolicyIteration::PrintEnvValues()
{
    env.PrintValue();
};

void PolicyIteration::PrintPolicy()
{
    for (size_t i = 0; i < pi.size(); i++)
    {
        cout << "第" << floor(i / u16EnvCols) << "行";
        cout << "第" << i % u16EnvRows << "列的策略：" << endl;
        cout << "向上概率: " << pi[i][0] << endl;
        cout << "向下概率: " << pi[i][1] << endl;
        cout << "向左概率: " << pi[i][2] << endl;
        cout << "向右概率: " << pi[i][3] << endl;
    }
}

void PolicyIteration::PrintActionValues()
{
    for(size_t i = 0; i < P.size(); i++)
    {
        cout << "状态在第" << floor(i / u16EnvCols) <<"行";
        cout << "第"<<i % u16EnvCols << "列: ";
        for(size_t j = 0; j < P[0].size(); j++)
        {
            cout << P[i][j] << " ";
        }
        cout << endl;
    }
}