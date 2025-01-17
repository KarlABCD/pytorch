#include "PolicyIteration.h"
using std::vector;
using std::cout;
using std::endl;

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
        v.resize(u16Row * u16Col);
    env = ENV;

    for (uint8 i; i < 4; i++)
    {
        pi[i] = 0.25;
    }
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
    cout << "向上概率: " << pi[0] << endl;
    cout << "向下概率: " << pi[1] << endl;
    cout << "向左概率: " << pi[2] << endl;
    cout << "向右概率: " << pi[3] << endl;
}