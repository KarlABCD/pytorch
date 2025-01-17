#include "ReforceLearning.h"

int main()
{
    //CWalkEnv env = CWalkEnv(12, 4);
    //env.PrintValue();
    PolicyIteration agent = PolicyIteration(CWalkEnv(12, 4), 12, 4);
    agent.PrintEnvValues();
    agent.PrintPolicy();

    return 0;
}