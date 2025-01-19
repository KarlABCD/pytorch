#include "ReforceLearning.h"
#include <gtest/gtest.h>

int main()
{
    //CWalkEnv env = CWalkEnv(12, 4);
    //env.PrintValue();
    PolicyIteration agent = PolicyIteration(CWalkEnv(4, 12), 4, 12);
//   agent.PrintEnvValues();
//    agent.PrintPolicy();
    agent.PrintActionValues();

    return 0;
}