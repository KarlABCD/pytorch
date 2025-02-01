#include "ReforceLearning.h"
//#include <gtest/gtest.h>

int main()
{

/*  Polciy Interation算法
    float32 theta = 0.1;
    float32 gamma = 0.9;
    PolicyIteration agent = PolicyIteration(4, 12, theta, gamma);
    agent.PolicyIterationMain();
*/
    float32 epsilon = 0.1F;
    float32 alpha = 0.1F;
    float32 gamma = 0.9F;
    uint16 num_episodes = 500U;
    QLearning agent = QLearning(12, 4, epsilon, alpha, gamma, num_episodes);
    agent.QLearningMain();
    return 0;
}