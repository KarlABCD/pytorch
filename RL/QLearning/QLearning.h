#ifndef QLEARNING_H_
#define QLEARNING_H_
#include "DataType.h"
#include "CWalkEnv.h"
#include "VectorCalc.h"
#include <iostream>
#include <cstdlib>

class QLearning
{
private:
    vector<vector<float32>> Q_table;
    uint16 n_action;
    float32 alpha;
    float32 gamma;
    float32 epsilon;
    uint16 num_episodes;

public:
    CWalkEnv env;
    QLearning();
    QLearning(uint16 const nEnvrow, uint16 const nEnvcol, 
                float32 const Inepsilon, float32 const Inalpha, 
                float32 const Ingamma, uint16 const num_episodes,
                uint16 const n_action = 4)
                : alpha(Inalpha), epsilon(Inepsilon), gamma(Ingamma), 
                n_action(n_action), num_episodes(num_episodes)
                {
                    Q_table = vector<vector<float32>>(nEnvcol*nEnvrow, 
                    vector<float32>(n_action, 0));
                    env = CWalkEnv(nEnvcol, nEnvrow);
                }
    ~QLearning();

    void QLearningMain();
    void TakeAction(uint16 state, uint16 & action);
    void BestAction();
    void Update(uint16 const state, uint16 const action, 
                float32 const reward, uint16 const next_state);
};

#endif