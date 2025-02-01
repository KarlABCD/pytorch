#include "QLearning.h"
using namespace std;

QLearning::QLearning()
{
    cout << "QLearning Default Init" << endl;
}

QLearning::~QLearning()
{
    cout << "QLearing Default Destroy" << endl;
};

void QLearning::QLearningMain()
{
    uint16 state;
    float32 episode_return = 0.0F;
    env.Reset(state);
    boolean bEpisodeDone = false;

    while (~bEpisodeDone)
    {
        uint16 next_state;
        float32 reward;
        //boolean bEpisodeDone;
        uint16 action;
        TakeAction(state, action);
        env.Step(action, next_state, reward, bEpisodeDone);
        episode_return += reward;
        Update(state, action, reward, next_state);
        state = next_state;
    }
}

void QLearning::TakeAction(uint16 state, uint16 & action)
{
    float random_value = rand()/(RAND_MAX + 1.0F);
    if (random_value < epsilon)
    {
        action = uint16(rand() % n_action);
    }
    else
    {
        VectorMaxIdx(&Q_table[state], action);
    }
}

void QLearning::Update(uint16 const state, uint16 const action, 
                        float32 const reward, uint16 const next_state)
{
    float32 max;
    VectorMax(&Q_table[next_state], max);
    float32 td_error = reward + gamma * max - Q_table[state][action];
    Q_table[state][action] = alpha * td_error;
}