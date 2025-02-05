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
    float32 episode_return;
    boolean bEpisodeDone;

    for (uint16 i = 0; i < num_episodes; i++)
    {
        bEpisodeDone = false;
        env.Reset(state);
        episode_return = 0.0F;
        while (!bEpisodeDone)
        {
            uint16 next_state = 0;
            float32 reward = 0.0F;
            //boolean bEpisodeDone;
            uint16 action = 0;
            TakeAction(state, action);
            env.Step(action, next_state, reward, bEpisodeDone);
            episode_return += reward;
            Update(state, action, reward, next_state);
            state = next_state;
        }
        return_list[i] = episode_return;
        cout << "次数: " << i << "episode_return: " << episode_return<< endl;
        //PrintWholeQTable();
    }
    PrintAgent();
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
        cout << "CurState: " << state << " ";
        cout << "Take Action" << endl;
        PrintQTableValue(state);
    }
}

void QLearning::Update(uint16 const state, uint16 const action, 
                    float32 const reward, uint16 const next_state)
{
    float32 max;
    VectorMax(&Q_table[next_state], max);
    cout << "before Update" << endl;
    PrintQTableValue(state);
    //PrintWholeQTable();
    float32 td_error = reward + gamma * max - Q_table[state][action];
    Q_table[state][action] += alpha * td_error;
    cout << "after Update" << endl;
    PrintQTableValue(state);
    //PrintWholeQTable();
}

void QLearning::PrintQTableValue(uint16 u16ExpState) const
{
    for(float32 StateValue : Q_table[u16ExpState])
    {
        cout << StateValue << " ";
    }
    cout << endl;
}

void QLearning::PrintWholeQTable() const
{
    for (uint16 i = 0; i < Q_table.size(); i++)
    {
        cout << "state " << i;
        for( uint16 j = 0; j < Q_table[0].size(); j++)
        {
            cout << " " << Q_table[i][j] << " ";
        }
        cout << endl;
    }
}

void QLearning::PrintAgent()
{
    cout << "策略: " << endl;
    string action_meaning[4] = {"^", "v", "<", ">"};
    vector<float32> a;
    
    for(uint8 i = 0; i < env.GetRows(); i++)
    {
        for(uint8 j = 0; j < env.GetCols(); j++)
        {
            if ( i*env.GetCols() + j >= 37 && i*env.GetCols() + j <= 46)
            {
                cout << "****" << " ";
            }
            else if (i*env.GetCols() + j == 47)
            {
                cout << "EEEE" << " ";
            }
            else
            {
                a = Q_table[i*env.GetCols() + j];
                string pi_str = {};
                float32 maxa;
                VectorMax(&a, maxa);
                for (uint8 k = 0; k < 4; k++)
                {
                    if(fabs(a[k] - maxa) < FlOAT32_EQ_TOLERANCE)
                    {
                        pi_str += action_meaning[k];
                    }
                    else
                    {
                        pi_str += "o";
                    }
                }
                cout << pi_str << " ";
            }
        }
        cout << endl;
    }
}