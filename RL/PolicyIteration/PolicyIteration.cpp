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
    iEnvRows = int16(u16Row);
    iEnvCols = int16(u16Col);
    pi = vector<vector<float32>>(u16Row * u16Col, vector<float32>(4, 0.25));
    createP();
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
    for (int16 i = 0; i < pi.size(); i++)
    {
        cout << "第" << floor(i / iEnvCols) << "行";
        cout << "第" << i % iEnvRows << "列的策略：" << endl;
        cout << "向上概率: " << pi[i][0] << endl;
        cout << "向下概率: " << pi[i][1] << endl;
        cout << "向左概率: " << pi[i][2] << endl;
        cout << "向右概率: " << pi[i][3] << endl;
    }
}

void PolicyIteration::PrintActionValues()
{
    for(int16 i = 0; i < P.size(); i++)
    {
        cout << "状态在第" << floor(i / iEnvCols) <<"行";
        cout << "第"<<i % iEnvCols << "列 : " << endl;
        cout << "向上 Reward: " <<P[i][0].Reward << " ";
        cout << "下一个状态 State: "<<P[i][0].u16NextState<< " " << endl;
        cout << "向下 Reward: " <<P[i][1].Reward << " ";
        cout << "下一个状态 State: "<<P[i][1].u16NextState<< " " << endl;
        cout << "向左 Reward: " <<P[i][2].Reward << " ";
        cout << "下一个状态 State: "<<P[i][2].u16NextState<< " " << endl; 
        cout << "向右 Reward: " <<P[i][3].Reward << " ";
        cout << "下一个状态 State: "<<P[i][3].u16NextState<< " " << endl;
    }
}

void PolicyIteration::createP()
{
    P = vector<vector<tActionInfo>>(iEnvRows * iEnvCols, 
                                    vector<tActionInfo>(4, {0, 0, 0, false}));
    vector<vector<int16>> change = {{0, int16(-1)}, {0, 1}, {int16(-1), 0}, {1, 0}};
    for (int16 i = 0; i < (int16)iEnvRows; i++)
    {
        for (int16 j = 0; j < (int16)iEnvCols; j++)
        {
            for (int16 a = 0; a < 4; a++)
            {
                if (i == iEnvRows - 1 && j > 0)
                {
                    P[i * iEnvCols + j][a].Possibility = 1.0F;
                    P[i * iEnvCols + j][a].Reward = 0.0F;
                    P[i * iEnvCols + j][a].u16NextState = i * iEnvCols + j;
                    P[i * iEnvCols + j][a].bEpisodeDone = true;
                    cout << P[i * iEnvCols + j][a].Reward << endl;
                    cout << i * iEnvCols + j << endl;
                    cout << a << endl;
                }
                else
                {
                    int16 next_x = min(int16(iEnvCols - 1), 
                                        max(int16(0), (int16)(j + change[a][0])));
                    int16 next_y = min(int16(iEnvRows - 1),
                                        max(int16(0), (int16)(i + change[a][1])));
                    int16 next_state = next_y * iEnvCols + next_x;
                    float32 reward = -1.0F;
                    boolean bEpisodeDone = false;
                    if (next_y == iEnvRows - 1 && next_x > 0)
                    {
                        bEpisodeDone = true;
                        if(next_x != iEnvCols - 1)
                        {
                            reward = -100.0F; 
                        }
                    }   
                    P[i * iEnvCols + j][a].Possibility = 1.0F;
                    P[i * iEnvCols + j][a].Reward = reward;
                    P[i * iEnvCols + j][a].u16NextState = uint16(next_state);
                    P[i * iEnvCols + j][a].bEpisodeDone = bEpisodeDone;
                }
            }
        }
    }
    //PrintActionValues();

}