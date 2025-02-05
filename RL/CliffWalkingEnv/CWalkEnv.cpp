#include "CWalkEnv.h"
#include <iostream>

using namespace std;

CWalkEnv::CWalkEnv()
{
    cout << "CWalkEnv Default Initialization" << endl;
};

CWalkEnv::CWalkEnv(uint16 nRow, uint16 nCol)
{
    cout << "CWalkEnv nCol nRow Initialization" << endl;
    u16Rows = nRow;
    u16Cols = nCol;
    x = 0U;
    y = nRow - 1;
    if (u16Cols * u16Rows < MAX_UINT16_VALUE)
        u16NumStates = u16Cols * u16Rows;
    else
        cout << "超过限制" << endl;
    CreateP();
}

CWalkEnv::~CWalkEnv()
{
    cout << "CWalk Destory" << endl;
}

void CWalkEnv::CreateP()
{
    P = vector<vector<tActionInfo>>(u16NumStates, 
                                    vector<tActionInfo>(4, {0, 0, 0, false}));
    vector<vector<int16>> change = {{0, int16(-1)}, {0, 1}, 
                                    {int16(-1), 0}, {1, 0}};
    cout << "环境模型: "<< endl;
    for (int16 i = 0; i < (int16)u16Rows; i++)
    {
        for (int16 j = 0; j < (int16)u16Cols; j++)
        {
            for (int16 a = 0; a < 4; a++)
            {
                if (i == u16Rows - 1 && j > 0)
                {
                    P[i * u16Cols + j][a].Possibility = 1.0F;
                    P[i * u16Cols + j][a].Reward = 0.0F;
                    P[i * u16Cols + j][a].u16NextState = i * u16Cols + j;
                    P[i * u16Cols + j][a].bEpisodeDone = true;
                }
                else
                {
                    int16 next_x = min(int16(u16Cols - 1), 
                                    max(int16(0), (int16)(j + change[a][0])));
                    int16 next_y = min(int16(u16Rows - 1),
                                    max(int16(0), (int16)(i + change[a][1])));
                    int16 next_state = next_y * u16Cols + next_x;
                    float32 reward = -1.0F;
                    boolean bEpisodeDone = false;
                    if (next_y == u16Rows - 1 && next_x > 0)
                    {
                        bEpisodeDone = true;
                        if(next_x != u16Cols - 1)
                        {
                            reward = -100.0F; 
                        }
                    }   
                    P[i * u16Cols + j][a].Possibility = 1.0F;
                    P[i * u16Cols + j][a].Reward = reward;
                    P[i * u16Cols + j][a].u16NextState = uint16(next_state);
                    P[i * u16Cols + j][a].bEpisodeDone = bEpisodeDone;
                }
                //cout << "行数: " << i << " 列数: " << j << endl;
                //cout << "Possibility: " << P[i * u16Cols + j][a].Possibility;
                //cout << " Reward: " << P[i * u16Cols + j][a].Reward;
                //cout << " u16NextState: " << P[i * u16Cols + j][a].u16NextState;
                //cout << " bEpisodeDone: " << P[i * u16Cols + j][a].bEpisodeDone;
                //cout << endl;
            }
        }
    }
};

void CWalkEnv::PrintEnvValue() const
{
    cout << "悬崖行数: " << u16Rows << endl;
    cout << "悬崖列数: " << u16Cols << endl;
    cout << "状态总数: " << u16NumStates << endl;
};

void CWalkEnv::Reset(uint16 & CurState)
{
    x = 0;
    y = int16(u16Rows) - 1;
    CurState = y * u16Cols + x;
};

void CWalkEnv::Step(uint16 const Action, uint16 & next_state, float32 & Reward, 
                    boolean & bEpisodeDone)
{
    vector<vector<int16>> change = {{0, int16(-1)}, {0, 1}, 
                                    {int16(-1), 0}, {1, 0}};
    x = min(int16(u16Cols - 1), max(int16(0), int16(x + change[Action][0])));
    y = min(int16(u16Rows - 1), max(int16(0), int16(y + change[Action][1])));
    next_state = uint16(y) * u16Cols + uint16(x);
    Reward = -1.0F;
    bEpisodeDone = false;
    if ( y == int16(u16Rows - 1) && x > 0)
    {
        bEpisodeDone = true;
        if (x != u16Cols - 1)
        {
            Reward = -100;
        }
    }
}