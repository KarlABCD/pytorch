#include "PolicyIteration.h"
using namespace::std;

tReturnType PolicyEqual(vector<vector<float32>> const * old_pi_ptr, 
                        vector<vector<float32>> const * pi_ptr, boolean & bPolicyEqual)
{
    bPolicyEqual = true;
    if (old_pi_ptr == nullptr || pi_ptr == nullptr 
        || (old_pi_ptr->size() != pi_ptr->size()))
    {
        bPolicyEqual = false;
        return FuncAbNormal;
    }
    else
    {
        for(uint16 i = 0; i < old_pi_ptr->size(); i++)
        {
            if((*old_pi_ptr)[i].size() != (*pi_ptr)[i].size())
            {
                bPolicyEqual = false;
                return FuncAbNormal;
            }
            else
            {
                for(uint16 j = 0; j < (*old_pi_ptr)[i].size(); j++)
                {
                    if(fabs((*old_pi_ptr)[i][j] - (*pi_ptr)[i][j]) 
                                    > FlOAT32_EQ_TOLERANCE)
                    {
                        bPolicyEqual = false;
                        return FuncNormal;
                    }
                }
            }
        }
        return FuncNormal;
    }
}

PolicyIteration::PolicyIteration()
{
    cout << "PolicyIteration Default Init" << endl;
};

PolicyIteration::PolicyIteration(uint16 u16Row, uint16 u16Col, 
                                    float32 theta, float32 gamma)
{
    cout << "PolicyIteration Row Col Init" << endl;
    env = CWalkEnv(u16Row, u16Col);
    if (env.GetRows() * env.GetCols() > MAX_UINT16_VALUE)
        cout << "input error" <<endl;
    else
        v = vector<float32>(env.GetNumStates(), 0);
    pi = vector<vector<float32>>(env.GetNumStates(), vector<float32>(4, 0.25));
    Potheta = theta;
    Pogamma = gamma;
};

PolicyIteration::~PolicyIteration()
{
    cout << "PolicyInteration Destory" << endl;
};

void PolicyIteration::PrintStateValues() const
{
    cout << endl;
    cout << "状态价值: " << endl;
    for (uint8 i = 0; i < env.GetRows(); i++ )
    {
        for (uint8 j = 0; j < env.GetCols(); j++)
        {
            cout << v[i*env.GetCols() + j] << " ";
        }
        cout << endl;
    }
    
};

void PolicyIteration::PrintPolicy() const
{
    for (int16 i = 0; i < pi.size(); i++)
    {
        cout << "第" << floor(i / env.GetCols()) << "行";
        cout << "第" << i % env.GetRows() << "列的策略：" << endl;
        cout << "向上概率: " << pi[i][0] << endl;
        cout << "向下概率: " << pi[i][1] << endl;
        cout << "向左概率: " << pi[i][2] << endl;
        cout << "向右概率: " << pi[i][3] << endl;
    }
}

void PolicyIteration::PrintActionValues() const
{
    for(int16 i = 0; i < env.GetNumStates(); i++)
    {
        cout << "状态在第" << floor(i / env.GetCols()) <<"行";
        cout << "第"<<i % env.GetCols() << "列 : " << endl;
        cout << "向上 Reward: " <<env.GetP(i, 0).Reward << " ";
        cout << "下一个状态 State: "<<env.GetP(i, 0).u16NextState<< " " << endl;
        cout << "向下 Reward: " <<env.GetP(i, 1).Reward << " ";
        cout << "下一个状态 State: "<<env.GetP(i, 1).u16NextState<< " " << endl;
        cout << "向左 Reward: " <<env.GetP(i, 2).Reward << " ";
        cout << "下一个状态 State: "<<env.GetP(i, 2).u16NextState<< " " << endl; 
        cout << "向右 Reward: " <<env.GetP(i, 3).Reward << " ";
        cout << "下一个状态 State: "<<env.GetP(i, 3).u16NextState<< " " << endl;
    }
}

void PolicyIteration::PolicyEvaluation()
{
    uint8 cnt = 1;
    vector<float32> qsa[4];
    while (true)
    {
        float32 max_diff = 0.0F;
        vector<float32> new_v = vector<float32>(env.GetNumStates(), 0);
        for (uint16 i = 0; i < env.GetNumStates(); i++)
        {
            vector<float32> qsa = vector<float32>(4, 0);
            for(uint16 j = 0; j < 4; j++)
            {
                if (~env.GetP(i, j).bEpisodeDone)
                {
                    qsa[j] = env.GetP(i, j).Possibility * 
                    (env.GetP(i, j).Reward 
                    + Pogamma * v[env.GetP(i, j).u16NextState]);
                }
                else
                {
                    qsa[j] = env.GetP(i, j).Possibility * env.GetP(i, j).Reward;
                }
                qsa[j] = qsa[j] * pi[i][j];
                //PrintPolicy();
            }
            VectorSum(&qsa, new_v[i]);
            max_diff = max(max_diff, fabs(new_v[i] - v[i]));
        }
        v = new_v;
        if (max_diff < Potheta)
            break;
        
        cnt += 1;
    }
     printf("策略评估进行%d轮后完成\n", cnt);
     PrintAgent();
}

void PolicyIteration::PolicyImprovement()
{   
    float32 maxq;
    for(uint16 s = 0; s < env.GetNumStates(); s++)
    {
        vector<float32> qsa = vector<float32>(4, 0);
        uint16 u16cntq;
        for(uint16 a = 0; a < 4; a++)
        {
            if(~env.GetP(s, a).bEpisodeDone)
            {
                qsa[a] = env.GetP(s,a).Possibility * ((env.GetP(s,a).Reward) 
                        + v[env.GetP(s,a).u16NextState] * Pogamma);
            }
            else
            {
                qsa[a] = env.GetP(s,a).Possibility * (env.GetP(s,a).Reward);
            }
        }
        VectorMax(&qsa, maxq);
        VectorCountEq(&qsa, maxq, u16cntq);
        for (uint16 i = 0; i < 4; i++)
        {
            if(fabs(qsa[i] - maxq) < FlOAT32_EQ_TOLERANCE)
                pi[s][i] = 1 / (float32)u16cntq;
            else
                pi[s][i] = 0.0F;
        }
    }
}


void PolicyIteration::PolicyIterationMain()
{
    boolean bPolicyEq;
    vector<vector<float32>> old_pi;
    while(true)
    {
        PolicyEvaluation();
        old_pi = pi;
        PolicyImprovement();
        PolicyEqual(&old_pi, &pi, bPolicyEq);
        if (bPolicyEq)
            break;
    }
}

void PolicyIteration::PrintAgent()
{
    PrintStateValues();
    cout << "策略: " << endl;
    string action_meaning[4] = {"^", "v", "<", ">"};
    
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
                vector<float32> a = pi[i*env.GetCols() + j];
                string pi_str = {};
                for (uint8 k = 0; k < 4; k++)
                {
                    if(a[k] > 0)
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