#include "CWalkEnv.h"
#include <iostream>

using namespace std;

CWalkEnv::CWalkEnv()
{
    cout << "CWalkEnv Default Initialization" << endl;
};

CWalkEnv::CWalkEnv(uint16 nCol, uint16 nRow)
{
    cout << "CWalkEnv nCol nRow Initialization" << endl;
    u16NCol = nCol;
    u16NRow = nRow;
    if (u16NCol*u16NRow < MAX_UINT16_VALUE)
        u16NumStates = u16NCol*u16NRow;
    else
        cout << "超过限制" << endl;
}

CWalkEnv::~CWalkEnv()
{
    cout << "CWalk Destory" << endl;
}

void CWalkEnv::CreateP()
{
    cout << "完成Action Value矩阵" << endl;
};

void CWalkEnv::PrintValue()
{
    cout << "悬崖行数: " << u16NRow << endl;
    cout << "悬崖列数: " << u16NCol << endl;
    cout << "状态总数: " << u16NumStates << endl;
};