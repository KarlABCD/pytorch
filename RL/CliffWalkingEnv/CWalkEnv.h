#pragma once
#ifndef CLIFFWALKINGEVN_H_
#define CLIFFWALKINGEVH_H_

#include "DataType.h"
#include <cmath>
#include <algorithm>

struct tActionInfo
{
    float32 Possibility;
    float32 Reward;
    uint16 u16NextState;
    boolean bEpisodeDone;
};

class CWalkEnv
{
private:
    uint16 u16Cols;
    uint16 u16Rows;
    uint16 u16NumStates;
    int16 x;
    int16 y;
    vector<vector<tActionInfo>> P;
public:
    CWalkEnv();
    CWalkEnv(uint16 nCol, uint16 nRow);
    ~CWalkEnv();
    void PrintEnvValue() const;
    void CreateP();
    void Reset(uint16 & CurState);
    void Step(uint16 const Action, uint16 & next_state, float32 & Reward, 
                    boolean & bEpisodeDone);
    inline uint16 GetCols() const {return u16Cols;}
    inline uint16 GetRows() const {return u16Rows;}
    inline uint16 GetNumStates() const {return u16NumStates;}
    inline tActionInfo GetP(uint16 x, uint16 y) const{return P[x][y];}
};
#endif