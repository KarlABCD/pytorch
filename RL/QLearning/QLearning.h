#include "DataType.h"

class QLearning
{
private:
    vector<vector<float32>> Q_table;
    uint16 n_action;
    float32 alpha;
    float32 gamma;
    float32 epsilon;

public:
    QLearning();
    QLearning(uint16 const nEnvrow, uint16 const nEnvcol, 
                float32 const Inepsilon, float32 const Inalpha, 
                float32 const Ingamma, uint16 const n_action = 4)
                : alpha(Inalpha), epsilon(Inepsilon), gamma(Ingamma), 
                n_action(n_action)
                {
                    Q_table = vector<vector<float32>>(nEnvcol*nEnvrow, 
                    vector<float32>(n_action, 0));
                }
    ~QLearning();

    void QLearningMain();
    void TakeAction();
    void BestAction();
    void Update();
};