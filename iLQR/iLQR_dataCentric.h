//
// Created by david on 13/04/2022.
//

#ifndef MUJOCO_ACROBOT_CONTROL_ILQR_DATACENTRIC_H
#define MUJOCO_ACROBOT_CONTROL_ILQR_DATACENTRIC_H

#include "mujoco.h"
#include "glfw3.h"
#include "../Utility/stdInclude/stdInclude.h"
//#include "../model/acrobotModel.h"
//#include "../model/boxPushBoxModel.h"
#include "../model/frankaArmAndBox.h"
#include "../Utility/MujocoController/MujocoUI.h"

#define MUJOCO_DT 0.004
//#define ILQR_DT 0.02
//#define NUM_MJSTEPS_PER_CONTROL 10
//#define ILQR_HORIZON_LENGTH 500
#define MIN_STEPS_PER_CONTROL 2
#define NUM_DATA_STRUCTURES 1250
#define NUM_SCALING_LEVELS  1
#define MUJ_STEPS_HORIZON_LENGTH 2500
#define EPS_GRAD_NEW_START_POINT 0.2

;
//template<int HORIZON_LENGTH>
class iLQR
{
    public:

    // constructor - mujoco model, data, initial controls and initial state
    iLQR(mjModel* m, mjData* d, m_state _X0, frankaModel* _modelTranslator, MujocoController* _mujocoController);

    /*      Data     */
    // MuJoCo model and data
    mjModel* model;
    mjData* mdata = NULL;
    frankaModel *modelTranslator;
    MujocoController *mujocoController;

    // Array of mujoco data structure along the trajectory
    mjData* dArray[NUM_DATA_STRUCTURES + 1];
    // Mujoco data for the initial state of the system
    mjData* d_init;
    mjData* d_current_start;
    m_state X0;

    /**************************************************************************
     *
     *  iLQR Parameters
     *
     *
     */
    float maxLamda = 10000;             // Maximum lambda before canceliing optimisation
    float minLamda = 0.00001;           // Minimum lamda
    float lamdaFactor = 10;             // Lamda multiplicative factor
    float epsConverge = 0.01;          // Satisfactory convergence of cost function 0.01
    int maxIterations = 20;

    int startingTimeIndex = 0;
    double startCostFromStartIndex = 0.0f;
    int scalingLevelCount = 0;
    int scalingLevel[NUM_SCALING_LEVELS] = {10};
    // these need to be integer multiples of one another
    int num_mj_steps_per_dynamics_deriv = scalingLevel[0];
    int num_mj_steps_per_control_deriv = 20;
    int ilqr_horizon_length = MUJ_STEPS_HORIZON_LENGTH / num_mj_steps_per_dynamics_deriv;

    std::vector<std::vector<double>> cumulativeCosts;

    std::vector<m_ctrl> initControls;
    std::vector<m_ctrl> finalControls;

    // Initialise partial differentiation matrices for all timesteps T
    // for linearised dynamics
    std::vector<m_state_state> f_x;
    std::vector<m_state_ctrl> f_u;

    std::vector<m_state_state> A_scaled;
    std::vector<m_state_ctrl> B_scaled;
    std::vector<m_state_state> A;
    std::vector<m_state_ctrl> B;

    // Quadratic cost partial derivatives
    std::vector<m_state> l_x;
    std::vector<m_state_state> l_xx;
    std::vector<m_ctrl> l_u;
    std::vector<m_ctrl_ctrl> l_uu;

    std::vector<m_state> l_x_o;
    std::vector<m_state_state> l_xx_o;
    std::vector<m_ctrl> l_u_o;
    std::vector<m_ctrl_ctrl> l_uu_o;

    // Initialise state feedback gain matrices
    std::vector<m_ctrl> k;
    std::vector<m_ctrl_state> K;

    // Initialise new controls and states storage for evaluation
    std::vector<m_ctrl> U_new;
    std::vector<m_ctrl> U_old;
    std::vector<m_state> X_final;
    //std::vector<m_state> X_old;

    float lamda = 0.1;
    int numIterations = 0;

    void optimise();
    double rollOutTrajectory();

    void getDerivatives();
    void copyDerivatives();
    void scaleLinearisation(Ref<m_state_state> A_scaled, Ref<m_state_ctrl> B_scaled, Ref<m_state_state> A, Ref<m_state_ctrl> B, int num_steps_per_dt);

    bool backwardsPass_Quu_reg();
    bool backwardsPass_Vxx_reg();
    bool isMatrixPD(Ref<MatrixXd> matrix);

    double forwardsPass(float oldCost);

    bool checkForConvergence(float newCost, float oldCost);
    int checkCostReductionForNewStartingPoint();

    bool updateScaling();
    void updateDataStructures();

    void lineariseDynamicsSerial_trial_step(Ref<MatrixXd> _A, Ref<MatrixXd> _B, mjData *linearisedData, float dt, Ref<m_state> l_x, Ref<m_state_state> l_xx, int controlNum, int totalControls, bool computeCost);

    m_ctrl returnDesiredControl(int controlIndex, bool finalControl);
    void setInitControls(std::vector<m_ctrl> _initControls);
    void makeDataForOptimisation();



};

#endif //MUJOCO_ACROBOT_CONTROL_ILQR_DATACENTRIC_H
