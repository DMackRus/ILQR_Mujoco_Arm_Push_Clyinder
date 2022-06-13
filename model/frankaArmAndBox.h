//
// Created by david on 03/05/22.
//

#ifndef MUJOCO_ACROBOT_CONTROL_FRANKAARMANDBOX_H
#define MUJOCO_ACROBOT_CONTROL_FRANKAARMANDBOX_H

#include "mujoco.h"
#include "../Utility/stdInclude/stdInclude.h"
#include "../Utility/MujocoController/MujocoController.h"

;class frankaModel{
public:
    frankaModel(mjModel *m, m_state _desiredState);

    // 20 * 20 (Torques) = 400, 400 * 0.001 = 0.4 (cost for indivudal control)
    // Compared to difference in cube state - 0.3 m * 0.3m = 0.09, 0.09 * 2 = 0.18 (cost for individual state difference)
    //float controlCost[NUM_CTRL] = {0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005};
    float controlCost[NUM_CTRL] = {0, 0, 0, 0, 0, 0.0005, 0.001};

    // State vector is: 7 joint angles, two cube pos (X and Y), cube rot, 7 joint velocities, two cube velocities (X and Y)
    float stateCosts[(2 * DOF)] = {0, 0, 0, 0, 0, 0, 0,
                                    1, 1, 0,
                                    0, 0, 0, 0, 0,0, 0,
                                    0.2, 0.2, 0.01};

//    float stateCosts[(2 * DOF)] = {0, 0, 0, 0, 0, 0, 0,
//                                   10, 0.5,
//                                   0, 0, 0, 0, 0,0, 0,
//                                   0.1, 0.1};

    int terminalConstant = 10;

    std::vector<std::string> stateNames;
    int stateIndexToStateName[DOF] = {0, 1, 2, 3, 4, 5, 6, 7, 7, 7};

    float torqueLims[NUM_CTRL] = {87, 87, 87, 87, 12, 12, 12};

    mjModel* model;
    m_state X_desired;
    DiagonalMatrix<double, NUM_CTRL> R;
    DiagonalMatrix<double, 2 * DOF> Q;
//    m_ctrl_ctrl R;
//    m_state_state Q;

    float getCost(mjData *d, m_ctrl lastControl, int controlNum, int totalControls, bool firstControl);
    // Given a set of mujoco data, what is the cost of its state and controls
    float costFunction(int controlNum, int totalControls, m_state X, m_ctrl U, m_ctrl lastControl, bool firstControl);

    // Given a set of mujoco data, what are its cost derivates with respect to state and control
    void costDerivatives(mjData *d, Ref<m_state> l_x, Ref<m_state_state> l_xx, Ref<m_ctrl> l_u, Ref<m_ctrl_ctrl> l_uu, int controlNum, int totalControls);
    void costDerivatives_fd(mjData *d, Ref<m_state> l_x, Ref<m_state_state> l_xx, Ref<m_ctrl> l_u, Ref<m_ctrl_ctrl> l_uu, int controlNum, int totalControls, m_ctrl U_last,  bool firstControl);
    m_ctrl costDerivatives_fd_1stOrder(m_state X, m_ctrl U, m_ctrl U_last, int controlNum, int totalControls, bool firstControl);

    // set the state of a mujoco data object as per this model
    void setState(mjData *d, m_state X);

    // Return the state of a mujoco data model
    m_state returnState(mjData *d);

    // Set the controls of a mujoco data object
    void setControls(mjData *d, m_ctrl U);

    // Return the controls of a mujoco data object
    m_ctrl returnControls(mjData *d);

    m_dof returnPositions(mjData *d);
    m_dof returnVelocities(mjData *d);
    m_dof returnAccelerations(mjData *d);

    void perturbVelocity(mjData *perturbedData, mjData *origData, int stateIndex, double eps);

    void perturbPosition(mjData *perturbedData, mjData *origData, int stateIndex, double eps);

    void stepModel(mjData *d, int numSteps);


};

#endif //MUJOCO_ACROBOT_CONTROL_FRANKAARMANDBOX_H
