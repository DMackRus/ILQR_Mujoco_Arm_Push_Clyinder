//
// Created by david on 29/04/22.
//

#ifndef MUJOCO_ACROBOT_CONTROL_BOXPUSHBOXMODEL_H
#define MUJOCO_ACROBOT_CONTROL_BOXPUSHBOXMODEL_H

#include "mujoco.h"
#include "../Utility/stdInclude/stdInclude.h"
#include "../Utility/MujocoController/MujocoController.h"

;class boxModel{
public:
    boxModel(mjModel* m, m_state _desiredState);
    int degreesOfFreedom = 4;
    int numberControls = 2;

    float controlCost[NUM_CTRL] = {0, 0};
    float stateCosts[(2 * DOF)] = {0, 0, 1, 1, 0, 0.1, 0.1, 0.1, 0.1, 0};

    mjModel* model;
    m_state X_desired;
    m_ctrl_ctrl R;
    m_state_state Q;

    int terminalConstant = 10;
    float heightFloat = 0.1;

    std::vector<std::string> stateNames;
    int stateIndexToStateName[DOF] = {0, 0, 1, 1, 1};

    // Given a set of mujoco data, what is the cost of its state and controls
    float getCost(mjData *d, int controlNum, int totalControls);
    float costFunction(int controlNum, int totalControls, m_state X, m_ctrl U);

    // Given a set of mujoco data, what are its cost derivates with respect to state and control
    void costDerivatives(mjData *d, Ref<m_state> l_x, Ref<m_state_state> l_xx, Ref<m_ctrl> l_u, Ref<m_ctrl_ctrl> l_uu, int controlNum, int totalControls);
    void costDerivatives_fd(mjData *d, Ref<m_state> l_x, Ref<m_state_state> l_xx, Ref<m_ctrl> l_u, Ref<m_ctrl_ctrl> l_uu, int controlNum, int totalControls);
    m_state costDerivatives_fd_1stOrder(m_state X, m_ctrl U, int controlNum, int totalControls);

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

#endif //MUJOCO_ACROBOT_CONTROL_BOXPUSHBOXMODEL_H
