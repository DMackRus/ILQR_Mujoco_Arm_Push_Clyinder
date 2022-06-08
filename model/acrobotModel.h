//
// Created by david on 27/04/22.
//

#ifndef MUJOCO_ACROBOT_CONTROL_ACROBOTMODEL_H
#define MUJOCO_ACROBOT_CONTROL_ACROBOTMODEL_H

#include "mujoco.h"
#include "../Utility/stdInclude/stdInclude.h"

;class acrobotModel{
public:
    acrobotModel(m_state _desiredState);
    int degreesOfFreedom = 2;
    int numberControls = 2;

    float controlCost[NUM_CTRL] = {0.01, 0.01};
    float stateCosts[(2 * DOF)] = {10, 1, 0.1, 0.1};

    m_state X_desired;
    m_ctrl_ctrl R;
    m_state_state Q;

    // Given a set of mujoco data, what is the cost of its state and controls
    float costFunction(mjData *d);

    // Given a set of mujoco data, what are its cost derivates with respect to state and control
    void costDerivatives(mjData *d, Ref<m_state> l_x, Ref<m_state_state> l_xx, Ref<m_ctrl> l_u, Ref<m_ctrl_ctrl> l_uu);

    // set the state of a mujoco data object as per this model
    void setState(mjData *d, m_state X);

    // Return the state of a mujoco data model
    m_state returnState(mjData *d);

    // Set the controls of a mujoco data object
    void setControls(mjData *d, m_ctrl U);

    // Return the controls of a mujoco data object
    m_ctrl returnControls(mjData *d);

    m_dof returnAccelerations(mjData *d);

    void perturbVelocity(mjData *perturbedData, mjData *origData, int stateIndex, double eps);

    void perturbPosition(mjData *perturbedData, mjData *origData, int stateIndex, double eps);

};

#endif //MUJOCO_ACROBOT_CONTROL_ACROBOTMODEL_H
