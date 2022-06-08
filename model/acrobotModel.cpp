//
// Created by david on 27/04/22.
//

#include "acrobotModel.h"

acrobotModel::acrobotModel(m_state _desiredState){
    X_desired = _desiredState.replicate(1, 1);

    R.setIdentity();
    for(int i = 0; i < NUM_CTRL; i++){
        R(i, i) = controlCost[i];
    }
    Q.setIdentity();
    for(int i = 0; i < DOF; i++){
        Q(i, i) = stateCosts[i];
    }
}

// Given a set of mujoco data, what is the cost of its state and controls
float acrobotModel::costFunction(mjData *d){
    float stateCost;
    m_state X_diff;
    m_state X;
    m_ctrl U;

    X = returnState(d);
    U = returnControls(d);

    VectorXd temp(1);

    // actual - desired
    X_diff = X - X_desired;

    temp = (X_diff.transpose() * Q * X_diff) + (U.transpose() * R * U);

    stateCost = temp(0);

    return stateCost;
}

// Given a set of mujoco data, what are its cost derivates with respect to state and control
void acrobotModel::costDerivatives(mjData *d, Ref<m_state> l_x, Ref<m_state_state> l_xx, Ref<m_ctrl> l_u, Ref<m_ctrl_ctrl> l_uu){
    m_state X_diff;
    m_state X;
    m_ctrl U;

    X = returnState(d);
    U = returnControls(d);

    // actual - desired
    X_diff = X - X_desired;

    l_x = 2 * Q * X_diff;
    l_xx = 2 * Q;

    l_u = 2 * R * U;
    l_uu = 2 * R;
}

// set the state of a mujoco data object as per this model
void acrobotModel::setState(mjData *d, m_state X){
    for(int i = 0; i < NUM_CTRL; i++){
        d->qpos[i] = X(i);
        d->qvel[i] = X(i+2);
    }
}

// Return the state of a mujoco data model
m_state acrobotModel::returnState(mjData *d){
    m_state state;

    for(int i = 0; i < NUM_CTRL; i++){
        state(i) = d->qpos[i];
        state(i + 2) = d->qvel[i];
    }

    return state;
}

// Set the controls of a mujoco data object
void acrobotModel::setControls(mjData *d, m_ctrl U){
    for(int i = 0; i < NUM_CTRL; i++){
        d->ctrl[i] = U(i);
    }
}

// Return the controls of a mujoco data object
m_ctrl acrobotModel::returnControls(mjData *d){
    m_ctrl controls;
    for(int i = 0; i < NUM_CTRL; i++){
        controls(i) = d->ctrl[i];
    }

    return controls;
}

m_dof acrobotModel::returnAccelerations(mjData *d){
    m_dof accelerations;
    for(int i = 0; i < DOF; i++){
        accelerations(i) = d->qacc[i];
    }

    return accelerations;
}

void acrobotModel::perturbVelocity(mjData *perturbedData, mjData *origData, int stateIndex, double eps){
    perturbedData->qvel[stateIndex] = origData->qvel[stateIndex] + eps;
}

void acrobotModel::perturbPosition(mjData *perturbedData, mjData *origData, int stateIndex, double eps){
    perturbedData->qpos[stateIndex] = origData->qpos[stateIndex] + eps;
}
