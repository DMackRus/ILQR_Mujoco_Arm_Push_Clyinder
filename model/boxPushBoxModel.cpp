//
// Created by david on 29/04/22.
//

#include "boxPushBoxModel.h"

extern MujocoController *globalMujocoController;

boxModel::boxModel(mjModel *m, m_state _desiredState){
    X_desired = _desiredState.replicate(1, 1);

    R.setIdentity();
    for(int i = 0; i < NUM_CTRL; i++){
        R(i, i) = controlCost[i];
    }
    Q.setIdentity();
    for(int i = 0; i < (2 * DOF); i++){
        Q(i, i) = stateCosts[i];
    }

    cout << "Q" << Q << endl;
    cout << "R" << R << endl;

    model = m;

    // Magic number, 2 items
    for(int i = 0; i < 2; i++){
        stateNames.push_back(std::string());

    }

    stateNames[0] = "actuated_box";
    stateNames[1] = "unactuated_box";

}

float boxModel::getCost(mjData *d, int controlNum, int totalControls){
    float cost;
    m_state X_diff;
    m_state X;
    m_ctrl U;

    X = returnState(d);
    U = returnControls(d);

    cost = costFunction(controlNum, totalControls, X, U);

    return cost;
}

// Given a set of mujoco data, what is the cost of its state and controls
float boxModel::costFunction(int controlNum, int totalControls, m_state X, m_ctrl U){
    float stateCost;
    m_state X_diff;
    float posDiffFactor = 0.5;
    float posDiffCost = 0;

    //cout << "current State: " << endl << X << endl;

    VectorXd stateDiffCost(1);
    VectorXd controlCost(1);

    // actual - desired
    X_diff = X - X_desired;
    float percentageDone = (float)controlNum / (float)totalControls;
    float terminalScalar = (percentageDone * terminalConstant) + 1;
    terminalScalar = 1;

    //cout << "state diff was: " << endl << X_diff << endl;
    //cout << "Q: " << Q << endl;

    stateDiffCost = (X_diff.transpose() * Q * X_diff);
    controlCost = (U.transpose() * R * U);

    //cout << "state diff cost: " << endl << stateDiffCost << endl;
    //cout << "controlCost: " << endl << controlCost << endl;

    //float diffY = X(3) - X(1);
    //float diffX = X(2) - X(0);
    //posDiffCost = posDiffFactor * (pow(diffX, 2) + pow(diffY, 2));
    stateCost = stateDiffCost(0) + controlCost(0) + posDiffCost;

    return stateCost;
}

// Given a set of mujoco data, what are its cost derivates with respect to state and control
void boxModel::costDerivatives(mjData *d, Ref<m_state> l_x, Ref<m_state_state> l_xx, Ref<m_ctrl> l_u, Ref<m_ctrl_ctrl> l_uu, int controlNum, int totalControls){
    m_state X_diff;
    m_state X;
    m_ctrl U;

    X = returnState(d);
    U = returnControls(d);

    // actual - desired
    X_diff = X - X_desired;
    float percentageDone = (float)controlNum / (float)totalControls;
    float terminalScalar = (percentageDone * terminalConstant) + 1;
    terminalScalar = 1;

    l_x = 2 * Q * X_diff;
    l_xx = 2 * Q;

    l_u = 2 * R * U;
    l_uu = 2 * R;
}

void boxModel::costDerivatives_fd(mjData *d, Ref<m_state> l_x, Ref<m_state_state> l_xx, Ref<m_ctrl> l_u, Ref<m_ctrl_ctrl> l_uu, int controlNum, int totalControls){
    m_state X;
    m_ctrl U;
    float eps = 1e-2;

    X = returnState(d);
    U = returnControls(d);

    l_x = costDerivatives_fd_1stOrder(X, U, controlNum, totalControls);

    for(int i = 0; i < 2 * DOF; i++){
        m_state X_inc;
        m_state X_dec;
        m_state l_x_inc;
        m_state l_x_dec;

        X_inc = X.replicate(1,1);
        X_dec = X.replicate(1,1);

        X_inc(i) += eps;
        X_dec(i) -= eps;

        l_x_inc = costDerivatives_fd_1stOrder(X_inc, U, controlNum, totalControls);
        l_x_dec = costDerivatives_fd_1stOrder(X_dec, U, controlNum, totalControls);

        for(int j = 0; j < 2 * DOF; j++){
            l_xx(j, i) = (l_x_inc(j) - l_x_dec(j)) / (2 * eps);
        }

    }

    l_u = 2 * R * U;
    l_uu = 2 * R;
}

m_state boxModel::costDerivatives_fd_1stOrder(m_state X, m_ctrl U, int controlNum, int totalControls){
    m_state l_x;
    float eps = 1e-2;

    for(int i = 0; i < 2 * DOF; i++){
        m_state X_inc;
        m_state X_dec;
        float costInc;
        float costDec;

        X_inc = X.replicate(1,1);
        X_dec = X.replicate(1,1);

        X_inc(i) += eps;
        X_dec(i) -= eps;

        costInc = costFunction(controlNum, totalControls, X_inc, U);
        costDec = costFunction(controlNum, totalControls, X_dec, U);

        l_x(i) = (costInc - costDec) / (2 * eps);

    }

    return l_x;
}

// set the state of a mujoco data object as per this model
void boxModel::setState(mjData *d, m_state X){

    int actuatedId = mj_name2id(model, mjOBJ_BODY, stateNames[0].c_str());
    globalMujocoController->set_qPosVal(model, d, actuatedId, true, 0, X(0));
    globalMujocoController->set_qPosVal(model, d, actuatedId, true, 1, X(1));
    globalMujocoController->set_qVelVal(model, d, actuatedId, true, 0, X(DOF));
    globalMujocoController->set_qVelVal(model, d, actuatedId, true, 1, X(DOF + 1));

    int unactuatedId = mj_name2id(model, mjOBJ_BODY, stateNames[1].c_str());
    globalMujocoController->set_qPosVal(model, d, unactuatedId, true, 0, X(2));
    globalMujocoController->set_qPosVal(model, d, unactuatedId, true, 1, X(3));
    globalMujocoController->set_qVelVal(model, d, unactuatedId, true, 0, X(DOF + 2));
    globalMujocoController->set_qVelVal(model, d, unactuatedId, true, 1, X(DOF + 3));
    globalMujocoController->set_qVelVal(model, d, unactuatedId, true, 5, X(DOF + 4));

    m_quat currQuat = globalMujocoController->returnBodyQuat(model, d, unactuatedId);
    m_point axis = globalMujocoController->quat2Axis(currQuat);
    axis(2) = X(4);
    m_quat setQuat = globalMujocoController->axis2Quat(axis);
    globalMujocoController->setBodyQuat(model, d, unactuatedId, setQuat);

}

// Return the state of a mujoco data model
m_state boxModel::returnState(mjData *d){
    m_state state;

    int actuatedId = mj_name2id(model, mjOBJ_BODY, stateNames[0].c_str());
    state(0) = globalMujocoController->return_qPosVal(model, d, actuatedId, true, 0);
    state(1) = globalMujocoController->return_qPosVal(model, d, actuatedId, true, 1);
    state(DOF) = globalMujocoController->return_qVelVal(model, d, actuatedId, true, 0);
    state(DOF + 1) = globalMujocoController->return_qVelVal(model, d, actuatedId, true, 1);

    int unactuatedId = mj_name2id(model, mjOBJ_BODY, stateNames[1].c_str());
    state(2) = globalMujocoController->return_qPosVal(model, d, unactuatedId, true, 0);
    state(3) = globalMujocoController->return_qPosVal(model, d, unactuatedId, true, 1);
    state(DOF + 2) = globalMujocoController->return_qVelVal(model, d, unactuatedId, true, 0);
    state(DOF + 3) = globalMujocoController->return_qVelVal(model, d, unactuatedId, true, 1);
    state(DOF + 4) = globalMujocoController->return_qVelVal(model, d, unactuatedId, true, 5);

    m_quat currQuat = globalMujocoController->returnBodyQuat(model, d, unactuatedId);
    m_point axis = globalMujocoController->quat2Axis(currQuat);
    state(4) = axis(2);

    return state;
}

// Set the controls of a mujoco data object
void boxModel::setControls(mjData *d, m_ctrl U){
    for(int i = 0; i < NUM_CTRL; i++){
        d->ctrl[i] = U(i);
    }
}

// Return the controls of a mujoco data object
m_ctrl boxModel::returnControls(mjData *d){
    m_ctrl controls;
    for(int i = 0; i < NUM_CTRL; i++){
        controls(i) = d->ctrl[i];
    }

    return controls;
}

m_dof boxModel::returnPositions(mjData *d){
    m_dof positions;

    int actuatedId = mj_name2id(model, mjOBJ_BODY, stateNames[0].c_str());
    positions(0) = globalMujocoController->return_qPosVal(model, d, actuatedId, true, 0);
    positions(1) = globalMujocoController->return_qPosVal(model, d, actuatedId, true, 1);

    int unactuatedId = mj_name2id(model, mjOBJ_BODY, stateNames[1].c_str());
    positions(2) = globalMujocoController->return_qPosVal(model, d, unactuatedId, true, 0);
    positions(3) = globalMujocoController->return_qPosVal(model, d, unactuatedId, true, 1);

    m_quat currQuat = globalMujocoController->returnBodyQuat(model, d, unactuatedId);
    m_point axis = globalMujocoController->quat2Axis(currQuat);
    positions(4) = axis(2);

    return positions;
}

m_dof boxModel::returnVelocities(mjData *d){
    m_dof velocities;

    int actuatedId = mj_name2id(model, mjOBJ_BODY, stateNames[0].c_str());
    velocities(0) = globalMujocoController->return_qVelVal(model, d, actuatedId, true, 0);
    velocities(1) = globalMujocoController->return_qVelVal(model, d, actuatedId, true, 1);

    int unactuatedId = mj_name2id(model, mjOBJ_BODY, stateNames[1].c_str());
    velocities(2) = globalMujocoController->return_qVelVal(model, d, unactuatedId, true, 0);
    velocities(3) = globalMujocoController->return_qVelVal(model, d, unactuatedId, true, 1);
    velocities(4) = globalMujocoController->return_qVelVal(model, d, unactuatedId, true, 5);

    return velocities;
}

m_dof boxModel::returnAccelerations(mjData *d){
    m_dof accelerations;

    int actuatedId = mj_name2id(model, mjOBJ_BODY, stateNames[0].c_str());
    accelerations(0) = globalMujocoController->return_qAccVal(model, d, actuatedId, true, 0);
    accelerations(1) = globalMujocoController->return_qAccVal(model, d, actuatedId, true, 1);

    int unactuatedId = mj_name2id(model, mjOBJ_BODY, stateNames[1].c_str());
    accelerations(2) = globalMujocoController->return_qAccVal(model, d, unactuatedId, true, 0);
    accelerations(3) = globalMujocoController->return_qAccVal(model, d, unactuatedId, true, 1);
    accelerations(4) = globalMujocoController->return_qAccVal(model, d, unactuatedId, true, 5);

    return accelerations;
}

void boxModel::perturbVelocity(mjData *perturbedData, mjData *origData, int stateIndex, double eps){
    int stateNameIndex = stateIndexToStateName[stateIndex];
    int bodyId = mj_name2id(model, mjOBJ_BODY, stateNames[stateNameIndex].c_str());
    int freeJntIndex[DOF] = {0, 1, 0, 1, 5};

    double origVelocity = globalMujocoController->return_qVelVal(model, origData, bodyId, true, freeJntIndex[stateIndex]);
    double perturbedVel = origVelocity + eps;
    globalMujocoController->set_qVelVal(model, perturbedData, bodyId, true, freeJntIndex[stateIndex], perturbedVel);

}

void boxModel::perturbPosition(mjData *perturbedData, mjData *origData, int stateIndex, double eps){
    int stateNameIndex = stateIndexToStateName[stateIndex];
    int bodyId = mj_name2id(model, mjOBJ_BODY, stateNames[stateNameIndex].c_str());
    int freeJntIndex[DOF] = {0, 1, 0, 1, 5};

    if(stateIndex < 4){
        double origPos = globalMujocoController->return_qPosVal(model, origData, bodyId, true, freeJntIndex[stateIndex]);
        double perturbedPos = origPos + eps;
        globalMujocoController->set_qPosVal(model, perturbedData, bodyId, true, freeJntIndex[stateIndex], perturbedPos);
    }
    else{
        // quaternion code
        m_quat currQuat = globalMujocoController->returnBodyQuat(model, origData, bodyId);
        m_point currAxis = globalMujocoController->quat2Axis(currQuat);

        m_point newAxis = currAxis.replicate(1,1);
        newAxis(2) += eps;
        m_quat newQuat = globalMujocoController->axis2Quat(newAxis);
        globalMujocoController->setBodyQuat(model, perturbedData, bodyId, newQuat);
    }


}

void boxModel::stepModel(mjData *d, int numSteps){

//    for(int i = 0; i < numSteps; i++){
//        mj_step(model, d);
//        d->qpos[2] = heightFloat;
//        int bodyId = mj_name2id(model, mjOBJ_BODY, stateNames[0].c_str());
//        m_point upright;
//        upright << 0, 0, 0;
//        m_quat quatSet = globalMujocoController->axis2Quat(upright);
//        globalMujocoController->setBodyQuat(model, d, bodyId, quatSet);
//    }

    for(int i = 0; i < numSteps; i++){
        mj_step(model, d);

    }

}
