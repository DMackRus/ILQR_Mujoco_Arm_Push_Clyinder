//
// Created by davem on 21/02/2022.
//

#ifndef MUJOCO_SANDBOX_STDINCLUDE_H
#define MUJOCO_SANDBOX_STDINCLUDE_H

#include <iostream>
#include <Eigen/Dense>
#include <chrono>
#include <fstream>
#include <stdio.h>
#include <string>
#include <vector>
#include <math.h>

#define DOF 10
#define NUM_CTRL 7
#define NUM_JOINTS 0
#define PI 3.141519265

#define ARM_MODEL 0

using namespace Eigen;
using namespace std;
using namespace std::chrono;

enum VecPos{
    x,
    y,
    z
};


typedef Vector<double, 3> m_point;
typedef Vector<double, 4> m_quat;
typedef Vector<double, 6> m_pose;

typedef Matrix<double, NUM_CTRL, 1> m_ctrl;
typedef Matrix<double, (2*DOF), 1> m_state;
typedef Matrix<double, DOF, 1> m_dof;


typedef Matrix<double, (2*DOF), (2*DOF)> m_state_state;
typedef Matrix<double, (2*DOF), NUM_CTRL> m_state_ctrl;
typedef Matrix<double, NUM_CTRL, (2*DOF)> m_ctrl_state;
typedef Matrix<double, NUM_CTRL, NUM_CTRL> m_ctrl_ctrl;
typedef Matrix<double, DOF, DOF> m_dof_dof;
typedef Matrix<double, DOF, NUM_CTRL> m_dof_ctrl;

#endif //MUJOCO_SANDBOX_STDINCLUDE_H
