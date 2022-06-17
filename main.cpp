#include "Utility/MujocoController/MujocoUI.h"
#include "iLQR/iLQR_dataCentric.h"
#include "model/frankaArmAndBox.h"
#include "Utility/stdInclude/stdInclude.h"

#include "mujoco.h"


#define RUN_ILQR 1
#define TEST_LINEARISATION 0


extern MujocoController *globalMujocoController;
extern mjModel* model;						// MuJoCo model
extern mjData* mdata;						// MuJoCo data

extern iLQR* optimiser;
frankaModel* modelTranslator;

std::vector<m_state> X_dyn;
std::vector<m_state> X_lin;

m_state X0;
m_state X_desired;

ofstream outputDiffDyn;
std::string diffDynFilename = "diffDyn.csv";
ofstream outputFile;
std::string filename = "finalTrajectory.csv";

extern mjvCamera cam;                   // abstract camera
extern mjvScene scn;                    // abstract scene
extern mjvOption opt;			        // visualization options
extern mjrContext con;				    // custom GPU context
extern GLFWwindow *window;

extern std::vector<m_ctrl> testInitControls;
extern mjData* d_init_test;

void saveStates();
void saveTrajecToCSV();

void testILQR(m_state X0);

void simpleTest();
void initControls();

int main() {
    initMujoco();
    // Franka arm with end effector parallel to ground configuration
//    X0 << 0, 0.275, -0.119, -2.76, 2.97, 0, 0,
//            0.7, 0,
//            0, 0, 0, 0, 0, 0, 0,
//            0, 0;

// 0.6 normal starting point for cube
    X0 << -0.12, 0.5, 0.06, -2.5, 0, 1.34, 0,
            2, 0.02, 0,
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0;

    X_desired << 0 ,0, 0, 0, 0, 0, 0,
            0.8, 0.2, 0,
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0;


    if(RUN_ILQR){

        auto iLQRStart = high_resolution_clock::now();

        modelTranslator = new frankaModel(model, X_desired);
        optimiser = new iLQR(model, mdata, X0, modelTranslator, globalMujocoController);
        initControls();
        optimiser->setInitControls(testInitControls);
        optimiser->makeDataForOptimisation();
        testILQR(X0);

        auto iLQRStop = high_resolution_clock::now();
        auto iLQRDur = duration_cast<microseconds>(iLQRStop - iLQRStart);

        cout << "iLQR took " << iLQRDur.count()/1000000 << " seconds" << endl;

        render();
    }
    else{
        modelTranslator = new frankaModel(model, X_desired);
        optimiser = new iLQR(model, mdata, X0, modelTranslator, globalMujocoController);
        simpleTest();
        render_simpleTest();
    }

    return 0;
}

//void initControls(){
//    d_init_test = mj_makeData(model);
//    modelTranslator->setState(mdata, X0);
//    for(int i = 0; i < 200; i++){
//        mj_step(model, mdata);
//    }
//    cpMjData(model, d_init_test, mdata);
//
//    const std::string test1 = "panda0_link0";
//    const std::string test2 = "panda0_rightfinger";
//    const std::string test3 = "box_obstacle_1";
//    int test1id = mj_name2id(model, mjOBJ_BODY, test1.c_str());
//    int test2id = mj_name2id(model, mjOBJ_BODY, test2.c_str());
//    int test3id = mj_name2id(model, mjOBJ_BODY, test3.c_str());
//    cout << "id 1: " << test1id << "id 2: " << test2id << " id: 3: " << test3id <<  endl;
//    cout << "model nv: " << model->nv << " model nq: " << model->nq << " model nu: " << model->nu << endl;
//
//    const std::string endEffecName = "franka_gripper";
//    int endEffecId = mj_name2id(model, mjOBJ_BODY, endEffecName.c_str());
//
//    m_pose startPose = globalMujocoController->returnBodyPose(model, mdata, endEffecId);
//    m_quat startQuat = globalMujocoController->returnBodyQuat(model, mdata, endEffecId);
//    cout << "start quat: " << startQuat << endl;
//
//    cout << "start pose is: " << startPose << endl;
//    m_pose endPose;
//    m_pose direction;
//    direction << 0.6, 0, 0, 0, 0, 0;
//    float magnitudeDiff = sqrt((pow(direction(0), 2)) + (pow(direction(1), 2)) + (pow(direction(2), 2)));
//    //float forceMagnitude = 50;
//    float forceMagnitude = 50;
//    // normalise vector diff
//    direction /= magnitudeDiff;
//
//    m_pose linearInterpolationDesiredForce;
//    linearInterpolationDesiredForce = direction * forceMagnitude;
//    cout << "linear interpolation desired force: " << linearInterpolationDesiredForce << endl;
//    endPose = startPose + direction;
//
//    for(int i = 0; i <= optimiser->ilqr_horizon_length; i++){
//
//        m_pose currentEEPose = globalMujocoController->returnBodyPose(model, mdata, endEffecId);
//        m_quat currentQuat = globalMujocoController->returnBodyQuat(model, mdata, endEffecId);
//        m_quat invCurrentQuat = globalMujocoController->invQuat(currentQuat);
//
//        m_quat quatDiff = globalMujocoController->multQuat(startQuat, invCurrentQuat);
//        //cout << "quat diff: " << quatDiff << endl;
//        //cout << "end effec pose: " << currentEEPose << endl;
//        MatrixXd Jac = globalMujocoController->calculateJacobian(model, mdata, endEffecId);
//        MatrixXd Jac_t = Jac.transpose();
//
//        m_point axisDiff = globalMujocoController->quat2Axis(quatDiff);
//
//        m_ctrl desiredControls;
//        m_pose desiredEEForce;
//        m_pose diff;
//        // cout << "currentEEPoint: " << currentEEPoint << endl;
//        diff = (currentEEPose - endPose);
//        diff(3) = axisDiff(0);
//        diff(4) = axisDiff(1);
//        diff(5) = axisDiff(2);
//        desiredEEForce = linearInterpolationDesiredForce;
//
//        float zAxisRedFactor = 100 * diff(2);
//        float rollAxisRedFactor = 10 * diff(3);
//        float pitchAxisRedFactor = 10 * diff(4);
//        float yawAxisRedFactor = 10 * diff(5);
//        desiredEEForce(2) -= zAxisRedFactor;
//        desiredEEForce(3) -= rollAxisRedFactor;
//        desiredEEForce(4) -= pitchAxisRedFactor;
//        desiredEEForce(5) -= yawAxisRedFactor;
//
//        desiredControls = Jac_t * desiredEEForce;
//
//        testInitControls.push_back(m_ctrl());
//
//        for(int k = 0; k < NUM_CTRL; k++){
//
//
//            testInitControls[i](k) = desiredControls(k) + mdata->qfrc_bias[k];
//            mdata->ctrl[k] = testInitControls[i](k);
//        }
//
//        for(int j = 0; j < optimiser->num_mj_steps_per_control; j++){
//            mj_step(model, mdata);
//        }
//    }
//}

void initControls(){
    d_init_test = mj_makeData(model);
    modelTranslator->setState(mdata, X0);

    for(int i = 0; i < 200; i++){
        mj_step(model, mdata);
    }
    cpMjData(model, d_init_test, mdata);

    for(int i = 0; i <= optimiser->ilqr_horizon_length; i++){

        m_state X_diff;
        m_state X = modelTranslator->returnState(mdata);
        m_ctrl nextControl;
        X_diff = X_desired - X;
        int K[7] = {870, 870, 870, 870, 120, 120, 120};

        testInitControls.push_back(m_ctrl());

        nextControl(0) = X_diff(0) * K[0];
        nextControl(1) = X_diff(1) * K[1];
        nextControl(2) = X_diff(2) * K[2];
        nextControl(3) = X_diff(3) * K[3];
        nextControl(4) = X_diff(4) * K[4];
        nextControl(5) = X_diff(5) * K[5];
        nextControl(6) = X_diff(6) * K[6];


        for(int k = 0; k < NUM_CTRL; k++){

            //nextControl(k) = X_diff(k) * K[i];
            testInitControls[i](k) = nextControl(k);
            mdata->ctrl[k] = testInitControls[i](k);

        }

        cout << "x_diff[i]" << X_diff << endl;
        cout << "next control: " << nextControl << endl;
        cout << "testInitControls[i]" << testInitControls[i] << endl;

        for(int j = 0; j < optimiser->num_mj_steps_per_control; j++){
            mj_step(model, mdata);
        }
    }
}

void simpleTest(){

    initControls();

    m_state testState;
//    float cost;
//    modelTranslator->setState(mdata, X0);
//    cost = modelTranslator->costFunction(mdata, 0, 0);

//    testState << 1, 1, 0.3, 0.1, 1, 1, 1, 1 << ;
//    modelTranslator->setState(mdata, testState);
//    cost = modelTranslator->costFunction(mdata, 0, 0);
    int a = 1;


}

void testILQR(m_state X0){

    outputFile.open(filename);
    outputFile << "Control Number" << ",";

    outputFile << "T1" << ",";
    outputFile << "T2" << ",";
    outputFile << "T3" << ",";
    outputFile << "T4" << ",";
    outputFile << "T5" << ",";
    outputFile << "T6" << ",";
    outputFile << "T7" << ",";

    outputFile << "T1_init" << ",";
    outputFile << "T2_init" << ",";
    outputFile << "T3_init" << ",";
    outputFile << "T4_init" << ",";
    outputFile << "T5_init" << ",";
    outputFile << "T6_init" << ",";
    outputFile << "T7_init" << ",";

    outputFile << "J1" << ",";
    outputFile << "J2" << ",";
    outputFile << "J3" << ",";
    outputFile << "J4" << ",";
    outputFile << "J5" << ",";
    outputFile << "J6" << ",";
    outputFile << "J7" << ",";

    outputFile << "Cube x" << ",";
    outputFile << "Cube y" << ",";
    outputFile << "Cube w" << ",";

    outputFile << "V1" << ",";
    outputFile << "V2" << ",";
    outputFile << "V3" << ",";
    outputFile << "V4" << ",";
    outputFile << "V5" << ",";
    outputFile << "V6" << ",";
    outputFile << "V7" << ",";

    outputFile << "Cube x V" << ",";
    outputFile << "Cube y V" << ",";
    outputFile << "Cube w V" << ",";



    outputFile << endl;

    for(int i = 0; i < MUJ_STEPS_HORIZON_LENGTH+1; i++){
        X_dyn.push_back(m_state());
        X_lin.push_back(m_state());
    }


    double cubeVelDiff = 0.0f;
    double joint4Diff = 0.0f;

    bool firstXDot = true;
    m_state mujXDot;
    m_ctrl prevControl, currControl;

    if(TEST_LINEARISATION){
        cpMjData(model, mdata, d_init_test);
        for(int i = 0;  i < MUJ_STEPS_HORIZON_LENGTH; i++){
            MatrixXd A = ArrayXXd::Zero((2 * DOF), (2 * DOF));
            MatrixXd B = ArrayXXd::Zero((2 * DOF), NUM_CTRL);

            MatrixXd A_scaled = ArrayXXd::Zero((2 * DOF), (2 * DOF));
            MatrixXd B_scaled = ArrayXXd::Zero((2 * DOF), NUM_CTRL);

            if(i == 0) {
                // set X0 dyn and lin as the initial state of the system
                X_dyn[0] = modelTranslator->returnState(mdata);
                X_lin[0] = modelTranslator->returnState(mdata);

                for(int k = 0; k < NUM_CTRL; k++){
                    currControl(k) = testInitControls[i](k);
                }

                modelTranslator->setControls(mdata, currControl);

                // TODO FIX LINEARISATION TESTING WITH NEW SCALING METHOD
                modelTranslator->stepModel(mdata, 1);

                X_dyn[1] = modelTranslator->returnState(mdata);
                X_lin[1] = modelTranslator->returnState(mdata);

                for(int k = 0; k < NUM_CTRL; k++){
                    currControl(k) = testInitControls[i](k);
                }

                modelTranslator->setControls(mdata, currControl);

                modelTranslator->stepModel(mdata, 1);
            }
            else{
                // Calculate X bar and U bar at current iteration by comparing current state and control with last state and control
                m_state currentState_dyn, lastState_dyn, X_bar;
                m_ctrl U_bar;
                m_state X_bar_dot, X_dot;
                X_dyn[i] = modelTranslator->returnState(mdata);
                //cout << ""

                currentState_dyn = X_dyn[i].replicate(1,1);
                lastState_dyn = X_dyn[i - 1].replicate(1,1);

                //currentState_dyn = X_lin[i].replicate(1,1);
                //lastState_dyn = X_dyn[i - 1].replicate(1,1);

                X_bar = currentState_dyn - lastState_dyn;

                prevControl = modelTranslator->returnControls(mdata);
                for(int k = 0; k < NUM_CTRL; k++){
                    currControl(k) = testInitControls[i](k);
                }

                modelTranslator->setControls(mdata, currControl);
                U_bar = currControl - prevControl;

                // Calculate A and B matrices by linearising around previous state
                optimiser->lineariseDynamicsSerial_trial_step(A, B, mdata, MUJOCO_DT);


                if(i >= 500 and i <= 505){
                    cout << "------------------------ i = 1190 ----------------------------" << endl;
                    cout << "A is" << endl << A << endl;
                    cout << "B is" << endl << B << endl;

                }
//
//                if(i == 900){
//                    cout << "------------------------ i = 900 ----------------------------" << endl;
//                    cout << "A is" << endl << A << endl;
//                    cout << "B is" << endl << B << endl;
//
//                    for(int i = 0; i < 10; i++){
//                        mjrRect viewport = { 0, 0, 0, 0 };
//                        glfwGetFramebufferSize(window, &viewport.width, &viewport.height);
//
//                        // update scene and render
//                        mjv_updateScene(model, mdata, &opt, NULL, &cam, mjCAT_ALL, &scn);
//                        mjr_render(viewport, &scn, &con);
//
//                        // swap OpenGL buffers (blocking call due to v-sync)
//                        glfwSwapBuffers(window);
//
//                        // process pending GUI events, call GLFW callbacks
//                        glfwPollEvents();
//                    }
//
//                    int a = 1;
//                    int b = a;
//                }

                // Calculate X bar dot via X(.) = Ax + BU
                X_bar_dot = (A_scaled * X_bar) + (B_scaled * U_bar);

                // temporary, compare linearised x dot to mujoco velocities and accelerations:
                m_dof velocities = modelTranslator->returnVelocities(mdata);
                m_dof accelerations = modelTranslator->returnAccelerations(mdata);
                for(int j = 0; j < DOF; j++){
                    mujXDot(j) = velocities(j);
                    mujXDot(j + DOF) = accelerations(j);
                }


                // Calculate next state via linearisation using euler integration of current X_dot
                if(0){
                    X_lin[i + 1] = X_dyn[i] + X_bar_dot;
                    //X_lin[i + 1] = X_dyn[i] + (X_dot * MUJOCO_DT);

                }
                else{
                    X_lin[i + 1] = X_lin[i] + X_bar_dot;
                }

                modelTranslator->stepModel(mdata, 1);


                if(i % 50){
//                    m_state x_dyn_diff, X_lin_diff;
//                    cout << "x bar dot was: " << X_bar_dot << endl;
//                    cout << "linearised x dot was: " << X_dot << endl;
//                    cout << "mujoco X dot was: " << mujXDot << endl;
//
//                    x_dyn_diff = X_dyn[i] - X_dyn[i - 1];
//                    X_lin_diff = X_lin[i] - X_lin[i - 1];
//
//                    cout << "dyn diff: " << endl << x_dyn_diff << endl;
//                    cout << "lin diff: " << endl << X_lin_diff << endl;
//                    int a;
                }

//                if(X_lin[i + 1](17) - X_dyn[i + 1](17) > 0.1){
//                    cout << "iteration big diff: " << i << endl;
//                }

                //cubeVelDiff += pow(((X_lin[i + 1](17) - X_dyn[i + 1](17)) * ILQR_DT),2);
                //joint4Diff += pow(((X_lin[i + 1](14) - X_dyn[i + 1](14)) * ILQR_DT),2);

                cubeVelDiff += pow(((X_lin[i](6) - X_dyn[i](6)) * optimiser->ilqr_dt),2);

            }
        }
        saveStates();
    }
    cout << "cube vel diff total: " << cubeVelDiff << endl;
    cout << "joint 4 vel diff total: " << joint4Diff << endl;

    // Reset data to initial state
    cpMjData(model, mdata, optimiser->d_init);

    optimiser->optimise();

    saveTrajecToCSV();

}

void saveTrajecToCSV(){

    for(int i = 0; i < MUJ_STEPS_HORIZON_LENGTH; i++){
        outputFile << i << ",";

        for(int j = 0; j < NUM_CTRL; j++){
            outputFile << optimiser->finalControls[i](j) << ",";
        }

        for(int j = 0; j < NUM_CTRL; j++){
            outputFile << optimiser->initControls[i](j) << ",";
        }

        for(int j = 0; j < DOF; j++){
            outputFile << optimiser->X_final[i](j) << ",";
        }

        for(int j = 0; j < DOF; j++){
            outputFile << optimiser->X_final[i](j+DOF) << ",";
        }
        outputFile << endl;
    }

    outputFile.close();
}

void saveStates(){

    cout << "X_dyn[end]: " << X_dyn[MUJ_STEPS_HORIZON_LENGTH] << endl;
    cout << "X_lin[end]: " << X_lin[MUJ_STEPS_HORIZON_LENGTH] << endl;

    cout << "X_dyn[0]: " << X_dyn[0] << endl;
    cout << "X_lin[0]: " << X_lin[0] << endl;


    outputDiffDyn.open(diffDynFilename);
    outputDiffDyn << "Joint 0 dyn" << "," << "Joint 0 lin" << "," << "Joint 0 diff" << "," << "Joint 1 dyn" << "," << "Joint 1 lin" << "," << "Joint 1 diff" << ",";
    outputDiffDyn << "Joint 2 dyn" << "," << "Joint 2 lin" << "," << "Joint 2 diff" << "," << "Joint 3 dyn" << "," << "Joint 3 lin" << "," << "Joint 3 diff" << ",";
    outputDiffDyn << "Joint 4 dyn" << "," << "Joint 4 lin" << "," << "Joint 4 diff" << "," << "Joint 5 dyn" << "," << "Joint 5 lin" << "," << "Joint 5 diff" << ",";
    outputDiffDyn << "Joint 6 dyn" << "," << "Joint 6 lin" << "," << "Joint 6 diff" << ",";
    outputDiffDyn << "Cube X dyn" << "," << "Cube X lin" << "," << "Cube X diff" << "," << "Cube Y dyn" << "," << "Cube Y lin" << "," << "Cube Y diff" << "," << "Cube rot dyn" << "," << "Cube rot lin" << "," << "Cube rot diff" << ",";
    outputDiffDyn << "Joint 0 vel dyn" << "," << "Joint 0 vel lin" << "," << "Joint 0 vel diff" << ",";
    outputDiffDyn << "Joint 1 vel dyn" << "," << "Joint 1 vel lin" << "," << "Joint 1 vel diff" << "," << "Joint 2 vel dyn" << "," << "Joint 2 vel lin" << "," << "Joint 2 vel diff" << ",";
    outputDiffDyn << "Joint 3 vel dyn" << "," << "Joint 3 vel lin" << "," << "Joint 3 vel diff" << "," << "Joint 4 vel dyn" << "," << "Joint 4 vel lin" << "," << "Joint 4 vel diff" << ",";
    outputDiffDyn << "Joint 5 vel dyn" << "," << "Joint 5 vel lin" << "," << "Joint 5 vel diff" << "," << "Joint 6 vel dyn" << "," << "Joint 6 vel lin" << "," << "Joint 6 vel diff" << ",";
    outputDiffDyn << "Cube X  vel dyn" << "," << "Cube X vel lin" << "," << "Cube X vel diff" << "," << "Cube Y vel dyn" << "," << "Cube Y vel lin" << "," << "Cube Y vel diff" << "," << "Cube rot vel dyn" << "," << "Cube rot vel lin" << "," << "Cube rot vel diff" << endl;

    for(int i = 0; i < MUJ_STEPS_HORIZON_LENGTH; i++){
        for(int j = 0; j < (2 * DOF); j++){
            float val;
            val = X_dyn[i](j);
            outputDiffDyn << val << ",";
            val = X_lin[i](j);
            outputDiffDyn << val << ",";
            val = X_lin[i](j) - X_dyn[i](j);
            outputDiffDyn << val << ",";
        }
        outputDiffDyn << endl;
    }

    outputDiffDyn.close();
}
