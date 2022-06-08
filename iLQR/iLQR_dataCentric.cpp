//
// Created by david on 13/04/2022.
//

#include "iLQR_dataCentric.h"


iLQR* optimiser;
extern mjvCamera cam;                   // abstract camera
extern mjvScene scn;                    // abstract scene
extern mjvOption opt;			        // visualization options
extern mjrContext con;				    // custom GPU context
extern GLFWwindow *window;

iLQR::iLQR(mjModel* m, mjData* d, m_state _X0, frankaModel* _modelTranslator, MujocoController* _mujocoController){
    numIterations = 0;
    lamda = 0.001;

    for(int i=0; i < ILQR_HORIZON_LENGTH; i++) {
        initControls.push_back(m_ctrl());
        finalControls.push_back(m_ctrl());
    }

    for(int i = 0; i < ILQR_HORIZON_LENGTH; i++){
        f_x.push_back(m_state_state());
        f_u.push_back(m_state_ctrl());

        A_scaled.push_back(m_state_state());
        B_scaled.push_back(m_state_ctrl());

        l_x.push_back(m_state());
        l_xx.push_back(m_state_state());
        l_u.push_back(m_ctrl());
        l_uu.push_back(m_ctrl_ctrl());

        k.push_back(m_ctrl());
        K.push_back(m_ctrl_state());

        U_new.push_back(m_ctrl());
        U_old.push_back(m_ctrl());

        X_new.push_back(m_state());
        X_old.push_back(m_state());
    }

    // Extra as one more state than controls
    l_x.push_back(m_state());
    l_xx.push_back(m_state_state());

    // Initialise internal iLQR model and data
    model = m;
    mujocoController = _mujocoController;
    modelTranslator = _modelTranslator;
    mdata = mj_makeData(model);
    cpMjData(model, mdata, d);
    X0 = _X0.replicate(1,1);


}

void iLQR::optimise(){
    bool optimisationFinished = false;
    int numIterations = 0;
    float newCost = 0;
    float oldCost = 1000;

    //oldCost = rollOutTrajectory();
    cout << "initial Trajectory cost: " << oldCost << endl;
    cout << "---------------------------------------------------- " << endl;

    // iterate until optimisation finished, convergence or if lamda > maxLamda
    for(int i = 0; i < maxIterations; i++){
        numIterations++;

        auto start = high_resolution_clock::now();

        // Linearise the dynamics and save cost values at each state
        // STEP 1 - Linearise dynamics and calculate cost quadratics at every time step
        getDerivatives();

        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        cout << "Linearising model: " << duration.count()/1000 << " milliseconds" << endl;

        bool validBackPass = false;
        bool lamdaExit = false;

        // STEP 2 - Backwards pass to compute optimal linear and feedback gain matrices k and K
        // Keep running this step until the backwards pass does NOT encounter a non_PD Q_uu_reg matrix
        while(!validBackPass) {

            validBackPass = backwardsPass_Quu_reg();

            if (!validBackPass) {
                if (lamda < maxLamda) {
                    lamda *= lamdaFactor;
                } else {
                    lamdaExit = true;
                    optimisationFinished = true;
                    break;
                }
            } else {
                if (lamda > minLamda) {
                    lamda /= lamdaFactor;
                }
            }
        }

        if(!lamdaExit){
            // STEP 3 - Forwards pass to calculate new optimal controls - with optional alpha backtracking line search
            newCost = forwardsPass(oldCost);

            // STEP 4 - Check for convergence
            optimisationFinished = checkForConvergence(newCost, oldCost);
            if(optimisationFinished){
                break;
            }

            oldCost = newCost;
        }
        else{
            cout << "optimisation exited after lamda exceed lamda max, iteration: " << i << endl;
            break;
        }
    }

    for(int i = 0; i < ILQR_HORIZON_LENGTH; i++){
        finalControls[i] = U_new[i].replicate(1, 1);
    }

}

float iLQR::rollOutTrajectory(){
    float cost = 0;

    cpMjData(model, mdata, d_init);
    for(int i = 0; i < ILQR_HORIZON_LENGTH; i++){
        modelTranslator->setControls(mdata, U_old[i]);
        float stateCost = modelTranslator->getCost(mdata, i, ILQR_HORIZON_LENGTH);
        cost += (stateCost * ILQR_DT);
        modelTranslator->stepModel(mdata, NUM_MJSTEPS_PER_CONTROL);
    }
    m_state termState = modelTranslator->returnState(mdata);
    cout << "--------------------------------------------------" << endl;
    cout << "terminal state, cube x: " << termState(7) << ", Y: " << termState(8) << endl;
    cpMjData(model, mdata, d_init);

    return cost;
}

void iLQR::getDerivatives(){

    MatrixXd A = ArrayXXd::Zero((2 * DOF), (2 * DOF));
    MatrixXd B = ArrayXXd::Zero((2 * DOF), NUM_CTRL);

    int save_iterations = model->opt.iterations;
    mjtNum save_tolerance = model->opt.tolerance;

    model->opt.iterations = 30;
    model->opt.tolerance = 0;

    int numReplacements = 0;
    const int threshold = 100;

    cpMjData(model, mdata, d_init);
    X_old[0] = modelTranslator->returnState(mdata);

    // Linearise the dynamics along the trajectory
    for(int t = 0; t < ILQR_HORIZON_LENGTH; t++){
        // Calculate linearised dynamics for current time step via finite differencing
        modelTranslator->setControls(mdata, U_old[t]);
        lineariseDynamicsSerial_trial_step(A, B, mdata, MUJOCO_DT);

        f_x[t] = A.replicate(1,1);
        f_u[t] = B.replicate(1,1);

        modelTranslator->costDerivatives(mdata, l_x[t], l_xx[t], l_u[t], l_uu[t], t, ILQR_HORIZON_LENGTH);

        l_x[t]  *= ILQR_DT;
        l_xx[t] *= ILQR_DT;
        l_u[t]  *= ILQR_DT;
        l_uu[t] *= ILQR_DT;

        for(int i = 0; i < NUM_MJSTEPS_PER_CONTROL; i++){
            modelTranslator->stepModel(mdata, 1);
        }

        X_old[t + 1] = modelTranslator->returnState(mdata);
    }

    model->opt.iterations = save_iterations;
    model->opt.tolerance = save_tolerance;

    //TODO FIX FACT THAT THERE SHOULD BE NO CONTROL COST AT END OF TRAJECTORY
    modelTranslator->costDerivatives(mdata, l_x[ILQR_HORIZON_LENGTH], l_xx[ILQR_HORIZON_LENGTH], l_u[ILQR_HORIZON_LENGTH-1], l_uu[ILQR_HORIZON_LENGTH-1], ILQR_HORIZON_LENGTH, ILQR_HORIZON_LENGTH);
    l_x [ILQR_HORIZON_LENGTH] *= ILQR_DT;
    l_xx[ILQR_HORIZON_LENGTH] *= ILQR_DT;

}


void iLQR::scaleLinearisation(Ref<m_state_state> A_scaled, Ref<m_state_ctrl> B_scaled, Ref<m_state_state> A, Ref<m_state_ctrl> B, int num_steps_per_dt){

    // TODO look into ways of speeding up matrix to the power of calculation
    A_scaled = A.replicate(1, 1);
    B_scaled = B.replicate(1, 1);
    m_state_ctrl currentBTerm;

    for(int i = 0; i < num_steps_per_dt - 1; i++){
        A_scaled *= A;
    }

    currentBTerm = B.replicate(1, 1);
    for(int i = 0; i < num_steps_per_dt - 1; i++){
        currentBTerm = A * currentBTerm;
        B_scaled += currentBTerm;
    }
}

bool iLQR::backwardsPass_Quu_reg(){
    m_state V_x;
    V_x = l_x[ILQR_HORIZON_LENGTH];
    m_state_state V_xx;
    V_xx = l_xx[ILQR_HORIZON_LENGTH];

    for(int t = ILQR_HORIZON_LENGTH - 2; t > -1; t--){
        m_state Q_x;
        m_ctrl Q_u;
        m_state_state Q_xx;
        m_ctrl_ctrl Q_uu;
        m_ctrl_state Q_ux;

        Q_u = l_u[t] + (f_u[t].transpose() * V_x);

        Q_x = l_x[t] + (f_x[t].transpose() * V_x);

        Q_ux = (f_u[t].transpose() * (V_xx * f_x[t]));

        Q_uu = l_uu[t] + (f_u[t].transpose() * (V_xx * f_u[t]));

        Q_xx = l_xx[t] + (f_x[t].transpose() * (V_xx * f_x[t]));


        m_ctrl_ctrl Q_uu_reg = Q_uu.replicate(1, 1);

        for(int i = 0; i < NUM_CTRL; i++){
            Q_uu_reg(i, i) += lamda;
        }

        if(!isMatrixPD(Q_uu_reg)){
            cout << "iteration " << t << endl;
            cout << "f_x[t - 3] " << f_x[t - 3] << endl;
            cout << "f_x[t - 2] " << f_x[t - 2] << endl;
            cout << "f_x[t - 1] " << f_x[t - 1] << endl;
            cout << "f_x[t] " << f_x[t] << endl;
            cout << "Q_uu_reg " << Q_uu_reg << endl;
            return false;
        }

        auto temp = (Q_uu_reg).ldlt();
        m_ctrl_ctrl I;
        I.setIdentity();
        m_ctrl_ctrl Q_uu_inv = temp.solve(I);

        k[t] = -Q_uu_inv * Q_u;
        K[t] = -Q_uu_inv * Q_ux;

        V_x = Q_x + (K[t].transpose() * (Q_uu * k[t])) + (K[t].transpose() * Q_u) + (Q_ux.transpose() * k[t]);
        V_xx = Q_xx + (K[t].transpose() * (Q_uu * K[t])) + (K[t].transpose() * Q_ux) + (Q_ux.transpose() * K[t]);

        V_xx = (V_xx + V_xx.transpose()) / 2;

//        cout << "l_x " << l_x[t] << endl;
//        cout << "l_xx " << l_xx[t] << endl;
//        cout << "Q_ux " << Q_ux << endl;
//        cout << "f_u[t] " << f_u[t] << endl;
//        cout << "Q_uu " << Q_uu << endl;
//        cout << "Q_uu_inv " << Q_uu_inv << endl;
//        cout << "V_xx " << V_xx << endl;
//        cout << "V_x " << V_x << endl;
//        cout << "K[t] " << K[t] << endl;
        int a = 1;
    }

    return true;
}

bool iLQR::backwardsPass_Vxx_reg(){
    m_state V_x;
    V_x = l_x[ILQR_HORIZON_LENGTH - 1];
    m_state_state V_xx;
    V_xx = l_xx[ILQR_HORIZON_LENGTH - 1];

    for(int t = ILQR_HORIZON_LENGTH - 1; t > 0; t--){
        m_state Q_x;
        m_ctrl Q_u;
        m_state_state Q_xx;
        m_ctrl_ctrl Q_uu;
        m_ctrl_state Q_ux;
        m_state_state V_xx_reg;

        V_xx_reg = V_xx.replicate(1, 1);
        for(int i = 0; i < (2 * DOF); i++){
            V_xx_reg(i, i) += lamda;
        }

        Q_x = l_x[t] + (f_x[t].transpose() * V_x);

        Q_u = l_u[t] + (f_u[t].transpose() * V_x);

        Q_xx = l_xx[t] + (f_x[t].transpose() * (V_xx * f_x[t]));

        Q_uu = l_uu[t] + (f_u[t].transpose() * (V_xx * f_u[t]));

        Q_ux = (f_u[t].transpose() * (V_xx * f_x[t]));

        m_ctrl_ctrl Q_uu_reg;
        m_ctrl_state Q_ux_reg;

        Q_uu_reg = l_uu[t] + (f_u[t].transpose() * (V_xx_reg * f_u[t]));

        Q_ux_reg = (f_u[t].transpose() * (V_xx_reg * f_x[t]));

        if(!isMatrixPD(Q_uu_reg)){
            cout << "iteration " << t << endl;
            cout << "f_x[t] " << f_x[t] << endl;
            cout << "Q_uu_reg " << Q_uu_reg << endl;
            return false;
        }

        auto temp = (Q_uu_reg).ldlt();
        m_ctrl_ctrl I;
        I.setIdentity();
        m_ctrl_ctrl Q_uu_inv = temp.solve(I);

        k[t] = -Q_uu_inv * Q_u;
        K[t] = -Q_uu_inv * Q_ux_reg;

        V_x = Q_x + (K[t].transpose() * (Q_uu * k[t])) + (K[t].transpose() * Q_u) + (Q_ux.transpose() * k[t]);

        V_xx = Q_xx + (K[t].transpose() * (Q_uu * K[t])) + (K[t].transpose() * Q_ux) + (Q_ux.transpose() * K[t]);

        V_xx = (V_xx + V_xx.transpose()) / 2;

    }

    return true;
}

bool iLQR::isMatrixPD(Ref<MatrixXd> matrix){
    bool matrixPD = true;
    //TODO implement cholesky decomp for PD check and maybe use result for inverse Q_uu

    Eigen::LLT<Eigen::MatrixXd> lltOfA(matrix); // compute the Cholesky decomposition of the matrix
    if(lltOfA.info() == Eigen::NumericalIssue)
    {
        matrixPD = false;
        //throw std::runtime_error("Possibly non semi-positive definitie matrix!");
    }

    return matrixPD;
}

float iLQR::forwardsPass(float oldCost){
    // TODO check if ths needs to be changed to a standard vector?
    float alpha = 1.0;
    float newCost = 0;
    bool costReduction = false;
    int alphaCount = 0;

    while(!costReduction){
        cpMjData(model, mdata, d_init);
        newCost = 0;
        m_state stateFeedback;
        m_state X;
        m_ctrl U_last;

        for(int t = 0; t < ILQR_HORIZON_LENGTH - 1; t++){

            X = X_old[t].replicate(1, 1);
            X_new[t] = modelTranslator->returnState(mdata);
            U_last = U_old[t].replicate(1, 1);

            stateFeedback = X_new[t] - X;
            m_ctrl feedBackGain = K[t] * stateFeedback;

            if(alphaCount == 9){
                U_new[t] = U_last;
            }
            else{
                U_new[t] = U_last + (alpha * k[t]) + feedBackGain;
            }


//            cout << "k " << k[t] << endl;
//            cout << "K " << K[t] << endl;
//            cout << "feedBackGain " << feedBackGain << endl;
//            cout << "U_last " << U_last << endl;
//            cout << "New U: " << U_new[t] << endl;


            modelTranslator->setControls(mdata, U_new[t]);

            // calc current state cost and keep running tally
            float currentCost = modelTranslator->getCost(mdata, t, ILQR_HORIZON_LENGTH);

            newCost += (currentCost * ILQR_DT);

            for(int i = 0; i < NUM_MJSTEPS_PER_CONTROL; i++){
                modelTranslator->stepModel(mdata, 1);
            }

//            if(t % 20 == 0){
//                cout << "old control: " << U_last << endl;
//                cout << "new control: " << U_new[t] << endl;
//            }

//            if(t % 20 == 0){
//                mjrRect viewport = { 0, 0, 0, 0 };
//                glfwGetFramebufferSize(window, &viewport.width, &viewport.height);
//
//                // update scene and render
//                mjv_updateScene(model, mdata, &opt, NULL, &cam, mjCAT_ALL, &scn);
//                mjr_render(viewport, &scn, &con);
//
//                // swap OpenGL buffers (blocking call due to v-sync)
//                glfwSwapBuffers(window);
//
//                // process pending GUI events, call GLFW callbacks
//                glfwPollEvents();
//            }


        }

        if(newCost < oldCost){
            costReduction = true;
        }
        else{
            alpha = alpha - 0.1;
            alphaCount++;
            if(alpha <= 0){
                newCost = oldCost;
                break;
            }
        }
    }

    m_state termStateBest = modelTranslator->returnState(mdata);
    //cout << "terminal state best: " << endl << termStateBest << endl;
    cout << "best alpha was " << alpha << endl;
//    cout << "best cost was " << newCost << endl;
//    cout << "-------------------- END FORWARDS PASS ------------------------" << endl;

    return newCost;
}

bool iLQR::checkForConvergence(float newCost, float oldCost){
    bool convergence = false;
    m_state terminalState = modelTranslator->returnState(mdata);

    std::cout << "--------------------------------------------------" <<  std::endl;
    std::cout << "New cost: " << newCost <<  std::endl;
//    double cubeX = terminalState(7);
//    double cubeY = terminalState(8);
//    std::cout << "terminal state is, cube X: " << cubeX << " cube Y: " << cubeY << endl;

    double cubeX = terminalState(7);
    double cubeY = terminalState(8);
    std::cout << "terminal state is, cube X: " << cubeX << " cube Y: " << cubeY << endl;

    numIterations++;
    float costGrad = (oldCost - newCost)/newCost;

    if((numIterations > 2) && costGrad < epsConverge) {
        convergence = true;
        cout << "ilQR converged, num Iterations: " << numIterations << " final cost: " << newCost << endl;
    }

    // store new controls
    for(int i = 0; i < ILQR_HORIZON_LENGTH; i++){
        U_old[i] = U_new[i].replicate(1, 1);
    }

    return convergence;
}

void iLQR::lineariseDynamicsSerial_trial(Ref<MatrixXd> _A, Ref<MatrixXd> _B, int controlNum, float ilqr_dt){

    // Initialise variables
    static int nwarmup = 3;
    float eps = 1e-2;

    _A.block(0, 0, DOF, DOF).setIdentity();
    _A.block(0, DOF, DOF, DOF).setIdentity();
    _A.block(0, DOF, DOF, DOF) *= model->opt.timestep;
    _B.setZero();

    // Initialise matrices for forwards dynamics
    m_dof_dof dqaccdq;
    m_dof_dof dqaccdqvel;
    m_dof_ctrl dqaccdctrl;

    m_dof acellDec;
    m_dof acellInc;

    // Create a copy of the current data that we want to differentiate around
    mjData *saveData;
    saveData = mj_makeData(model);
    cpMjData(model, saveData, dArray[controlNum]);

    // Allocate memory for variables
    mjtNum* warmstart = mj_stackAlloc(saveData, DOF);

    // Compute mj_forward once with no skips
    mj_forward(model, saveData);

    // Compute mj_forward a few times to allow optimiser to get a more accurate value for qacc
    // skips position and velocity stages (TODO LOOK INTO IF THIS IS NEEDED FOR MY METHOD)
    for( int rep=1; rep<nwarmup; rep++ )
        mj_forwardSkip(model, saveData, mjSTAGE_VEL, 1);


    // save output for center point and warmstart (needed in forward only)
    mju_copy(warmstart, saveData->qacc_warmstart, DOF);

    // CALCULATE dqaccdctrl
    for(int i = 0; i < NUM_CTRL; i++){
        saveData->ctrl[i] = dArray[controlNum]->ctrl[i] + eps;

        // evaluate dynamics, with center warmstart
        mju_copy(saveData->qacc_warmstart, warmstart, model->nv);
        mj_forwardSkip(model, saveData, mjSTAGE_VEL, 1);

        // copy and store +perturbation
        acellInc = modelTranslator->returnAccelerations(saveData);

        // perturb selected target -
        cpMjData(model, saveData, dArray[controlNum]);
        saveData->ctrl[i] = dArray[controlNum]->ctrl[i] - eps;

        // evaluate dynamics, with center warmstart
        mju_copy(saveData->qacc_warmstart, warmstart, model->nv);
        mj_forwardSkip(model, saveData, mjSTAGE_VEL, 1);

        acellDec = modelTranslator->returnAccelerations(saveData);

        for(int j = 0; j < DOF; j++){
            dqaccdctrl(j, i) = (acellInc(j) - acellDec(j))/(2*eps);
        }

        // undo pertubation
        cpMjData(model, saveData, dArray[controlNum]);

    }

    // CALCULATE dqaccdvel
    for(int i = 0; i < DOF; i++){
        // perturb velocity +

        modelTranslator->perturbVelocity(saveData, dArray[controlNum], i, eps);

        // evaluate dynamics, with center warmstart
        mju_copy(saveData->qacc_warmstart, warmstart, model->nv);
        mj_forwardSkip(model, saveData, mjSTAGE_POS, 1);

        // copy and store +perturbation
        acellInc = modelTranslator->returnAccelerations(saveData);
        //cout << "acellInc " << endl << acellInc << endl;

        // perturb velocity -
        modelTranslator->perturbVelocity(saveData, dArray[controlNum], i, -eps);

        // evaluate dynamics, with center warmstart
        mju_copy(saveData->qacc_warmstart, warmstart, model->nv);
        mj_forwardSkip(model, saveData, mjSTAGE_POS, 1);

        acellDec = modelTranslator->returnAccelerations(saveData);
        //cout << "acellDec " << endl << acellDec << endl;

        // compute column i of derivative 1
        for(int j = 0; j < DOF; j++){
            dqaccdqvel(j, i) = (acellInc(j) - acellDec(j))/(2*eps);
        }
        //cout << "dq/dvel " << endl << dqaccdqvel << endl;

        // undo perturbation
        cpMjData(model, saveData, dArray[controlNum]);
    }

    // CALCULATE dqaccdcqpos
    for(int i = 0; i < DOF; i++){
        // perturb position +
        modelTranslator->perturbPosition(saveData, dArray[controlNum], i, eps);
//        cout << "qacc Number 1: ";
//        for(int j = 0; j < NUM_CTRL; j++){
//            cout << saveData->qacc[j] << " ";
//        }
//        cout << endl;

        // evaluate dynamics, with center warmstart
        mju_copy(saveData->qacc_warmstart, warmstart, model->nv);
        mj_forwardSkip(model, saveData, mjSTAGE_NONE, 1);
        acellInc = modelTranslator->returnAccelerations(saveData);
        //cout << "accel inc is: " << endl << acellInc << endl;
//        for( int rep=1; rep<5; rep++ ){
//            mj_forwardSkip(model, saveData, mjSTAGE_NONE, 1);
//            acellInc = modelTranslator->returnAccelerations(saveData);
//            cout << "accel inc is: " << endl << acellInc << endl;
//        }


        // copy and store +perturbation
        acellInc = modelTranslator->returnAccelerations(saveData);
//        cout << "qacc Number 1: ";
//        for(int j = 0; j < NUM_CTRL; j++){
//            cout << saveData->qacc[j] << " ";
//        }
//        cout << endl;

        // perturb position -
        modelTranslator->perturbPosition(saveData, dArray[controlNum], i, -eps);

        // evaluate dynamics, with center warmstart
        mju_copy(saveData->qacc_warmstart, warmstart, model->nv);
        mj_forwardSkip(model, saveData, mjSTAGE_NONE, 1);

        // Additional mj_forward steps when computing dqac/dqpos
        for( int rep=1; rep<nwarmup; rep++ )
            mj_forwardSkip(model, saveData, mjSTAGE_NONE, 1);

        acellDec = modelTranslator->returnAccelerations(saveData);
        //cout << "accel inc is: " << endl << acellInc << endl;
//        cout << "qacc from output ";
//        for(int j = 0; j < NUM_CTRL; j++){
//            cout << output[j] << " ";
//        }
//        cout << endl;
//


        // compute column i of derivative 1
        for(int j = 0; j < DOF; j++){
            dqaccdq(j, i) = (acellInc(j) - acellDec(j))/(2*eps);
        }

        // undo perturbation
        cpMjData(model, saveData, dArray[controlNum]);
    }

    mj_deleteData(saveData);

//    cout << " dqaccdqis: " << dqaccdq << endl;
//    cout << " dqaccdqvel: " << dqaccdqvel << endl;
    //cout << " dqaccdctrl: " << dqaccdctrl << endl;

    _A.block(DOF, 0, DOF, DOF) = dqaccdq * ilqr_dt;
    _A.block(DOF, DOF, DOF, DOF).setIdentity();
    _A.block(DOF, DOF, DOF, DOF) += dqaccdqvel * ilqr_dt;
    _B.block(DOF, 0, DOF, NUM_CTRL) = dqaccdctrl * ilqr_dt;

    //cout << "A matrix is: " << _A << endl;
//    cout << " B Mtrix is: " << _B << endl;

}

void iLQR::lineariseDynamicsSerial_trial_step(Ref<MatrixXd> _A, Ref<MatrixXd> _B, mjData *linearisedData, float dt){
    // Initialise variables
    static int nwarmup = 3;

    float epsControls = 1e-5;
    float epsVelocities = 1e-5;
    float epsPos[DOF] = {1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5};

    _A.block(0, 0, DOF, DOF).setIdentity();
    _A.block(0, DOF, DOF, DOF).setIdentity();
    _A.block(0, DOF, DOF, DOF) *= MUJOCO_DT;
    _B.setZero();

    // Initialise matrices for forwards dynamics
    m_dof_dof dqveldq;
    m_dof_dof dqveldqvel;
    m_dof_dof dqposdqvel;
    m_dof_ctrl dqveldctrl;
    m_dof_dof dqaccdqvel;
    m_dof_dof dqaccdq;
    m_dof_ctrl dqaccdctrl;

    m_dof velDec;
    m_dof velInc;
    m_dof posInc;
    m_dof posDec;
    m_dof acellInc, acellDec;

    // Create a copy of the current data that we want to differentiate around
    mjData *saveData;
    saveData = mj_makeData(model);
    cpMjData(model, saveData, linearisedData);

    // Allocate memory for variables
    mjtNum* warmstart = mj_stackAlloc(saveData, DOF);

//    cout << "accel before: " << saveData->qacc[0] << endl;
//    // Compute mj_forward once with no skips
    mj_forward(model, saveData);
//    cout << "accel before: " << saveData->qacc[0] << endl;

    // Compute mj_forward a few times to allow optimiser to get a more accurate value for qacc
    // skips position and velocity stages (TODO LOOK INTO IF THIS IS NEEDED FOR MY METHOD)
    for( int rep=1; rep<nwarmup; rep++ )
        mj_forwardSkip(model, saveData, mjSTAGE_VEL, 1);


    // save output for center point and warmstart (needed in forward only)
    mju_copy(warmstart, saveData->qacc_warmstart, DOF);

    // CALCULATE dqveldctrl

    //calculate dqacc/dctrl
    for(int i = 0; i < NUM_CTRL; i++){
        saveData->ctrl[i] = linearisedData->ctrl[i] + epsControls;

        // evaluate dynamics, with center warmstart
        mju_copy(saveData->qacc_warmstart, warmstart, model->nv);
        mj_forwardSkip(model, saveData, mjSTAGE_VEL, 1);

        // copy and store +perturbation
        acellInc = modelTranslator->returnAccelerations(saveData);

        // perturb selected target -
        cpMjData(model, saveData, linearisedData);
        saveData->ctrl[i] = linearisedData->ctrl[i] - epsControls;

        // evaluate dynamics, with center warmstart
        mju_copy(saveData->qacc_warmstart, warmstart, model->nv);
        mj_forwardSkip(model, saveData, mjSTAGE_VEL, 1);

        acellDec = modelTranslator->returnAccelerations(saveData);

        for(int j = 0; j < DOF; j++){
            dqaccdctrl(j, i) = (acellInc(j) - acellDec(j))/(2*epsControls);
        }

        // undo pertubation
        cpMjData(model, saveData, linearisedData);

    }

    // CALCULATE dqaccdvel
//    for(int i = 0; i < DOF; i++){
//        // perturb velocity +
//
//        modelTranslator->perturbVelocity(saveData, linearisedData, i, epsVelocities);
//
//        // evaluate dynamics, with center warmstart
//        mju_copy(saveData->qacc_warmstart, warmstart, model->nv);
//        mj_forwardSkip(model, saveData, mjSTAGE_POS, 1);
//
//        // copy and store +perturbation
//        acellInc = modelTranslator->returnAccelerations(saveData);
//        //cout << "acellInc " << endl << acellInc << endl;
//
//        // perturb velocity -
//        modelTranslator->perturbVelocity(saveData, linearisedData, i, -epsVelocities);
//
//        // evaluate dynamics, with center warmstart
//        mju_copy(saveData->qacc_warmstart, warmstart, model->nv);
//        mj_forwardSkip(model, saveData, mjSTAGE_POS, 1);
//
//        acellDec = modelTranslator->returnAccelerations(saveData);
//        //cout << "acellDec " << endl << acellDec << endl;
//
//        // compute column i of derivative 1
//        for(int j = 0; j < DOF; j++){
//            dqaccdqvel(j, i) = (acellInc(j) - acellDec(j))/(2*epsVelocities);
//        }
//        //cout << "dq/dvel " << endl << dqaccdqvel << endl;
//
//        // undo perturbation
//        cpMjData(model, saveData, linearisedData);
//    }

    // CALCULATE dqveldvel
    for(int i = 0; i < DOF; i++){
        // perturb velocity +
        modelTranslator->perturbVelocity(saveData, linearisedData, i, epsVelocities);

        // evaluate dynamics, with center warmstart
        mju_copy(saveData->qacc_warmstart, warmstart, model->nv);
        modelTranslator->stepModel(saveData, 1);

        // copy and store +perturbation
        velInc = modelTranslator->returnVelocities(saveData);
        posInc = modelTranslator->returnPositions(saveData);

        // undo perturbation
        cpMjData(model, saveData, linearisedData);

        // perturb //    for(int i = 0; i < DOF; i++){
        // perturb velocity +
        modelTranslator->perturbVelocity(saveData, linearisedData, i, epsVelocities);

        // evaluate dynamics, with center warmstart
        mju_copy(saveData->qacc_warmstart, warmstart, model->nv);
        modelTranslator->perturbVelocity(saveData, linearisedData, i, -epsVelocities);

        // evaluate dynamics, with center warmstart
        mju_copy(saveData->qacc_warmstart, warmstart, model->nv);
        modelTranslator->stepModel(saveData, 1);

        velDec = modelTranslator->returnVelocities(saveData);
        posDec = modelTranslator->returnPositions(saveData);
        //cout << "acellDec " << endl << velDec << endl;

        // compute column i of derivative 1
        for(int j = 0; j < DOF; j++){
            double diffScaled = (velInc(j) - velDec(j));
            //double diffScaled = (velInc(j) - velDec(j));
            dqveldqvel(j, i) = diffScaled/(2*epsVelocities);

            dqposdqvel(j, i) = (posInc(j) - posDec(j)) / (2*epsVelocities);
        }
        //cout << "dqaccdqvel " << endl << dqaccdqvel << endl;

        // undo perturbation
        cpMjData(model, saveData, linearisedData);
    }

    // CALUCLATE dqaccdqpos
    for(int i = 0; i < DOF; i++){
        // perturb position +
        modelTranslator->perturbPosition(saveData, linearisedData, i, epsPos[i]);

        // evaluate dynamics, with center warmstart
        mju_copy(saveData->qacc_warmstart, warmstart, model->nv);
        mj_forwardSkip(model, saveData, mjSTAGE_NONE, 1);
        acellInc = modelTranslator->returnAccelerations(saveData);
        //cout << "accel inc is: " << endl << acellInc << endl;
//        for( int rep=1; rep<5; rep++ ){
//            mj_forwardSkip(model, saveData, mjSTAGE_NONE, 1);
//            acellInc = modelTranslator->returnAccelerations(saveData);
//            cout << "accel inc is: " << endl << acellInc << endl;
//        }

//        cout << "qacc Number 1: ";
//        for(int j = 0; j < NUM_CTRL; j++){
//            cout << saveData->qacc[j] << " ";
//        }
//        cout << endl;

        // perturb position -
        modelTranslator->perturbPosition(saveData, linearisedData, i, -epsPos[i]);

        // evaluate dynamics, with center warmstart
        mju_copy(saveData->qacc_warmstart, warmstart, model->nv);
        mj_forwardSkip(model, saveData, mjSTAGE_NONE, 1);

        // Additional mj_forward steps when computing dqac/dqpos
        for( int rep=1; rep<nwarmup; rep++ )
            mj_forwardSkip(model, saveData, mjSTAGE_NONE, 1);

        acellDec = modelTranslator->returnAccelerations(saveData);
        //cout << "acellDec is: " << endl << acellDec << endl;
//        cout << "qacc from output ";
//        for(int j = 0; j < NUM_CTRL; j++){
//            cout << output[j] << " ";
//        }
//        cout << endl;
//
        // compute column i of derivative 1
        for(int j = 0; j < DOF; j++){
            dqaccdq(j, i) = (acellInc(j) - acellDec(j))/(2*epsPos[i]);
        }

        //cout << "dqaccdq " << endl << dqaccdq << endl;

        // undo perturbation
        cpMjData(model, saveData, linearisedData);
    }

//CALCULATE dqveldqpos
//    for(int i = 0; i < DOF; i++){
//        // perturb position +
//        modelTranslator->perturbPosition(saveData, linearisedData, i, epsPos[i]);
//
//        // evaluate dynamics, with center warmstart
//        mju_copy(saveData->qacc_warmstart, warmstart, model->nv);
//        modelTranslator->stepModel(saveData, 1);
//        velInc = modelTranslator->returnVelocities(saveData);
//        //cout << "vel inc: " << velInc << endl;
//
//        // undo perturbation
//        cpMjData(model, saveData, linearisedData);
//
//        // perturb position -
//        modelTranslator->perturbPosition(saveData, linearisedData, i, -epsPos[i]);
//
//        // evaluate dynamics, with center warmstart
//        mju_copy(saveData->qacc_warmstart, warmstart, model->nv);
//        modelTranslator->stepModel(saveData, 1);
//
//        velDec = modelTranslator->returnVelocities(saveData);
//        //cout << "vel dec: " << velDec << endl;
//
//        // compute column i of derivative 1
//        for(int j = 0; j < DOF; j++){
//            double diffScaled = (velInc(j) - velDec(j));
//            //double diffScaled = (velInc(j) - velDec(j));
//            dqveldq(j, i) = (diffScaled)/(2*epsPos[i]);
//        }
//        //cout << "dqvel by dq: " << dqveldq << endl;
//
//        // undo perturbation
//        cpMjData(model, saveData, linearisedData);
//    }

    //cout << "dqacc method: " << endl << scaleddqacc << endl;
    //cout << "step method: " << endl << dqveldq << endl;


    mj_deleteData(saveData);

//    cout << " dqveldq is: " << dqveldq << endl;
//    cout << " dqveldqvel: " << dqveldqvel << endl;
//    cout << " dveldctrl: " << dqveldctrl << endl;

    //_A.block(0, DOF, DOF, DOF) = dqposdqvel;

    _A.block(DOF, 0, DOF, DOF) = (dqaccdq * MUJOCO_DT);
    _A.block(DOF, DOF, DOF, DOF).setIdentity();
    _A.block(DOF, DOF, DOF, DOF) = dqveldqvel;
    _B.block(DOF, 0, DOF, NUM_CTRL) = (dqaccdctrl * MUJOCO_DT);

//    cout << "A matrix is: " << _A << endl;
//    cout << " B Mtrix is: " << _B << endl;

}

m_ctrl iLQR::returnDesiredControl(int controlIndex, bool finalControl){
    if(finalControl){
        return initControls[controlIndex];
    }
    else{
        return finalControls[controlIndex];
    }
}

void iLQR::setInitControls(std::vector<m_ctrl> _initControls){
    for(int i = 0; i < ILQR_HORIZON_LENGTH; i++){
        initControls[i] = _initControls[i].replicate(1,1);
        U_old[i] = _initControls[i].replicate(1,1);
    }
}

void iLQR::makeDataForOptimisation(){
    // Set initial state and run mj_step several times to stabilise system
    modelTranslator->setState(mdata, X0);
    for(int i = 0; i < 200; i++){
        modelTranslator->stepModel(mdata, 1);
    }

    d_init = mj_makeData(model);
    cpMjData(model, d_init, mdata);

//    for(int i = 0; i <= ILQR_HORIZON_LENGTH; i++){
//        // populate dArray with mujoco data objects from start of trajec until end
//        dArray[i] = mj_makeData(model);
//
//        for(int k = 0; k < NUM_CTRL; k++){
//            //initControls[i](k) = desiredControls(k) + mdata->qfrc_bias[k];
//            mdata->ctrl[k] = initControls[i](k);
//        }
//        // copy current data into current data array at correct timestep
//        cpMjData(model, dArray[i], mdata);
//
//        // step simulation with initialised controls
//        for(int i = 0; i < NUM_MJSTEPS_PER_CONTROL; i++){
//            modelTranslator->stepModel(mdata, 1);
//        }
//
//        if(i == 2500){
//            int a;
//        }
//    }
//    dArray[ILQR_HORIZON_LENGTH] = mj_makeData(model);
//    cpMjData(model, dArray[ILQR_HORIZON_LENGTH], mdata);

    // reset mdata back to initial state
    cpMjData(model, mdata, d_init);
}


void cpMjData(const mjModel* m, mjData* d_dest, const mjData* d_src){
    d_dest->time = d_src->time;
    mju_copy(d_dest->qpos, d_src->qpos, m->nq);
    mju_copy(d_dest->qvel, d_src->qvel, m->nv);
    mju_copy(d_dest->qacc, d_src->qacc, m->nv);
    mju_copy(d_dest->qacc_warmstart, d_src->qacc_warmstart, m->nv);
    mju_copy(d_dest->qfrc_applied, d_src->qfrc_applied, m->nv);
    mju_copy(d_dest->xfrc_applied, d_src->xfrc_applied, 6*m->nbody);
    mju_copy(d_dest->ctrl, d_src->ctrl, m->nu);
}


// CALCULATE dqveldctrl
//    for(int i = 0; i < NUM_CTRL; i++){
//        saveData->ctrl[i] = linearisedData->ctrl[i] + epsControls[i];
//
//        // evaluate dynamics, with center warmstart
//        mju_copy(saveData->qacc_warmstart, warmstart, model->nv);
//        for(int j = 0; j < numStepsLinearisation; j++){
//            modelTranslator->stepModel(saveData, 1);
//            //mj_step(model, mdata);
//        }
//
//        // copy and store +perturbation
//        velInc = modelTranslator->returnVelocities(saveData);
//        //cout << "velinc " << endl << velInc << endl;
//
//        // undo perturbation
//        cpMjData(model, saveData, linearisedData);
//
//        // perturb selected target -
//        cpMjData(model, saveData, linearisedData);
//        saveData->ctrl[i] = linearisedData->ctrl[i] - epsControls[i];
//
//        // evaluate dynamics, with center warmstart
//        mju_copy(saveData->qacc_warmstart, warmstart, model->nv);
//        for(int j = 0; j < numStepsLinearisation; j++){
//            modelTranslator->stepModel(saveData, 1);
//            //mj_step(model, saveData);
//        }
//
//        //cout << "ctrl dec " << endl << saveData->ctrl[i] << endl;
//        velDec = modelTranslator->returnVelocities(saveData);
//        //cout << "veldec " << endl << velDec << endl;
//
//        for(int j = 0; j < DOF; j++){
//            double diffScaled = (velInc(j) - velDec(j)) / numStepsLinearisation;
//            //double diffScaled = (velInc(j) - velDec(j));
//            dqveldctrl(j, i) = diffScaled/(2*epsControls[i]);
//        }
//        //cout << "dqveldctrl " << endl << dqveldctrl << endl;
//
//        // undo pertubation
//        cpMjData(model, saveData, linearisedData);
//
//    }

//CALCULATE dqveldqpos
//    for(int i = 0; i < DOF; i++){
//        // perturb position +
//        modelTranslator->perturbPosition(saveData, linearisedData, i, epsPos[i]);
//
//        // evaluate dynamics, with center warmstart
//        mju_copy(saveData->qacc_warmstart, warmstart, model->nv);
//        for(int j = 0; j < numStepsLinearisation; j++){
//            modelTranslator->stepModel(saveData, 1);
//        }
//        velInc = modelTranslator->returnVelocities(saveData);
//        //cout << "vel inc: " << velInc << endl;
//
//        // undo perturbation
//        cpMjData(model, saveData, linearisedData);
//
//        // perturb position -
//        modelTranslator->perturbPosition(saveData, linearisedData, i, -epsPos[i]);
//
//        // evaluate dynamics, with center warmstart
//        mju_copy(saveData->qacc_warmstart, warmstart, model->nv);
//        for(int j = 0; j < numStepsLinearisation; j++){
//            modelTranslator->stepModel(saveData, 1);
//        }
//
//        velDec = modelTranslator->returnVelocities(saveData);
//        //cout << "vel dec: " << velDec << endl;
//
//        // compute column i of derivative 1
//        for(int j = 0; j < DOF; j++){
//            double diffScaled = (velInc(j) - velDec(j)) / numStepsLinearisation;
//            //double diffScaled = (velInc(j) - velDec(j));
//            dqveldq(j, i) = (diffScaled)/(2*epsPos[i]);
//        }
//        //cout << "dqvel by dq: " << dqveldq << endl;
//
//        // undo perturbation
//        cpMjData(model, saveData, linearisedData);
//    }

// CALCULATE dqveldvel
//    for(int i = 0; i < DOF; i++){
//        // perturb velocity +
//        modelTranslator->perturbVelocity(saveData, linearisedData, i, epsVelocities[i]);
//
//        // evaluate dynamics, with center warmstart
//        mju_copy(saveData->qacc_warmstart, warmstart, model->nv);
//        modelTranslator->stepModel(saveData, 1);
//
//        // copy and store +perturbation
//        velInc = modelTranslator->returnVelocities(saveData);
//        posInc = modelTranslator->returnPositions(saveData);
//
//        // undo perturbation
//        cpMjData(model, saveData, linearisedData);
//
//        // perturb velocity -
//        modelTranslator->perturbVelocity(saveData, linearisedData, i, -epsVelocities[i]);
//
//        // evaluate dynamics, with center warmstart
//        mju_copy(saveData->qacc_warmstart, warmstart, model->nv);
//        modelTranslator->stepModel(saveData, 1);
//
//        velDec = modelTranslator->returnVelocities(saveData);
//        posDec = modelTranslator->returnPositions(saveData);
//        //cout << "acellDec " << endl << velDec << endl;
//
//        // compute column i of derivative 1
//        for(int j = 0; j < DOF; j++){
//            double diffScaled = (velInc(j) - velDec(j));
//            //double diffScaled = (velInc(j) - velDec(j));
//            dqveldqvel(j, i) = diffScaled/(2*epsVelocities[i]);
//
//            dqposdqvel(j, i) = (posInc(j) - posDec(j)) / (2*epsVelocities[i]);
//        }
//        //cout << "dqaccdqvel " << endl << dqaccdqvel << endl;
//
//        // undo perturbation
//        cpMjData(model, saveData, linearisedData);
//    }

// calculate dqpos/dqvel
//for(int i = 0; i < DOF; i++){
//// perturb velocity +
//modelTranslator->perturbVelocity(saveData, linearisedData, i, epsVelocities[i]);
//
//// evaluate dynamics, with center warmstart
//mju_copy(saveData->qacc_warmstart, warmstart, model->nv);
//modelTranslator->stepModel(saveData, 1);
//
//// copy and store +perturbation
//posInc = modelTranslator->returnPositions(saveData);
//
//// undo perturbation
//cpMjData(model, saveData, linearisedData);
//
//// perturb velocity -
//modelTranslator->perturbVelocity(saveData, linearisedData, i, -epsVelocities[i]);
//
//// evaluate dynamics, with center warmstart
//mju_copy(saveData->qacc_warmstart, warmstart, model->nv);
//modelTranslator->stepModel(saveData, 1);
//
//posDec = modelTranslator->returnPositions(saveData);
////cout << "acellDec " << endl << velDec << endl;
//
//// compute column i of derivative 1
//for(int j = 0; j < DOF; j++){
//dqposdqvel(j, i) = (posInc(j) - posDec(j)) / (2*epsVelocities[i]);
//}
////cout << "dqaccdqvel " << endl << dqaccdqvel << endl;
//
//// undo perturbation
//cpMjData(model, saveData, linearisedData);
//}