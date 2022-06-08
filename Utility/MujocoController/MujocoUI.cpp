//
// Created by davem on 20/01/2022.
//
#include "MujocoUI.h"
#include "../../iLQR/iLQR_dataCentric.h"

extern mjModel *model;                  // MuJoCo model
extern mjData *mdata;                   // MuJoCo data
extern mjvCamera cam;                   // abstract camera
extern mjvScene scn;                    // abstract scene
extern mjvOption opt;			        // visualization options
extern mjrContext con;				    // custom GPU context
extern GLFWwindow *window;
extern MujocoController *globalMujocoController;
extern iLQR* optimiser;
extern frankaModel* modelTranslator;

std::vector<m_ctrl> testInitControls;
mjData* d_init_test;

bool button_left = false;
bool button_middle = false;
bool button_right = false;
double lastx = 0;
double lasty = 0;

// keyboard callback
void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods){
    // backspace: reset simulation
    if (act == GLFW_PRESS && key == GLFW_KEY_BACKSPACE)
    {
        mj_resetData(model, mdata);
        mj_forward(model, mdata);
    }

    float ctrlChange = 0.02;
    if(key == 265 && act == GLFW_PRESS ){
        // forwards arrow
        mdata->ctrl[0] += ctrlChange;
    }
    else if(key == 263 && act == GLFW_PRESS ){
        // back arrow
        mdata->ctrl[0] -= ctrlChange;
    }
    else if(key == 264 && act == GLFW_PRESS ){
        // left arrow
        mdata->ctrl[1] += ctrlChange;
    }
    else if(key == 262 && act == GLFW_PRESS ){
        // right arrow
        mdata->ctrl[1] -= ctrlChange;
    }
    else if(key == 257 && act == GLFW_PRESS ){
        //enter key
        m_state collState;
        //collState = modelTranslator->returnState(mdata);
        //collState(0) += 0.00001;
        collState << 0, 0, 0.02, 0, 0, 0, 0, 0;

        modelTranslator->setState(mdata, collState);
        mj_forward(model, mdata);
        m_dof accels;
        accels = modelTranslator->returnAccelerations(mdata);
        cout << "accelerations: " << endl << accels << endl;
        mju_copy(mdata->qacc_warmstart, mdata->qacc, model->nv);
        for( int rep=1; rep<5; rep++ ){
            mju_copy(mdata->qacc_warmstart, mdata->qacc, model->nv);
            mj_forward(model, mdata);
            //mj_forwardSkip(model, mdata, mjSTAGE_VEL, 1);
            accels = modelTranslator->returnAccelerations(mdata);
            cout << "accelerations: " << endl << accels << endl;

        }

    }

    cout << "x direction ctrl: " << mdata->ctrl[0] << " y direction ctrl: " << mdata->ctrl[1] << endl;

}


// mouse button callback
void mouse_button(GLFWwindow* window, int button, int act, int mods){
    // update button state
    button_left = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
    button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS);
    button_right = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);

    // update mouse position
    glfwGetCursorPos(window, &lastx, &lasty);
}


// mouse move callback
void mouse_move(GLFWwindow* window, double xpos, double ypos){
    // no buttons down: nothing to do
    if (!button_left && !button_middle && !button_right)
        return;

    // compute mouse displacement, save
    double dx = xpos - lastx;
    double dy = ypos - lasty;
    lastx = xpos;
    lasty = ypos;

    // get current window size
    int width, height;
    glfwGetWindowSize(window, &width, &height);

    // get shift key state
    bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                      glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);

    // determine action based on mouse button
    mjtMouse action;
    if (button_right)
        action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    else if (button_left)
        action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    else
        action = mjMOUSE_ZOOM;

    // move camera
    mjv_moveCamera(model, action, dx / height, dy / height, &scn, &cam);
}


// scroll callback
void scroll(GLFWwindow* window, double xoffset, double yoffset){
    // emulate vertical mouse motion = 5% of window height
    mjv_moveCamera(model, mjMOUSE_ZOOM, 0, -0.05 * yoffset, &scn, &cam);
}

void windowCloseCallback(GLFWwindow * /*window*/) {
    // Use this flag if you wish not to terminate now.
    // glfwSetWindowShouldClose(window, GLFW_FALSE);
    mjv_freeScene(&scn);
    mjr_freeContext(&con);

    // free MuJoCo model and data, deactivate
    mj_deleteData(mdata);
    mj_deleteModel(model);
    mj_deactivate();
}

void setupMujocoWorld(){
    char error[1000];

    model = mj_loadXML("franka_emika/franka_panda.xml", NULL, error, 1000);

    if( !model ) {
        printf("%s\n", error);
    }

    // make data corresponding to model
    mdata = mj_makeData(model);

    // init GLFW, create window, make OpenGL context current, request v-sync
    // init GLFW
    if (!glfwInit())
        mju_error("Could not initialize GLFW");
    window = glfwCreateWindow(1200, 900, "iLQR_Testing", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // initialize visualization data structures
    mjv_defaultCamera(&cam);
    //mjv_defaultPerturb(&pert);				// what data type for pert?
    mjv_defaultOption(&opt);
    mjr_defaultContext(&con);

    //model->opt.gravity[2] = 0;
    //model->opt.integrator = mjINT_EULER;

    // create scene and context
    mjv_makeScene(model, &scn, 2000);
    mjr_makeContext(model, &con, mjFONTSCALE_150);

    // install GLFW mouse and keyboard callbacks
    glfwSetKeyCallback(window, keyboard);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetScrollCallback(window, scroll);
    glfwSetWindowCloseCallback(window, windowCloseCallback);
}

void render(){
    // run main loop, target real-time simulation and 60 fps rendering
    int controlNum = 0;
    bool showFinalControls = true;
    m_ctrl nextControl;
    cpMjData(model, mdata, optimiser->d_init);
    while (!glfwWindowShouldClose(window))
    {
        // advance interactive simulation for 1/60 sec
        //  Assuming MuJoCo can simulate faster than real-time, which it usually can,
        //  this loop will finish on time for the next frame to be rendered at 60 fps.
        //  Otherwise add a cpu timer and exit this loop when it is time to render.
        mjtNum simstart = mdata->time;
        while (mdata->time - simstart < 1.0 / 60.0){
            nextControl = optimiser->returnDesiredControl(controlNum, showFinalControls);
            for(int k = 0; k < NUM_CTRL; k++){
                mdata->ctrl[k] = nextControl(k);
            }


            for (int i = 0; i < NUM_MJSTEPS_PER_CONTROL; i++) {
                modelTranslator->stepModel(mdata, 1);
            }

            controlNum++;

            if(controlNum >= ILQR_HORIZON_LENGTH){
                controlNum = 0;
                cpMjData(model, mdata, optimiser->d_init);
                simstart = mdata->time;
                showFinalControls = 1 - showFinalControls;
            }
        }

        // get framebuffer viewport
        mjrRect viewport = { 0, 0, 0, 0 };
        glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

        // update scene and render
        mjv_updateScene(model, mdata, &opt, NULL, &cam, mjCAT_ALL, &scn);
        mjr_render(viewport, &scn, &con);

        // swap OpenGL buffers (blocking call due to v-sync)
        glfwSwapBuffers(window);

        // process pending GUI events, call GLFW callbacks
        glfwPollEvents();
    }

}

void render_simpleTest(){
    // run main loop, target real-time simulation and 60 fps rendering
    m_state currentState;
    int controlNum = 0;
    cpMjData(model, mdata, d_init_test);
    while (!glfwWindowShouldClose(window))
    {
        // advance interactive simulation for 1/60 sec
        //  Assuming MuJoCo can simulate faster than real-time, which it usually can,
        //  this loop will finish on time for the next frame to be rendered at 60 fps.
        //  Otherwise add a cpu timer and exit this loop when it is time to render.
        mjtNum simstart = mdata->time;
        while (mdata->time - simstart < 1.0 / 60.0) {
            for (int k = 0; k < NUM_CTRL; k++) {
                mdata->ctrl[k] = testInitControls[controlNum](k);
            }

//            const std::string boxstring = "unactuated_box";
//            int boxId = mj_name2id(model, mjOBJ_BODY, boxstring.c_str());
//
//            m_state currState = modelTranslator->returnState(mdata);
//            currState(4) += 0.005;
//            modelTranslator->setState(mdata, currState);



//            m_quat currQuat = globalMujocoController->returnBodyQuat(model, mdata, boxId);
//            m_point axis = globalMujocoController->quat2Axis(currQuat);
//            axis(2) += 0.001;
//            m_quat newQuat = globalMujocoController->axis2Quat(axis);
//            globalMujocoController->setBodyQuat(model, mdata, boxId, newQuat);
            //currState(9) = 1;
            //modelTranslator->setState(mdata, currState);



            //modelTranslator->setState(mdata, currState);

//            const std::string link = "panda0_link7";
//            int linkId = mj_name2id(model, mjOBJ_BODY, link.c_str());
//            double jointVal = globalMujocoController->return_qPosVal(boxId, true, 6);
//            jointVal -= 0.001;
//            globalMujocoController->set_qPosVal(boxId, true, 6, jointVal);


//            m_quat boxQuat = globalMujocoController->returnBodyQuat(model, mdata, boxId);
//
//            cout << "box quat returned: " << boxQuat << endl;
//            m_point bodyAxis = globalMujocoController->quat2Axis(boxQuat);
//            cout << "box axis returned: " << bodyAxis << endl;
//            bodyAxis(1) += 0.005;
//            cout << "box axis given: " << bodyAxis << endl;
//            boxQuat = globalMujocoController->axis2Quat(bodyAxis);
//            cout << "box quat given: " << boxQuat << endl;
//            globalMujocoController->setBodyQuat(model, mdata, boxId, boxQuat);

//            m_quat boxQuat = globalMujocoController->returnBodyQuat(boxId);
//
//            m_point boxAxis = globalMujocoController->quat2Axis(boxQuat);
//
//            boxAxis(1) += 0.01;
//
//            m_quat newBoxQuat = globalMujocoController->axis2Quat(boxAxis);
//            globalMujocoController->setBodyQuat(boxId, newBoxQuat);
//

            for (int i = 0; i < NUM_MJSTEPS_PER_CONTROL; i++) {
                modelTranslator->stepModel(mdata, 1);
            }

            controlNum++;

            if (controlNum >= ILQR_HORIZON_LENGTH) {
                controlNum = 0;
                cpMjData(model, mdata, d_init_test);
                simstart = mdata->time;
            }

            if(controlNum == 984){
                int a = 1;
            }
        }

        // get framebuffer viewport
        mjrRect viewport = { 0, 0, 0, 0 };
        glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

        // update scene and render
        mjv_updateScene(model, mdata, &opt, NULL, &cam, mjCAT_ALL, &scn);
        mjr_render(viewport, &scn, &con);

        // swap OpenGL buffers (blocking call due to v-sync)
        glfwSwapBuffers(window);

        // process pending GUI events, call GLFW callbacks
        glfwPollEvents();
    }

}

void updateScreen(){
    mjrRect viewport = { 0, 0, 0, 0 };
    glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

    // update scene and render
    mjv_updateScene(model, mdata, &opt, NULL, &cam, mjCAT_ALL, &scn);
    mjr_render(viewport, &scn, &con);
    glfwSwapBuffers(window);
}

void initMujoco(){

    setupMujocoWorld();
    globalMujocoController = new MujocoController();
    updateScreen();

}