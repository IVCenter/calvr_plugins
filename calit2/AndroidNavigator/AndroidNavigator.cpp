#include "AndroidNavigator.h"

#define PORT 8888

#include <stdio.h>
#include <cvrKernel/PluginHelper.h>
#include <osg/Vec4>
#include <osg/Matrix>
#include <osgUtil/SceneView>
#include <cvrConfig/ConfigManager.h>
#include <cvrMenu/MenuSystem.h>
#include <osg/Node>
#include <osgDB/ReadFile>
#include <string>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <unistd.h>
#include <cvrKernel/ComController.h>
#include <math.h>
#include <algorithm>
#include <cvrKernel/InteractionManager.h>
#include <osg/Shape>
#include <osg/ShapeDrawable>
#include <osg/Geode>
#include <osg/Quat>
#include "AndroidTransform.h"
#include <sstream>
#include <cstring>
#include <cvrKernel/SceneManager.h>
#include <osg/io_utils>

using namespace std;
using namespace osg;
using namespace cvr;

//class ArtifactVis;

bool useHeadTracking;
bool useDeviceOrientationTracking;
double orientation = 0.0;

CVRPLUGIN(AndroidNavigator)

AndroidNavigator::AndroidNavigator()
{
}

AndroidNavigator::~AndroidNavigator()
{
}

bool AndroidNavigator::init()
{
    std::cerr << "Android Navigator init\n"; 

    useHeadTracking = false;
    useDeviceOrientationTracking = false;
    bool status = false;
    _root = new osg::MatrixTransform();
    char* name = "None";
    node = new AndroidTransform(name);

    if(ComController::instance()->isMaster())
    {
        status = true;
        makeThread();
        ComController::instance()->sendSlaves((char *)&status, sizeof(bool));
    }

    else
    {
        ComController::instance()->readMaster((char *)&status, sizeof(bool));
    }


    // Adds drawables bears for testing AndroidTransform 
    osg::Node* objNode = NULL;
    objNode = osgDB::readNodeFile("teddy.obj");    
    name = "Bear1"; 
    AndroidTransform* trans1 = new AndroidTransform(name);
    name="Bear2";  
    AndroidTransform* trans2 = new AndroidTransform(name); 
    osg::MatrixTransform* trans3 = new osg::MatrixTransform(); 
    _root->addChild(trans1);
    _root->addChild(trans2);
    _root->addChild(trans3);
    trans1->addChild(objNode);
    trans2->addChild(objNode);
    trans3->addChild(objNode);
    trans1->setTrans(50.0,0.0,0.0);
    trans2->setTrans(100.0,0.0,100.0);
    
    SceneManager::instance()->getObjectsRoot()->addChild(_root);
  
    // Adds a menu option for AndroidNav (just so you know it's loaded). Doesn't actually do anything...  
    _andMenu = new SubMenu("AndroidNavigator", "AndroidNavigator");
    _andMenu->setCallback(this); 
    _isOn = new MenuCheckbox("On", false);
    _isOn->setCallback(this);
    _andMenu->addItem(_isOn);
    MenuSystem::instance()->addMenuItem(_andMenu);
   
    // Global set
    transMult = ConfigManager::getFloat("Plugin.AndroidNavigator.TransMult", 1.0);
    rotMult = ConfigManager::getFloat("Plugin.AndroidNavigator.RotMult", 1.0);
    transcale = -1 * transMult;
    rotscale = -.012 *rotMult; 
    newMode = false; 
    old_ry = 0.0; 
    node_name = NULL;
    velocity = 0;  // Default velocity --> not moving
    _tagCommand = -1; // Default tag command --> no movement mode

    std::cerr<<"AndroidNavigator done"<<endl;
    return status;
}

void AndroidNavigator::preFrame()
{
    Matrixd finalmat;

    double height = 0.0;
    double magnitude = 1.0;
    int position = 0;
    
    if(ComController::instance()->isMaster())
    {  

        int RECVCONST = 48;
        double x, y, z;
        x = y = z = 0.0;
        double rx, ry, rz;
        rx = ry = rz = 0.0;
        double sx, sy, sz; 
        sx = sy = sz = 1.0;
    
        int tag = 9;
	int type = 0;
        int mode = -1; // Navigation = 8, Node = 9, Command = 7

        double angle [3] = {0.0, 0.0, 0.0};
        double coord [3] = {0.0, 0.0, 0.0};

        char send_data[3];
        string str;
        const char* recv_data;
        char* split_str = NULL;
        
        _mutex.lock();
        while(!queue.empty()){
                         
            str = queue.front();
            queue.pop();
             
            split_str = strtok(const_cast<char*>(str.c_str()), " ");

            type = split_str[1] - RECVCONST;
            tag = split_str[2] - RECVCONST;
            mode = split_str[0] - RECVCONST;            

            send_data[0] = type + RECVCONST;
            send_data[1] = tag + RECVCONST;
            send_data[2] = '\0';
            int tagNum = atoi(send_data);

               /**
                * Changes movement type to the tag: 
                * 0 = Manual, 1 = Drive, 2 = Airplane, 3 = Old Fly
                */ 
            if(type == 2){
                _tagCommand = tag;
                sendto(sock, &tagNum, sizeof(int), 0, (struct sockaddr *)&client_addr, addr_len);
            }
              /** 
               * Process Commands from the android phone
               * 0 = Connect to server
               * 1 = Hide Node
               * 2 = Flip view
               * 3 = Show Node
               * 4 = Find AndroidTransform Nodes (to send to phone)
               * 5 = Gets back a selected AndroidTransform Node from phone, allows nodes to move.
               * 6 = Use Head Tracking
               * 7 = Use Device for Orientation Tracking
               * 8 = reset view
               */
            else if(type == 3)
            {
                if(tag == 0){
                    cout<<"Socket Connected"<<endl;
                    sendto(sock, &tagNum, sizeof(int), 0, (struct sockaddr *)&client_addr, addr_len);
                }
                else if(tag == 1){
                    sendto(sock, &tagNum, sizeof(int), 0, (struct sockaddr *)&client_addr, addr_len);
                    node->setNodeMask(0x0); 
                }
                else if(tag == 2){
                    sendto(sock, &tagNum, sizeof(int), 0, (struct sockaddr *)&client_addr, addr_len);
                    angle[1] = PI/rotscale;   
                }
                else if(tag == 3){
                    sendto(sock, &tagNum, sizeof(int), 0, (struct sockaddr *)&client_addr, addr_len);
                    node->setNodeMask(0xffffff);
                }
                else if(tag == 4){
                    sendto(sock, &tagNum, sizeof(int), 0, (struct sockaddr *)&client_addr, addr_len);
                    AndroidVisitor* visitor = new AndroidVisitor();
                    SceneManager::instance()->getObjectsRoot()->accept(*visitor);
                    nodeMap = visitor->getMap();
                    delete visitor;
                    
                    int num = (int) nodeMap.size();
                    sendto(sock, &num, 4, 0, (struct sockaddr *)&client_addr, addr_len);
                    
                    map<char*, AndroidTransform*>::iterator iter;
                    for(iter=nodeMap.begin(); iter != nodeMap.end(); iter++){
                      char* name = iter->first;
                      int size = (int) strlen(name);
                      sendto(sock, &size, sizeof(int), 0, (struct sockaddr *)&client_addr, addr_len); 
                      sendto(sock, name, strlen(name), 0, (struct sockaddr *)&client_addr, addr_len);
                    }
                }
                else if(tag == 5){
                    _tagCommand = 5; 
                    split_str = strtok(NULL, " ");
                    node_name = new char[strlen(split_str)];
                    strcpy(node_name, split_str);
                   
                    bool found = false;
 
                    map<char*, AndroidTransform*>::iterator iter;
                    for(iter=nodeMap.begin(); iter != nodeMap.end(); iter++){
                        if(strcmp(node_name, iter->first) == 0){
                            cout<<node_name<<" found"<<endl;
                            node = iter->second;
                            found = true;
                        }
                    }                    
                    if(!found) cout<<node_name<<" was not found..."<<endl;
                }
                else if(tag == 6){
                    sendto(sock, &tagNum, sizeof(int), 0, (struct sockaddr *)&client_addr, addr_len);
                    char *sPtr = strtok(NULL, " ");
                    useHeadTracking = (0 == strcmp(sPtr, "true"));
                }
                else if(tag == 7){
                    sendto(sock, &tagNum, sizeof(int), 0, (struct sockaddr *)&client_addr, addr_len);
                    char *sPtr = strtok(NULL, " ");
                    useDeviceOrientationTracking = (0 == strcmp(sPtr, "true"));
                }
                else if(tag == 8){
                	sendto(sock, &tagNum, sizeof(int), 0, (struct sockaddr *)&client_addr, addr_len);
                	osg::Matrix m;
                	SceneManager::instance()->setObjectMatrix(m);
                	SceneManager::instance()->setObjectScale(1.0);
                }
            }
            /** 
             * Updates Movement data
             * 0 = Rotation data
             * 1 = Translation data
             * 2 = Zcoord Translation
             * 3 = Velocity
             * 4 = Node Movement Data
             * 5 = Orientation angle
             */ 
            else if (type == 1)
            {
                // Updates Rotation data 
                if (tag == 0){
                    // First angle
                    split_str = strtok(NULL, " ");
                    angle[0] += atof(split_str);

                    // Second angle
                    split_str = strtok(NULL, " ");
                    angle[1] += atof(split_str);

                    // Third angle
                    split_str = strtok(NULL, " ");
                    angle[2] += atof(split_str);
                }

                // Updates touch movement data
                else if (tag == 1){
                    
                    // First coord 
                    split_str = strtok(NULL, " ");
                    coord[0] += atof(split_str);

                    // Second coord
                    split_str = strtok(NULL, " ");
                    coord[1] += atof(split_str);
                }
                
                // Node Adjustment Data
                else if (tag == 4){
    
                    // Height
                    split_str = strtok(NULL, " ");
                    height += atof(split_str);

                    // Magnitude
                    split_str = strtok(NULL, " ");
                    magnitude += atof(split_str);
 
                    // Axis
                    split_str = strtok(NULL, " ");
                    position = atoi(split_str);
                }

                //Orientation
                else if (tag == 5) {
					// Angle
					split_str = strtok(NULL, " ");
					orientation = atof(split_str);
				}

                // Handles pinching movement (on touch screen) and drive velocity
                else{
                    split_str = strtok(NULL, " ");
                    if (tag == 2){
                        coord[2] += atof(split_str);
                    }
                    else if (tag == 3){
                        velocity = atof(split_str);
                        if(atof(split_str) == 0){
                            velocity = 0;
                        }
                    }
                }
            }
        }
        _mutex.unlock();

        //angle[1] rot around x
        //angle[2] rot around y
        //angle[0] rot around z

        switch(_tagCommand)
        {
            case 0:
                // For Manual movement
            	rz += angle[0];
                rx += angle[1];
                ry += angle[2];
                if(angle[0] != 0)
                    old_ry = angle[0];
                x -= coord[0];
                z += coord[1];
                y += coord[2];
                break;
            case 1:
                // For DRIVE movement
                rz += angle[2];
                ry -= coord[0] * .5;  // Fixes orientation
                y += velocity;
                z += angle[1]; // For vertical movement
                break;
            case 2:
                // For Airplane movement
                rx += angle[1];
                ry -= angle[2];
                y += velocity;  // allow velocity to scale
                break;
            case 3:
                // Old fly mode
                rx += angle[1];
                ry -= coord[0] * .5; // Fixes orientation 
                rz += angle[2];
                y += velocity;  // allow velocity to scale
                break;
            case 5:
                if(node_name != NULL){
                   adjustNode(height, magnitude, position);
                }
                else cout<<"No Node Selected"<<endl;
                break;
        }

        // Scales data by set amount
        x *= transcale;
        y *= transcale;
        z *= transcale;
        rx *= rotscale;
        rz *= rotscale;
        ry *= rotscale;
  
        /*
         * If newMode (which occurs when Drive and New Fly starts),
         *  this takes in a new headMat camera pos.
         * If not, this takes in the old position, with the exception
         *  of the z axis, which corresponds to moving your head up and down
         *  to eliminate conflict between phone and head tracker movement.
         */ 
        Matrix world2head = PluginHelper::getHeadMat();
        Matrix view, mtrans;

        if(useDeviceOrientationTracking){

        	view.makeRotate(orientation,0,0,1);
            mtrans.makeTranslate(world2head.getTrans());
        	view = view * mtrans;
        } else
        	view = world2head;

        Vec3 campos = view.getTrans();

        // Gets translation
        Vec3 trans = Vec3(x, y, z);

        //Test
        if(useHeadTracking){
        	trans = (trans * view) - campos;
        }else if(useDeviceOrientationTracking){
        	trans = (trans * view) - campos;
        }

        Matrix tmat;
        tmat.makeTranslate(trans);

        Vec3 xa = Vec3(1.0, 0.0, 0.0);
        Vec3 ya = Vec3(0.0, 1.0, 0.0);
        Vec3 za = Vec3(0.0, 0.0, 1.0);

        //Test
        if(useHeadTracking){
        	xa = (xa * view) - campos;
        	ya = (ya * view) - campos;
        	za = (za * view) - campos;
        }else if(useDeviceOrientationTracking){
        	xa = (xa * view) - campos;
        	ya = (ya * view) - campos;
        	za = (za * view) - campos;
        }

        // Gets rotation
        Matrix rot;
        rot.makeRotate(rx, xa, ry, ya, rz, za);

        Matrix ctrans, nctrans;
        ctrans.makeTranslate(campos);
        nctrans.makeTranslate(-campos);

        // Calculates new objectMatrix (will send to Slaves).
        //finalmat = PluginHelper::getObjectMatrix() * nctrans * rot * tmat * ctrans;
        finalmat = PluginHelper::getObjectMatrix() * nctrans * rot * tmat * ctrans;
        ComController::instance()->sendSlaves((char *)finalmat.ptr(), sizeof(double[16]));
        PluginHelper::setObjectMatrix(finalmat);
    
    }
    else
    {
        ComController::instance()->readMaster((char *)finalmat.ptr(), sizeof(double[16]));
        PluginHelper::setObjectMatrix(finalmat);
    }

    
}

	
void AndroidNavigator::menuCallback(MenuItem* menuItem)
{
    if(menuItem == _isOn)
    {
        //If I ever need a menu item?
    }
}    

/*
 * Makes a new thread to take in data from phone
 * Port = 8888.
 */
void AndroidNavigator::makeThread(){
    cout<<"Starting socket..."<<endl;    
    if ((sock = socket(AF_INET, SOCK_DGRAM, 0)) == -1){
        cerr<<"Socket Error!"<<endl;
        exit(1); 
    }

    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(PORT);
    server_addr.sin_addr.s_addr = INADDR_ANY;
    bzero(&(server_addr.sin_zero), 8);

    if (bind(sock, (struct sockaddr *)&server_addr, sizeof(struct sockaddr)) == -1){
        cerr<<"Bind Error!"<<endl;
        exit(1);
    }

    addr_len = sizeof(struct sockaddr);
    cout<<"Server waiting for client on port: "<<PORT<<endl;
    _mkill = false;
    start();
}

/*
 * Runs thread that takes data. Queues it to be processed
 *   in preFrame.
 */
void AndroidNavigator::run()
{
    int bytes_read;
    char recv_data[1024];

    fd_set readfds;
    int rs;
    FD_ZERO(&readfds);
    FD_SET(sock, &readfds);
    string str;
    while(!_mkill)
    {
        rs = select(sock + 1, &readfds, 0, 0, 0);
        _mutex.lock();
        if(rs > 0){
            bytes_read = recvfrom(sock, recv_data, 1024 , 0, (struct sockaddr *)&client_addr, &addr_len);
            if(bytes_read <= 0){
                cerr<<"No data read."<<endl;
            }
            // Prepares data for processing...
            recv_data[bytes_read]='\0';
            str = recv_data;
            queue.push(str);
        }
        _mutex.unlock();
    }
}

/*
 * Adjusts the selected AndroidTransform node.
 *   (One has to be selected for this function to even be called).
 * Gets data from phone.
 */
void AndroidNavigator::adjustNode(double height, double magnitude, int position){

    double adjust = height * magnitude;
    double value = adjust * PluginHelper::getObjectScale();
    double node_rx, node_ry, node_rz;
    node_rx = node_ry = node_rz = 0.0;

    Vec3 newVec;
    switch(position){
       case 0:  // X Trans
           newVec = Vec3(value, 0, 0);
           break;
       case 1:  // Y Trans
           newVec = Vec3(0, value, 0);
           break;
       case 2:  // Z Trans
           newVec = Vec3(0, 0, value);
           break;
       case 3:  // X Rot
           node_rx += value * rotscale;
           break;
       case 4:  // Y Rot
           node_ry += value * rotscale;
           break; 
       case 5:  // Z Rot
           node_rz += value * rotscale;
           break;
    }    

    Vec3 pos = node->getTrans();

    if(position >= 0 && position <= 2){
        pos = pos + newVec;
    }
        
    Matrix view = PluginHelper::getHeadMat();
    Vec3 campos = view.getTrans();

    Vec3 xa = Vec3(1.0, 0.0, 0.0);
    Vec3 ya = Vec3(0.0, 1.0, 0.0);
    Vec3 za = Vec3(0.0, 0.0, 1.0);
    xa = (xa * view) - campos;
    ya = (ya * view) - campos;
    za = (za * view) - campos;
      
    node->setRotation(node_rx , xa, node_ry, ya, node_rz , za);
    node->setTrans(pos);
}
