#include "AndroidNavigator.h"

#define PORT 8888

#include <stdio.h>
#include <kernel/PluginHelper.h>
#include <osg/Vec4>
#include <osg/Matrix>
#include <osgUtil/SceneView>
#include <config/ConfigManager.h>
#include <menu/MenuSystem.h>
#include <osg/Node>
#include <osgDB/ReadFile>
#include <string>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <unistd.h>
#include <kernel/ComController.h>

using namespace std;
using namespace osg;
using namespace cvr;

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

    bool status = false;
    _root = new osg::MatrixTransform();
    if(ComController::instance()->isMaster())
    {
        status = true;

    _andMenu = new SubMenu("AndroidNavigator", "AndroidNavigator");
    _andMenu->setCallback(this);
    
    _isOn = new MenuCheckbox("On", false);
    _isOn->setCallback(this);
    _andMenu->addItem(_isOn);
    MenuSystem::instance()->addMenuItem(_andMenu);

    // For Socket
     
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

    ComController::instance()->sendSlaves((char *)&status, sizeof(bool));
    }
    else
    {
        ComController::instance()->readMaster((char *)&status, sizeof(bool));
    }

    // Adds drawable for testing
       
    osg::Node* objNode = NULL;
    objNode = osgDB::readNodeFile("/home/bschlamp/Desktop/teddy.obj");    
  
    osg::MatrixTransform* trans1 = new osg::MatrixTransform();  
    osg::MatrixTransform* trans2 = new osg::MatrixTransform(); 
    osg::MatrixTransform* trans3 = new osg::MatrixTransform(); 
    osg::MatrixTransform* trans4 = new osg::MatrixTransform(); 

    _root->addChild(trans1);
    _root->addChild(trans2);
    _root->addChild(trans3);
    _root->addChild(trans4);

    trans1->addChild(objNode);
    trans2->addChild(objNode);
    trans3->addChild(objNode);
    trans4->addChild(objNode);

    osg::Matrix markTrans;
    markTrans.makeTranslate(osg::Vec3 (50,0,0));
    trans1->setMatrix(markTrans);
    markTrans.makeTranslate(osg::Vec3 (0,50,50));
    trans2->setMatrix(markTrans);
    markTrans.makeTranslate(osg::Vec3 (50,50,100));
    trans3->setMatrix(markTrans);
    markTrans.makeTranslate(osg::Vec3 (0,0,0));
    trans4->setMatrix(markTrans);
    SceneManager::instance()->getObjectsRoot()->addChild(_root);
    
    // Matrix data
    transMult = ConfigManager::getFloat("Plugin.AndroidNavigator.TransMult", 1.0);
    rotMult = ConfigManager::getFloat("Plugin.AndroidNavigator.RotMult", 1.0);
    
    //transcale = -0.05  * transMult;
    //rotscale = -0.000009 * rotMult;
    transcale = -.5 * transMult;
    rotscale = -.012 *rotMult; 
    _menuUp = false;

    // Default Movement to FLY
    _tagCommand = 4;

    std::cerr<<"AndroidNavigator done"<<endl;
      
    return status;
}

void AndroidNavigator::preFrame()
{
    Matrixd finalmat;
    //int num;

    if(ComController::instance()->isMaster())
    {  // num = 0;
    cout<<"Starting Master"<<endl;

    int RECVCONST = 48;
    double x, y, z;
    x = y = z = 0.0;
    double rx, ry, rz;
    rx = ry = rz = 0.0;
    
    // 0 = rotation, 1 = move, 2 = scale, 9 = error/initalize
    int tag = 9;
    int size = 0; 
    int start = 0;
    char* value;

    double angle [3] = {0.0, 0.0, 0.0};
    double coord [3] = {0.0, 0.0, 0.0};

    int bytes_read;
    char recv_data[1024];
    char send_data[1];
    struct sockaddr_in client_addr;

    fd_set fds;
    struct timeval timeout;
    int rc, result;

    timeout.tv_sec = 0;

    double usecDuration = PluginHelper::getLastFrameDuration();
    timeout.tv_usec =((int) usecDuration * 1000000);    
    // Converts from sec to microseconds
          
    FD_ZERO(&fds); 
    FD_SET(sock, &fds);
    //while(true)
    //{

        // Selects on a socket for the given time(timeout). Processes the data queue.
        //rc = select(sizeof(fds)*8, &fds, NULL, NULL, &timeout);
        //rc = select(sizeof(fds)*8, &fds, NULL, NULL, NULL);  //No timeout for testing
        //if(rc < 0){
        //    cerr<<"Select Error!"<<endl;
        //    break;
        //}
        //if(rc == 0){
        //    cout<<"Timeout"<<endl;
        //    break;
       // }

        bytes_read = recvfrom(sock, recv_data, 1024, 0, (struct sockaddr *)&client_addr, &addr_len);
 
        if(bytes_read <= 0){
            cerr<<"No data read."<<endl;
        }
    
        // Prepare Data for processing...
        recv_data[bytes_read]='\0';
        tag = recv_data[0] - RECVCONST;
 
        // Checks tag to see if it's a command
        if(tag > 3 && tag < 7){
            _tagCommand = (int) tag;
            send_data[0] = tag;
            sendto(sock, send_data, 1, 0, (struct sockaddr *)&client_addr, addr_len);
        }
  
        else if(tag == 7)
        {
            cout<<"Socket Connected"<<endl;
            send_data[0] = tag;
            sendto(sock, send_data, 1, 0, (struct sockaddr *)&client_addr, addr_len);
        }
 
        else
        {
            //Takes in tag for which kind of motion
            tag = recv_data[1] - RECVCONST;

            // Updates Rotation data 
            if (tag == 0){
                
                // First angle
                size = recv_data[6] - RECVCONST;
                start = 21;
                value = new char[size];
                for(int i=start; i<start+size; i++){
                    value[i-start] = recv_data[i];
                }
                angle[0] = atof(value);

                // Second angle
                start += size + 2;            // 2 accounts for space in string. Size is from previous angle size.
                size = recv_data[11] - RECVCONST;
                value = new char[size];
                for(int i=start; i<start+size; i++){
                    value[i-start] = recv_data[i];
                }
                angle[1] = atof(value);

                // Third angle
                start += size + 2;            // 2 accounts for space in string. Size is from previous angle size.
                size = recv_data[16] - RECVCONST;
                value = new char[size];
                for(int i=start; i<start+size; i++){
                    value[i-start] = recv_data[i];
                }
                angle[2] = atof(value);
            }


            // Updates touch movement data
            else if (tag == 1){
                    
                //First coord
                size = recv_data[6] - RECVCONST;
                start = 16;
                value = new char[size];
                for(int i=start; i<start+size; i++){
                    value[i-start] = recv_data[i];
                }
                coord[0] = atof(value);

                // Second coord
                start += size + 2;            // 2 accounts for space in string. Size is from previous coord.
                size = recv_data[11] - RECVCONST;
                value = new char[size];
                for(int i=start; i<start+size; i++){
                    value[i-start] = recv_data[i];
                }
                coord[1] = atof(value);
            }
     
            // Handles pinching movement (on touch screen) 
            else{
                size = recv_data[6] - RECVCONST;
                value = new char[size]; 
                start = 11;
                for(int i = start; i<start+size; i++){
                    value[i-start] = recv_data[i];
                }
                coord[2] = atof(value);
            }
        }
    
        switch(_tagCommand)
        {
            case 4:
                // For FLY movement
                rx -= angle[2];
                ry += angle[0];
                rz -= angle[1];

                x -= coord[0];
                z += coord[1];
                y += coord[2];
                break;
            case 5:
                // For DRIVE movement
                rz -= angle[1];
                y -= angle[2] * 250; 
                break;
            case 6:
                // For MOVE_WORLD movement
                rx -= angle[2];
                ry += angle[0];
                rz -= angle[1];
                break;
        }

        x *= transcale;
        y *= transcale;
        z *= transcale;
        rx *= rotscale;
        ry *= rotscale;
        rz *= rotscale;
  
        Matrix view = PluginHelper::getHeadMat();

        Vec3 campos = view.getTrans();
        Vec3 trans = Vec3(x, y, z);

        trans = (trans * view) - campos;

        Matrix tmat;
        tmat.makeTranslate(trans);
        Vec3 xa = Vec3(1.0, 0.0, 0.0);
        Vec3 ya = Vec3(0.0, 1.0, 0.0);
        Vec3 za = Vec3(0.0, 0.0, 1.0);

        xa = (xa * view) - campos;
        ya = (ya * view) - campos;
        za = (za * view) - campos;

        Matrix rot;
        rot.makeRotate(rx, xa, ry, ya, rz, za);

        Matrix ctrans, nctrans;
        ctrans.makeTranslate(campos);
        nctrans.makeTranslate(-campos);

        finalmat = PluginHelper::getObjectMatrix() * nctrans * rot * tmat * ctrans;
        ComController::instance()->sendSlaves((char *)finalmat.ptr(), sizeof(double[16]));
        PluginHelper::setObjectMatrix(finalmat);  
        
    //}

    //ComController::instance()->sendSlaves((char *)num, sizeof(int));
    }
    else
    {   cout<<"Starting Slave"<<endl; 
        //ComController::instance()->readMaster((char *)num, sizeof(int));
        ComController::instance()->readMaster((char *)finalmat.ptr(), sizeof(double[16]));
        PluginHelper::setObjectMatrix(finalmat);
        cout<<"End slave"<<endl;
    }
}

	
void AndroidNavigator::menuCallback(MenuItem* menuItem)
{
    if(menuItem == _isOn)
    {
        //TODO
    }
}    

bool AndroidNavigator::addMenu()
{
    MenuSystem::instance()->updateStart();
    _menuUp = true;
    return true;
}

bool AndroidNavigator::removeMenu()
{   
    SceneManager::instance()->getMenuRoot()->removeChild(_root);
    _menuUp = false;
    return true;
}
