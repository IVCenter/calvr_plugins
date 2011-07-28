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

    _root = new osg::MatrixTransform();
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

    // Adds drawable for testing
       
    osg::Node* objNode = NULL;
    objNode = osgDB::readNodeFile("/home/bschlamp/Desktop/teddy.obj");    

    _root->addChild(objNode);
    
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

    return true;
}

void AndroidNavigator::preFrame()
{
    double x, y, z;
    x = y = z = 0.0;
    double rx, ry, rz;
    rx = ry = rz = 0.0;
    
    // 0 = rotation, 1 = move, 2 = scale, 9 = error/initalize
    int tag = 9;
    char num;
 
    Matrix finalmat;
            
    double angle [3] = {0.0, 0.0, 0.0};
    double coord [3] = {0.0, 0.0, 0.0};

    int bytes_read;
    char recv_data[1024];
    struct sockaddr_in client_addr;
           
    // Gets tag
    bytes_read = recvfrom(sock, recv_data, 1024, 0, (struct sockaddr *)&client_addr, &addr_len);
    recv_data[bytes_read] = '\0';
    if(bytes_read <= 0){
        cerr<<"No Tag Received"<<endl;
        tag = 9;
    }
    else{
        tag = atoi(recv_data);
    }

    if(tag > 3 && tag < 7){
       _tagCommand = tag;
    }
    
    // Takes in angle rotation data
    if (tag == 0){

       for (int i = 0; i < 3; i++){
           bytes_read = recvfrom(sock, recv_data, 1024, 0, (struct sockaddr *)&client_addr, &addr_len);
           if(bytes_read <= 0){
               cerr<<"No data received for angle "<<i+1<<endl;
               angle[i] = 0.0;
           }
           else{
               num = recv_data[bytes_read - 1];
               recv_data[bytes_read - 1] = '\0';
               angle[num] = atof(recv_data);
           }
       }
    }


    //Takes in touch movement data
    else if (tag == 1){
        for (int i = 0; i < 2; i++){    
           bytes_read = recvfrom(sock, recv_data, 1024, 0, (struct sockaddr *)&client_addr, &addr_len);
                
           if (bytes_read <= 0){
               cerr<<"No data received for coord "<<i+1<<endl;
               coord[i] = 0.0;
           }
           else{
               num = recv_data[bytes_read - 1];
               recv_data[bytes_read - 1] = '\0';
               coord[num] = atof(recv_data);
           }
        }
    }

    // Takes in last coord for touch movement data
    else if (tag == 2){
    
        bytes_read = recvfrom(sock, recv_data, 1024, 0, (struct sockaddr *)&client_addr, &addr_len);
        if (bytes_read <= 0){
            cerr<<"No data received for scale."<<endl;
            coord[2] = 0.0;
        }
        else{
            recv_data[bytes_read - 1] = '\0';
            coord[2] = atof(recv_data);
        }
    }

    switch(_tagCommand){
        case 4:
            // For FLY movement
            rx -= angle[2];
            ry += angle[0];
            rz -= angle[1];

            x += coord[0];
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
  
    //tag = 9;
 
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
    
    PluginHelper::setObjectMatrix(finalmat);
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
