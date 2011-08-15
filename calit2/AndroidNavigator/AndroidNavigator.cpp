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
#include <math.h>
//#include "/home/bschlamp/CalVR/plugins/calit2/ArtifactVis/ArtifactVis.h"
#include <algorithm>
#include <kernel/InteractionManager.h>

using namespace std;
using namespace osg;
using namespace cvr;

//class ArtifactVis;

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

        makeThread();

        ComController::instance()->sendSlaves((char *)&status, sizeof(bool));
    }

    else
    {
        ComController::instance()->readMaster((char *)&status, sizeof(bool));
    }

   /* 
    // Adds drawables bears for testing       
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
    */
    
    tracker = new TrackingInteractionEvent;
    Vec3 location = Vec3(0.0, 1.0, 0.0) * PluginHelper::getObjectMatrix();
    tracker->xyz[0] = location[0];
    tracker->xyz[1] = location[1];
    tracker->xyz[2] = location[2];

    tracker->hand = 0;
    tracker->button = 1;
    tracker->rot[0] = 0.0;
    tracker->rot[1] = 0.0;
    tracker->rot[2] = 0.0;
    tracker->rot[3] = 0.0;

    // Matrix data
    transMult = ConfigManager::getFloat("Plugin.AndroidNavigator.TransMult", 1.0);
    rotMult = ConfigManager::getFloat("Plugin.AndroidNavigator.RotMult", 1.0);
    
    transcale = -1 * transMult;
    rotscale = -.012 *rotMult; 
    _menuUp = false;
    
    // Default velocity for drive
    velocity = 15;

    std::cerr<<"AndroidNavigator done"<<endl;
      
    return status;
}

void AndroidNavigator::preFrame()
{
    Matrixd finalmat;

    if(ComController::instance()->isMaster())
    {  

        int RECVCONST = 48;
        double x, y, z;
        x = y = z = 0.0;
        double rx, ry, rz;
        rx = ry = rz = 0.0;
    
        // 0 = rotation, 1 = move, 2 = scale, 9 = error/initalize
        int tag = 9;

	int type = 0;
        int size = 0; 
        int start = 0;
        char* value;
        int VELO_CONST = 10;

        double angle [3] = {0.0, 0.0, 0.0};
        double coord [3] = {0.0, 0.0, 0.0};

        char send_data[2];
        string str;
        const char* recv_data;
        
        _mutex.lock();
        while(!queue.empty()){
                      
            str = queue.front();
            //cout<<queue.front()<<" received: "<<queue.size()<<endl;
            queue.pop();

            recv_data = new char[str.size()];
            recv_data = str.c_str();
            type = recv_data[1] - RECVCONST;
            tag = recv_data[2] - RECVCONST;
            
            send_data[0] = type;
            send_data[1] = tag;
            
            if(type == 2){
                _tagCommand = tag;
                sendto(sock, send_data, 2, 0, (struct sockaddr *)&client_addr, addr_len);
            }
  
            else if(type == 3)
            {
                if(tag == 0){
                    cout<<"Socket Connected"<<endl;
                    sendto(sock, send_data, 2, 0, (struct sockaddr *)&client_addr, addr_len);
                }
                else if(tag == 1){
                    if(_menuUp){ removeMenu();}
                    else{ addMenu();}
                    InteractionManager::instance()->addEvent(tracker);
                    sendto(sock, send_data, 2, 0, (struct sockaddr *)&client_addr, addr_len);
                }  
                else if(tag == 2){
		    sendto(sock, send_data, 2, 0, (struct sockaddr *)&client_addr, addr_len);
                    objectSelection();                
                }
            } 
            else if (type == 1)
            {
                // Updates Rotation data 
                if (tag == 0){
                
                    // First angle
                    size = recv_data[7] - RECVCONST;
                    start = 22;
                    value = new char[size];
                    for(int i=start; i<start+size; i++){
                        value[i-start] = recv_data[i];
                    }
                    angle[0] = atof(value);

                    // Second angle
                    start += size + 2;        // 2 accounts for space in string. Size is from previous angle size.
                    size = recv_data[12] - RECVCONST;
                    value = new char[size];
                    for(int i=start; i<start+size; i++){
                        value[i-start] = recv_data[i];
                    }
                    angle[1] = atof(value);

                    // Third angle
                    start += size + 2;        // 2 accounts for space in string. Size is from previous angle size.
                    size = recv_data[17] - RECVCONST;
                    value = new char[size];
                    for(int i=start; i<start+size; i++){
                        value[i-start] = recv_data[i];
                    }
                    angle[2] = atof(value);
                }


                // Updates touch movement data
                else if (tag == 1){
                    
                    //First coord
                    size = recv_data[7] - RECVCONST;
                    start = 17;
                    value = new char[size];
                    for(int i=start; i<start+size; i++){
                        value[i-start] = recv_data[i];
                    }
                    coord[0] = atof(value);

                    // Second coord
                    start += size + 2;        // 2 accounts for space in string. Size is from previous coord.
                    size = recv_data[12] - RECVCONST;
                    value = new char[size];
                    for(int i=start; i<start+size; i++){
                        value[i-start] = recv_data[i];
                    }
                    coord[1] = atof(value);
                }
     
                // Handles pinching movement (on touch screen) and drive velocity
                else{
                    size = recv_data[7] - RECVCONST;
                    value = new char[size]; 
                    start = 12;
                    for(int i = start; i<start+size; i++){
                        value[i-start] = recv_data[i];
                    }
                
                    if (tag == 2){
                        coord[2] = atof(value);
                    }
                    else if (tag == 3){
                        velocity = atof(value);
                    }
                }
            }
        }
        _mutex.unlock();

        if(_menuUp){
            // TODO menu interaction    
        }
        else{
            switch(_tagCommand)
            {
                case 0:
                    // For FLY movement
                    rx += angle[2];
                    ry += angle[0];
                    rz += angle[1];

                    x -= coord[0];
                    z += coord[1];
                    y += coord[2];
                    break;
                case 1:
                    // For DRIVE movement
                    rz += angle[1] * VELO_CONST/2;
                    y += angle[2] * velocity * VELO_CONST;
                    break;
                case 2:
                    // For MOVE_WORLD movement
                    rx += angle[2];
                    ry += angle[0];
                    rz += angle[1];
                    break;
            }
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
        //TODO
    }
}    

bool AndroidNavigator::addMenu()
{
    tracker = new TrackingInteractionEvent;
    Vec3 location = Vec3(0.0, 1.0, 0.0) * PluginHelper::getObjectMatrix();
    tracker->xyz[0] = location[0];
    tracker->xyz[1] = location[1];
    tracker->xyz[2] = location[2];

    tracker->hand = 0;
    tracker->button = 1;
    tracker->rot[0] = 0.0;
    tracker->rot[1] = 0.0;
    tracker->rot[2] = 0.0;
    tracker->rot[3] = 0.0;
    tracker->type = BUTTON_DOUBLE_CLICK;

    _menuUp = true;
    return true;
}

bool AndroidNavigator::removeMenu()
{   
    tracker = new TrackingInteractionEvent;
    Vec3 location = Vec3(0.0, 1.0, 0.0) * PluginHelper::getObjectMatrix();
    tracker->xyz[0] = location[0];
    tracker->xyz[1] = location[1];
    tracker->xyz[2] = location[2];

    tracker->hand = 0;
    tracker->button = 1;
    tracker->rot[0] = 0.0;
    tracker->rot[1] = 0.0;
    tracker->rot[2] = 0.0;
    tracker->rot[3] = 0.0;
    tracker->type = BUTTON_DOWN;

    _menuUp = false;
    return true;
}

/*
// For use with ArtifactVis.
// Iterates through items and determines which are in
// camera view for object selection.

void AndroidNavigator::objectSelection(){

    double objAngle;
    double minAngle = .05;
    int bytes_read;
    char recv_data[1024];
    char send_data[2];
    vector<Vec3> inRange;
    int rangeObj = 0;
    vector<osg::Vec3> position;
    
    Matrix objMatrix = PluginHelper::getObjectMatrix();
    osg::Vec3 camera = Vec3(0.0, 1.0, 0.0);

    Matrix viewOffsetM;
    osg::Vec3 viewOffset = Vec3f(ConfigManager::getFloat("x", "ViewerPosition", 0.0f),
		ConfigManager::getFloat("y", "ViewerPosition", 0.0f),
                ConfigManager::getFloat("z", "ViewerPosition", 0.0f)) * -1;
    viewOffsetM.makeTranslate(viewOffset);
    objMatrix = objMatrix * viewOffsetM;
   
    ArtifactVis* art = ArtifactVis::getInstance();
    if(art != NULL){ 
        cout<<"Getting artifacts"<<endl;
    	position = art->getArtifactsPos();
    }
    else{
        cout<<"ArtifactVis not initialized"<<endl;
        send_data[0] = 9;
        sendto(sock, send_data, 2, 0, (struct sockaddr *)&client_addr, addr_len);
    }

    if(position.empty()){
        cerr<<"No artifact positions received"<<endl;
        send_data[0] = 8;
        sendto(sock, send_data, 2, 0, (struct sockaddr *)&client_addr, addr_len);
        return;
    }

    for(int obj = 0; obj < position.size(); obj++){
        osg::Vec3 pos = position.at(obj); 
        osg::Vec3 alpha = (pos * objMatrix);
        objAngle = acos(alpha * camera / (alpha.length() * camera.length()));
        if (objAngle < minAngle){
            inRange.push_back(alpha);   
        }
    }

    if(inRange.empty()){
        cerr<<"No suitable objects"<<endl;
        send_data[0] = 7;
        sendto(sock, send_data, 2, 0, (struct sockaddr *)&client_addr, addr_len);
        return;
    }

    cout<<inRange.size()<<" objects found!"<<endl;
    send_data[0] = 6;
    sendto(sock, send_data, 2, 0, (struct sockaddr *)&client_addr, addr_len);

    std::sort (inRange.begin(), inRange.end(), compare());
    
    while(rangeObj < inRange.size() && rangeObj >= 0){
        osg::Vec3 pos = inRange.at(rangeObj) * PluginHelper::getObjectToWorldTransform();
        cout<<"Pos "<<rangeObj + 1<<": <"<<pos[0]<<", "<<pos[1]<<", "<<pos[2]<<">"<<endl;
        
        bytes_read = recvfrom(sock, recv_data, 1024, 0, (struct sockaddr *)&client_addr, &addr_len); 

        if(bytes_read <= 0){
            cerr<<"No data read."<<endl;
        }   
        recv_data[bytes_read]='\0';
        if(recv_data[0] == 0){ 
            cout<<"Object Found"<<endl;
            return;
        }
        else if(recv_data[0] == 1){
            cout<<"Next object..."<<endl;
            if(rangeObj + 1 >= inRange.size()){
                cerr<<"Cannot find Next: Out of range"<<endl;
            }
            else rangeObj++;
        }
        else if(recv_data[0] == 2){
            cout<<"Previous object..."<<endl;
            if(rangeObj - 1 < 0){
                cerr<<"Cannot find Previous: out of range"<<endl;
            }            
            else rangeObj--;
        }        
        else if(recv_data[0] == 3){
            return;
        }
    }
    
}
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

void AndroidNavigator::run()
{
    int bytes_read;
    char recv_data[1024];

    fd_set readfds;
    struct timeval timeout;
    int rs;

    FD_ZERO(&readfds);
    FD_SET(sock, &readfds);
    string str;

    timeout.tv_sec = 0; 
        // Gets last framerate to take in data more quickly
    timeout.tv_usec = (int) (PluginHelper::getLastFrameDuration() * 1000000);

    while(!_mkill)
    {
        
        rs = select(sock + 1, &readfds, 0, 0, 0);
        _mutex.lock();
        if(rs > 0){
            bytes_read = recvfrom(sock, recv_data, 1024, 0, (struct sockaddr *)&client_addr, &addr_len);

            if(bytes_read <= 0){
                cerr<<"No data read."<<endl;
            }
           
              // Prepares data for processing...
            recv_data[bytes_read]='\0';
            str = recv_data;
            queue.push(str);
            //cout<<str<<" added: "<<queue.size()<<endl;
        }
        _mutex.unlock();
    }
}
