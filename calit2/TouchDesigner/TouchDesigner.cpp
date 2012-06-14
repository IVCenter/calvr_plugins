#include "TouchDesigner.h"

#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/SceneManager.h>
#include <cvrMenu/MenuSystem.h>
#include <cvrKernel/PluginHelper.h>
#include <cvrKernel/ComController.h>
#include <cvrUtil/CVRSocket.h>
#include <cvrKernel/SceneObject.h>

#include <fcntl.h>
#include <iostream>
#include <cstring>

#include <osg/Matrix>
#include <osg/CullFace>
#include <osgDB/ReadFile>


using namespace osg;
using namespace std;
using namespace cvr;

#define PORT 19997
#define PACKETLEN 500  // # bytes in a packet



const float        X_SCALE	 = 1700;
const float        Y_SCALE	 = 600;
const float        Z_SCALE	 = 600;
const float        RAD_SCALE	 = 100;
float     _previous;
Vec3 _pos;
int x = -600;
int z = -600;
int TESS = 10;

static char recvData[PACKETLEN];
bool received = false;
Geode* geode;
bool startNext=false;
bool doneBuilding=false;

vector<string> swapData;
int numberInSwap=0;
int renderSize = 0;

int counter = 0;

CVRPLUGIN(TouchDesigner) 
	TouchDesigner::TouchDesigner()
{
	cerr << "1" << endl;
}

bool TouchDesigner::init()
{
	cerr << "TouchDesigner init\n";

	_menu = new SubMenu("TouchDesigner", "TouchDesigner");

	_menu->setCallback(this);

	_receiveButton = new MenuButton("Receive");
	_receiveButton->setCallback(this);
	_menu->addItem(_receiveButton);

	_port = ConfigManager::getEntry("Plugin.TouchDesigner.Port");

	MenuSystem::instance()->addMenuItem(_menu);

	if(ComController::instance()->isMaster()) // head node
	{
	  string name = "test run";
	  st = new SocketThread(name);
        }
             
	return true;
}

TouchDesigner::~TouchDesigner()
{
}

void TouchDesigner::preFrame()
{
	string data;

	if(ComController::instance()->isMaster()) // head node
	{
		data = st->getSerializedScene();
		ComController::instance()->sendSlaves(&data[0], PACKETLEN);
		ComController::instance()->sendSlaves(&received, sizeof(bool));
	}
	else  // rendering nodes
	{
		ComController::instance()->readMaster(recvData, PACKETLEN);
		data = recvData;
		ComController::instance()->readMaster(&received, sizeof(bool));
		
	}
			
	std::stringstream ss(data);
	osgDB::ReaderWriter * readerwriter =  Registry::instance()->getReaderWriterForExtension("ive");
	ReaderWriter::ReadResult result = readerwriter->readNode(ss);	
		
		
//	cerr <<" valid node "  << result.validNode() << endl;
	if (result.validNode())
	{
		//cerr << "valid node" << endl;
		Node * node = result.getNode();
		
		osg::Group * objectRoot = SceneManager::instance()->getObjectsRoot();
		
		// remove all children and add new node
		objectRoot->removeChildren(0, objectRoot->getNumChildren());
		objectRoot->addChild(node);
		
		//cerr << "data\t" << endl;
	}
	
	


}

void TouchDesigner::menuCallback(MenuItem* menuItem)
{
	if(menuItem == _receiveButton)
	{
		receiveGeometry();
		return;
	}
}

double TouchDesigner::random(){
	return rand() / double(RAND_MAX);
}

void TouchDesigner::receiveGeometry()
{
	int BUFSIZE = 500;
	cvr::CVRSocket server=cvr::CVRSocket(LISTEN, "128.54.37.189", 6662 ,AF_INET,SOCK_DGRAM);
	cerr << "server created" << endl;
	server.setReuseAddress(true);
	server.bind();
	cerr << "bound" << endl;	
	
	Vec4d nil(0.0,0.0,0.0,0.0);
	Vec4d krayBlue(0.0,0.5,1.0,1.0);

	string dummyT = "trianglec cx=0 cy=0 cz=0 length=300 c1r=0 c1g=0.5 c1b=1.0 c1a=0.8 c2r=0 c2g=0 c2b=0 c2a=1 comment=sh\0";
	string dummyTP = "trianglep p1x=0 p1y=0 p1z=0  p2x=100 p2y=0 p2z=0 p3x=50 p3y=0 p3z=50 c1r=0 c1g=0.5 c1b=1.0 c1a=0.8 comment=sh\0";
	string dummyUTP = "0 p1x=-100 p1y=0 p1z=0  p2x=0 p2y=0 p2z=0 p3x=150 p3y=0 p3z=150 c1r=0 c1g=0.5 c1b=0.7 c1a=0.6 c2r=0 c2g=0 c2b=0 c2a=1 comment=sh\0";
	string dummyC = "circle cx=-300 cy=0 cz=-300 radius=100 c1r=0 c1g=0.5 c1b=1.0 c1a=0.8 c2r=0 c2g=0 c2b=0 c2a=1 comment=sh\0";
	string dummyUC = "updatec id=2 cx=0 cy=0 cz=-300 radius=100 c1r=0.2 c1g=0.7 c1b=0.6 c1a=0.8 c2r=0 c2g=0 c2b=0 c2a=1 comment=sh\0";
	string dummyUR = "updaterc id=1 cx=0 cy=0 cz=300 height=300 width=300 c1r=0.3 c1g=0.7 c1b=0.4 c1a=0.8 c2r=0 c2g=0 c2b=0 c2a=1 comment=sh\0";
	string dummyUT = "updatetc id=0 cx=300 cy=0 cz=0 length=200 c1r=0.3 c1g=0.5 c1b=0.7 c1a=0.8 c2r=0 c2g=0 c2b=0 c2a=1 comment=sh\0";
	string dummyR = "rectc cx=300 cy=0 cz=300 height=300 width=300 c1r=0 c1g=0.5 c1b=1.0 c1a=0.8 c2r=0 c2g=0 c2b=0 c2a=1 comment=sh\0";
	
	string dummyRP = "rectp p1x=0 p1y=0 p1z=0  p2x=100 p2y=0 p2z=0 p3x=100 p3y=0 p3z=100 p4x=0 p4y=0 p4z=100 c1r=0 c1g=0.5 c1b=1.0 c1a=0.8 comment=sh\0";
	//string dummyURP = "updaterp p1x=-100 p1y=0 p1z=0  p2x=0 p2y=0 p2z=0 p3x=0 p3y=0 p3z=100 p4x=-100 p4y=0 p4z=100 c1r=0 c1g=0.5 c1b=1.0 c1a=0.8 c2r=0 c2g=0 c2b=0 c2a=1 comment=sh\0";
	
	string dummyURP = "1 p1x=-100 p1y=0 p1z=0  p2x=0 p2y=0 p2z=0 p3x=0 p3y=0 p3z=100 p4x=-100 p4y=0 p4z=100 c1r=0 c1g=0.5 c1b=1.0 c1a=0.8 c2r=0 c2g=0 c2b=0 c2a=1 comment=sh\0";


	SceneManager::instance()->getObjectsRoot()->addChild(st->getTestNode());
}


int TouchDesigner::random(int min, int max)
{
	return (max-min)*random() + min;
}

