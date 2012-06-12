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

#define PACKETLEN 500
bool received = false;
static string recvData = "";
int dsize = 0;


using namespace osg;
using namespace std;
using namespace cvr;


CVRPLUGIN( TouchDesigner)
	TouchDesigner::TouchDesigner() {
		cerr << "1" << endl;
}

bool TouchDesigner::init() {

	cerr << "TouchDesigner init\n";

	_menu = new SubMenu("TouchDesigner", "TouchDesigner");

	_menu->setCallback(this);

	_receiveButton = new MenuButton("Receive");
	_receiveButton->setCallback(this);
	_menu->addItem(_receiveButton);

	_port = ConfigManager::getEntry("Plugin.TouchDesigner.Port");

	prevNode = NULL;

	MenuSystem::instance()->addMenuItem(_menu);

	if (ComController::instance()->isMaster()) // head node
	{
		string name = "test run";
		st = new SocketThread(name);
	}

	return true;
}

TouchDesigner::~TouchDesigner() {

}

void TouchDesigner::preFrame() {
	string data = "";

	//	cerr << "painting\t" << endl;
	if (ComController::instance()->isMaster()) // head node
	{
		data = st->getSerializedScene();
		dsize = data.size();

		//		cerr << "in master doing nothing" << endl;
		ComController::instance()->sendSlaves(&dsize, sizeof(int));
		ComController::instance()->sendSlaves(&data[0], dsize);
		ComController::instance()->sendSlaves(&received, sizeof(bool));
	} else // rendering nodes
	{
		//		cerr<<"slave node"<<endl;	
		ComController::instance()->readMaster(&dsize, sizeof(int));
		recvData.resize(dsize);
		ComController::instance()->readMaster(&recvData[0], dsize);
		data = recvData;
		ComController::instance()->readMaster(&received, sizeof(bool));
	}



	//	cerr << "data size\t" << dsize << endl;
	std::stringstream ss(data);
	osgDB::ReaderWriter * readerwriter =
		Registry::instance()->getReaderWriterForExtension("ive");
	ReaderWriter::ReadResult result = readerwriter->readNode(ss);

	if (result.validNode()) {
		//printf("Node valid\n");
		Node * node = result.getNode();
		/*			
		osg::Geode * geode = dynamic_cast<osg::Geode* > (node);

		if( geode )
		{
		for(int i = 0; i < geode->getNumDrawables(); i++)
		{
		osg::Geometry* geom = dynamic_cast<osg::Geometry*> (geode->getDrawable(i));

		if( geom )
		{
		osg::Vec3Array* array = dynamic_cast<osg::Vec3Array *> (geom->getVertexArray());

		for(int j =0; j < array->size(); j++)
		{
		cerr << array->at(0)[0] << " " << array->at(0)[1] << " " << array->at(0)[2] << endl;
		}
		}
		}	
		}
		*/
		osg::Group * objectRoot = SceneManager::instance()->getObjectsRoot();

		// remove all children and add new node
		objectRoot->addChild(node);

		if (prevNode)
			objectRoot->removeChild(prevNode);

		prevNode = node;
		
				
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

}

int TouchDesigner::random(int min, int max) {
	return (max - min) * random() + min;
}


