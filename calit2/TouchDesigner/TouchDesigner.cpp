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

#include "util/TrackerTree.h"


#define PACKETLEN 500
bool received = false;
static string recvData = "";
int dsize = 0;


using namespace osg;
using namespace std;
using namespace cvr;

<<<<<<< HEAD

=======
>>>>>>> e7aefd8ba4f234f3d166f0a934e5cf6ea045a343
CVRPLUGIN( TouchDesigner)
TouchDesigner::TouchDesigner() {
	cerr << "1" << endl;
}

bool TouchDesigner::init() {
<<<<<<< HEAD

=======
>>>>>>> e7aefd8ba4f234f3d166f0a934e5cf6ea045a343
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
<<<<<<< HEAD

=======
>>>>>>> e7aefd8ba4f234f3d166f0a934e5cf6ea045a343
}

void TouchDesigner::preFrame() {
	string data = "";
<<<<<<< HEAD
	
//	cerr << "painting\t" << endl;
=======

>>>>>>> e7aefd8ba4f234f3d166f0a934e5cf6ea045a343
	if (ComController::instance()->isMaster()) // head node
	{
		data = st->getSerializedScene();
		dsize = data.size();
<<<<<<< HEAD
		
//		cerr << "in master doing nothing" << endl;
                ComController::instance()->sendSlaves(&dsize, sizeof(int));
=======
>>>>>>> e7aefd8ba4f234f3d166f0a934e5cf6ea045a343
		ComController::instance()->sendSlaves(&data[0], dsize);
		ComController::instance()->sendSlaves(&received, sizeof(bool));
	} else // rendering nodes
	{
<<<<<<< HEAD
//		cerr<<"slave node"<<endl;	
                ComController::instance()->readMaster(&dsize, sizeof(int));
                recvData.resize(dsize);
=======
>>>>>>> e7aefd8ba4f234f3d166f0a934e5cf6ea045a343
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
<<<<<<< HEAD
		  objectRoot->removeChild(prevNode);
=======
			objectRoot->removeChild(prevNode);
>>>>>>> e7aefd8ba4f234f3d166f0a934e5cf6ea045a343

		//cerr << objectRoot->getNumChildren() << " nodes are in root" << endl;
		//cerr << "previous node\t" << prevNode << endl;
		//cerr << "current node\t" << node << endl;
		  prevNode = node;
	}

	//cerr << "" << endl;

}

<<<<<<< HEAD

void TouchDesigner::menuCallback(MenuItem* menuItem)
{
	if(menuItem == _receiveButton)
	{
=======
void TouchDesigner::menuCallback(MenuItem* menuItem) {
	if (menuItem == _receiveButton) {
>>>>>>> e7aefd8ba4f234f3d166f0a934e5cf6ea045a343
		receiveGeometry();
		return;
	}
}

double TouchDesigner::random() {
	return rand() / double(RAND_MAX);
}

<<<<<<< HEAD
void TouchDesigner::receiveGeometry()
{
	Vec4d nil(0.0,0.0,0.0,0.0);
	Vec4d krayBlue(0.0,0.5,1.0,1.0);

	string dummyT = "trianglec cx=0 cy=0 cz=0 length=300 c1r=0 c1g=0.5 c1b=1.0 c1a=0.8 c2r=0 c2g=0 c2b=0 c2a=1 comment=sh\0";
	string dummyTP = "trianglep p1x=0 p1y=0 p1z=0  p2x=100 p2y=0 p2z=0 p3x=50 p3y=0 p3z=50 c1r=0 c1g=0.5 c1b=1.0 c1a=0.8 comment=sh\0";
	string dummyUTP = "0 p1x=-100 p1y=0 p1z=0  p2x=0 p2y=0 p2z=0 p3x=150 p3y=0 p3z=150 c1r=0 c1g=0.5 c1b=0.7 c1a=0.6 c2r=0 c2g=0 c2b=0 c2a=1 comment=sh\0";
	string dummyC = "circle cx=-300 cy=0 cz=-300 radius=100 c1r=0 c1g=0.5 c1b=1.0 c1a=0.8 c2r=0 c2g=0 c2b=0 c2a=1 comment=sh\0";
	string dummyUC = "updatec id=0 cx=0 cy=0 cz=-300 radius=100 c1r=0.2 c1g=0.7 c1b=0.6 c1a=0.8 c2r=0 c2g=0 c2b=0 c2a=1 comment=sh\0";
	string dummyUR = "updaterc id=1 cx=0 cy=0 cz=300 height=300 width=300 c1r=0.3 c1g=0.7 c1b=0.4 c1a=0.8 c2r=0 c2g=0 c2b=0 c2a=1 comment=sh\0";
	string dummyUT = "updatetc id=0 cx=300 cy=0 cz=0 length=200 c1r=0.3 c1g=0.5 c1b=0.7 c1a=0.8 c2r=0 c2g=0 c2b=0 c2a=1 comment=sh\0";
	string dummyR = "rectc cx=300 cy=0 cz=300 height=300 width=300 c1r=0 c1g=0.5 c1b=1.0 c1a=0.8 c2r=0 c2g=0 c2b=0 c2a=1 comment=sh\0";

	string dummyRP = "rectp p1x=0 p1y=0 p1z=0  p2x=100 p2y=0 p2z=0 p3x=100 p3y=0 p3z=100 p4x=0 p4y=0 p4z=100 c1r=0 c1g=0.5 c1b=1.0 c1a=0.8 comment=sh\0";
	//string dummyURP = "updaterp p1x=-100 p1y=0 p1z=0  p2x=0 p2y=0 p2z=0 p3x=0 p3y=0 p3z=100 p4x=-100 p4y=0 p4z=100 c1r=0 c1g=0.5 c1b=1.0 c1a=0.8 c2r=0 c2g=0 c2b=0 c2a=1 comment=sh\0";

	string dummyURP = "1 p1x=-100 p1y=0 p1z=0  p2x=0 p2y=0 p2z=0 p3x=0 p3y=0 p3z=100 p4x=-100 p4y=0 p4z=100 c1r=0 c1g=0.5 c1b=1.0 c1a=0.8 c2r=0 c2g=0 c2b=0 c2a=1 comment=sh\0";
	

	
  TrackerTree* tt = new TrackerTree();
  tt->root = tt->insert(string("Sphere1"),tt->root);
  tt->printTree(tt->root);
  tt->root = tt->insert(string("Sphere2"),tt->root);
  tt->printTree(tt->root);
  tt->root = tt->insert(string("Apple"),tt->root);
  tt->printTree(tt->root);
  tt->root = tt->insert(string("Bam"),tt->root);
  tt->printTree(tt->root);
  tt->root = tt->remove(string("Whopper"),tt->root);
  tt->printTree(tt->root);
  tt->root = tt->remove(string("Apple"),tt->root);
  tt->printTree(tt->root);
  tt->root = tt->insert(string("Accle"),tt->root);
  tt->printTree(tt->root);
  tt->root = tt->insert(string("Abble"),tt->root);
  tt->root = tt->insert(string("Apple"),tt->root);
  tt->printTree(tt->root);
  tt->root = tt->insert(string("AAAAAA"),tt->root);
  tt->printTree(tt->root);
  tt->root = tt->insert(string("AAAAAA"),tt->root);
  tt->printTree(tt->root);
  tt->root = tt->insert(string("AAAAAA"),tt->root);
  tt->printTree(tt->root);
  tt->root = tt->remove(string("AAAAAA"),tt->root);
  tt->printTree(tt->root);
  
  TrackerNode* tem = tt->get(string("Apple"),tt->root);
  cerr<<"tem = "<<tem->comment<<endl;
  tem = tt->get(string("AAAAAA"),tt->root);
  cerr<<"tem = "<<tem->comment<<endl;
  tem = tt->get(string("AAAAAA"),tt->root);
  cerr<<"tem = "<<tem->comment<<endl;

  //shapes
  Sphere *ss = new Sphere(Vec3(0,0,0),100);
  ShapeDrawable* ssD = new ShapeDrawable(ss);
  ssD->setColor(Vec4(0,0,1,1));
	
	Box *bx = new Box(Vec3(250,0,0),100);
	ShapeDrawable* bxD = new ShapeDrawable(bx);
	bxD->setColor(Vec4(1,0,0,1));
	
	Geode *node = new Geode();
	node->addDrawable(ssD);
	node->addDrawable(bxD);
	
	
	//textures
	Texture2D* tex = new Texture2D;
	tex->setDataVariance(osg::Object::DYNAMIC);
	Image* img = osgDB::readImageFile("texture.tga");
	if(!img) cerr<<"can't find texture"<<endl;
	tex->setImage(img);
	
	StateSet* stateblend = new StateSet();
	TexEnv* blendtexenv = new TexEnv;
	blendtexenv->setMode(osg::TexEnv::BLEND);
	stateblend->setTextureAttributeAndModes(0,tex,osg::StateAttribute::ON);
	stateblend->setTextureAttribute(0,blendtexenv);
	
	StateSet* statedec = new StateSet();
	TexEnv* dectexenv = new TexEnv;
	dectexenv->setMode(osg::TexEnv::DECAL);
	statedec->setTextureAttributeAndModes(0,tex,osg::StateAttribute::ON);
	statedec->setTextureAttribute(0,dectexenv);
	
	SceneManager::instance()->getObjectsRoot()->setStateSet(stateblend);
  node->setStateSet(statedec);

	SceneManager::instance()->getObjectsRoot()->addChild(node);

	//(st->getTestSH())->processData(&dummyUC[0]);

}

=======
void TouchDesigner::receiveGeometry() {

}

>>>>>>> e7aefd8ba4f234f3d166f0a934e5cf6ea045a343
int TouchDesigner::random(int min, int max) {
	return (max - min) * random() + min;
}


