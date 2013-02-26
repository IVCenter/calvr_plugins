#include "PluginTest.h"

#include <osg/Geometry>
#include <osgDB/WriteFile>

#include <iostream>
#include <cvrMenu/MenuSystem.h>
#include <cvrMenu/SubMenu.h>
#include <cvrKernel/PluginHelper.h>
#include <cvrKernel/ComController.h>
#include <cvrKernel/InteractionManager.h>
#include <cvrKernel/ThreadedLoader.h>
#include <cvrKernel/SceneManager.h>

#include <osgDB/ReadFile>

#include <cstring>
#include <cstdlib>
#include <time.h>

CVRPLUGIN(PluginTest)

using namespace cvr;
using namespace osg;

PluginTest::PluginTest()
{
    std::cerr << "PluginTest created." << std::endl;
    _loading = false;
    srand(time(NULL));
}

PluginTest::~PluginTest()
{
    std::cerr << "PluginTest destroyed." << std::endl;
    //delete testButton1;
    delete testButton2;
    delete testButton3;
    delete testButton4;
    delete testButton5;
    delete menu1;
    delete menu2;
    delete menu3;
    //delete pmenu1;
    delete checkbox1;
    delete rangeValue;
    //delete pcheckbox1;
    //delete pbutton1;
}

bool PluginTest::init()
{
    std::cerr << "PluginTest init()." << std::endl;

    menu1 = new SubMenu("My Menu1","Menu Bar1");
    menu1->setCallback(this);

    menu2 = new SubMenu("My Menu2","Menu Bar2");
    menu2->setCallback(this);

    menu3 = new SubMenu("My Menu3","Menu Bar3");
    menu3->setCallback(this);

    testButton1 = new MenuButton("Test Button 1");
    testButton1->setCallback(this);

    testButton2 = new MenuButton("Test Button 2");
    testButton2->setCallback(this);

    testButton3 = new MenuButton("Test Button 3");
    testButton3->setCallback(this);

    testButton4 = new MenuButton("Test Button 4");
    testButton4->setCallback(this);

    testButton5 = new MenuButton("Test Button 5");
    testButton5->setCallback(this);

    checkbox1 = new MenuCheckbox("Checkbox 1",true);
    checkbox1->setCallback(this);

    rangeValue = new MenuRangeValue("RangeValue 1",0.01,10.0,5.0);
    rangeValue->setCallback(this);

    textButtonSet1 = new MenuTextButtonSet(true, 400, 40, 3);
    textButtonSet1->setCallback(this);
    textButtonSet1->addButton("This");
    textButtonSet1->addButton("Is A");
    textButtonSet1->addButton("Test");
    textButtonSet1->addButton("Of");
    textButtonSet1->addButton("Multiple");
    textButtonSet1->addButton("Rows");

    menu1->addItem(testButton1);
    menu1->addItem(rangeValue);
    menu1->addItem(textButtonSet1);
    menu2->addItem(testButton3);
    menu2->addItem(menu3);
    menu2->addItem(checkbox1);
    menu3->addItem(testButton4);
    menu3->addItem(testButton5);

    MenuSystem::instance()->addMenuItem(menu1);
    MenuSystem::instance()->addMenuItem(menu2);
    MenuSystem::instance()->addMenuItem(testButton2);

    //MenuSystem::instance()->addMenuItem(testButton1);
    //MenuSystem::instance()->addMenuItem(testButton2);

    popup1 = new PopupMenu("Test Popup");
    popup1->setVisible(false);

    // test text from random wikipedia articles
    _mst = new MenuScrollText("There is a great range of specialisations within the ANC. Environmental noise consultants carry out measurement, calculation, evaluation and mitigation of noise pollution to fit within current noise regulation and produce an environmental impact assessment often leading to appearance as an expert witness at public inquiries. In building acoustics, sound insulation is tested between dwellings as required by approved document E of the Building Regulations, schools are designed for optimal learning conditions and the acoustic environments of performing arts venues are designed for their specific intended purposes.\n",500,4,1.0,false);
    popup1->addMenuItem(_mst);
    _mst->appendText("Vipul's Razor is a checksum-based, distributed, collaborative, spam-detection-and-filtering network. Through user contribution, Razor establishes a distributed and constantly updating catalogue of spam in propagation that is consulted by email clients to filter out known spam. Detection is done with statistical and randomized signatures that efficiently spot mutating spam content. User input is validated through reputation assignments based on consensus on report and revoke assertions which in turn is used for computing confidence values associated with individual signatures. \n");

    createPointsNode();
    return true;

    pcheckbox1 = new MenuCheckbox("Popup Check", true);
    pmenu1 = new SubMenu("Popup menu line","Popup Menu head");
    pbutton1 = new MenuButton("Popup button");
    pmenu1->addItem(pbutton1);

    popup1->addMenuItem(pcheckbox1);
    popup1->addMenuItem(pmenu1);

    tdp1 = new TabbedDialogPanel(400,40,3,"Tabbed Dialog Panel");
    tdp1->setVisible(false);

    tdp1->addTextTab("Tab1","I am Tab 1");
    tdp1->addTextTab("Tab2","I am Tab 2");
    tdp1->addTextTab("Tab3","This is a test of the maxWidth attribute of osgText::Text. Hopefully if will wrap on whitespace.");

    //createSphereTexture();

    _testobj = new SceneObject("My Test Object", true, true, false, true, true);
    osg::Node * node = osgDB::readNodeFile("/home/aprudhom/data/heart/heart00.iv");
    if(node)
    {
	_testobj->addChild(node);
    }

    _testobj->setScale(0.3);
    _testobj->addMoveMenuItem();
    //SceneManager::instance()->registerSceneObject(_testobj,"PluginTest");
    //_testobj->attachToScene();

    _testobj2 = new SceneObject("TestObject2", true, true, false, true, true);
    _testobj2->addMoveMenuItem();
    _testobj2->addNavigationMenuItem();
    node = osgDB::readNodeFile("/home/aprudhom/data/PDB/cache/4HHBcart.wrl");
    if(node)
    {
	_testobj2->addChild(node);
    }

    SceneManager::instance()->registerSceneObject(_testobj2,"PluginTest");
    _testobj2->attachToScene();
    _testobj2->setScale(20.0);
    _testobj2->addChild(_testobj);

    //std::cerr << "NodeMask: " << PluginHelper::getObjectsRoot()->getNodeMask() << std::endl;

    return true;
}

void PluginTest::menuCallback(MenuItem * item)
{
    std::cerr << "Got menu item callback." << std::endl;
    if(item == testButton1)
    {
	std::cerr << "Test Button 1" << std::endl;
	menu1->removeItem(testButton1);
	delete testButton1;
    }
    else if(item == testButton2)
    {
	std::cerr << "Test Button 2" << std::endl;
    }
    else if(item == testButton3)
    {
	std::cerr << "Test Button 3" << std::endl;
	for(int i = 0; i < menu3->getNumChildren(); i++)
	{
	    MenuItem * mi = menu3->getChild(i);
	    menu3->removeItem(menu3->getChild(i));
	    delete mi;
	}
	menu2->removeItem(menu3);
	delete menu3;
    }
    else if(item == testButton4)
    {
	std::cerr << "Test Button 4" << std::endl;
    }
    else if(item == testButton5)
    {
	std::cerr << "Test Button 5" << std::endl;
    }
    else if(item == checkbox1)
    {
	std::cerr << "Checkbox 1 value: " << checkbox1->getValue() << std::endl;
    }
    else if(item == rangeValue)
    {
	std::cerr << "RangeValue 1 value: " << rangeValue->getValue() << std::endl;
    }
}

void PluginTest::preFrame()
{
    if(_pointsNode)
    {
	/*for(int i = 0; i < _pointsNode->getNumPoints(); i++)
	{
	    float mult = 0.001;
	    osg::Vec3 diffvec(((rand() % 10 + 1) - 5) * mult, ((rand() % 10 + 1) - 5) * mult, ((rand() % 10 + 1) - 5) * mult);
	    _pointsNode->setPointPosition(i,_pointsNode->getPointPosition(i) + diffvec);
	}*/
	if(_pointsNode->getNumPoints())
	{
	int delindex = rand() % _pointsNode->getNumPoints();
	_pointsNode->removePoints(delindex,10000);
	}
    }
    //std::cerr << "PluginTest preFrame()." << std::endl;
    if(_loading)
    {
	//std::cerr << "Loading status: " << ThreadedLoader::instance()->getProgress(_job) << std::endl;
	if(ThreadedLoader::instance()->isDone(_job))
	{
	    std::cerr << "Job done." << std::endl;
	    osg::ref_ptr<osg::Node> node;
	    ThreadedLoader::instance()->getNodeFile(_job,node);
	    if(!node)
	    {
		std::cerr << "Node is null." << std::endl;
	    }
	    else
	    {
		PluginHelper::getObjectsRoot()->addChild(node);
	    }
	    ThreadedLoader::instance()->remove(_job);
	    _loading = false;
	}
    }
    //testMulticast();
}

void PluginTest::postFrame()
{
    //std::cerr << "PluginTest postFrame()." << std::endl;
}

bool PluginTest::processEvent(InteractionEvent * event)
{
    if(event->asKeyboardEvent())
    {
	if(event->getInteraction() == KEY_UP && _pointsNode)
	{
	    if(_pointMode == PointsNode::POINTS_GL_POINTS)
	    {
		_pointMode = PointsNode::POINTS_SHADED_SPHERES;
	    }
	    else if(_pointMode == PointsNode::POINTS_SHADED_SPHERES)
	    {
		_pointMode = PointsNode::POINTS_GL_POINTS;
	    }
	    _pointsNode->setPointsMode(_pointMode);
	}
    }
    return false;
}

void PluginTest::createSphereTexture()
{
    osg::Image * image = new osg::Image();
    image->allocateImage(256,256,1,GL_RED,GL_FLOAT);
    image->setInternalTextureFormat(1);

    float * textureData = (float *)image->data();

    int index = 0;
    for(int i = 0; i < 256; i++)
    {
	for(int j = 0; j < 256; j++)
	{
	    float x, y, z;
	    x = ((float)j) / 255.0;
	    x -= 0.5;
	    x *= 2.0;

	    y = ((float)i) / 255.0;
	    y -= 0.5;
	    y *= 2.0;

	    if(x*x + y*y > 1.0)
	    {
		textureData[index] = 0;
	    }
	    else
	    {
		z = sqrt(1.0 - x*x - y*y);
		textureData[index] = z;
	    }
	    index++;
	}
    }

    osg::Texture2D * texture = new osg::Texture2D(image);

    osg::Geometry * geometry = new osg::Geometry();

    Vec3Array* vertices = new Vec3Array(4);
    (*vertices)[0].set(-100,0,-100);
    (*vertices)[1].set(100,0,-100);
    (*vertices)[2].set(100,0,100);
    (*vertices)[3].set(-100,0,100);
    geometry->setVertexArray(vertices);

    /*Vec4Array* colors = new Vec4Array(1);
    (*colors)[0].set(1.0, 0.0, 0.0, 1.0);
    geometry->setColorArray(colors);
    geometry->setColorBinding(Geometry::BIND_OVERALL);*/

    Vec2Array* texcoords = new Vec2Array(4);
    (*texcoords)[0].set(0.0, 0.0);
    (*texcoords)[1].set(1.0, 0.0);
    (*texcoords)[2].set(1.0, 1.0);
    (*texcoords)[3].set(0.0, 1.0);
    geometry->setTexCoordArray(0,texcoords);

    geometry->addPrimitiveSet(new DrawArrays(PrimitiveSet::QUADS, 0, 4));

    StateSet* stateset = geometry->getOrCreateStateSet();
    stateset->setMode(GL_LIGHTING, StateAttribute::OFF);
    stateset->setTextureAttributeAndModes(0, texture, StateAttribute::ON);

    osg::Geode * geode = new osg::Geode();
    geode->addDrawable(geometry);
    PluginHelper::getScene()->addChild(geode);

    if(geometry->areFastPathsUsed())
    {
	std::cerr << "Using GL fast path." << std::endl;
    }
    else
    {
	std::cerr << "Not using GL fast path." << std::endl;
    }
}

void PluginTest::createPointsNode()
{
    std::cerr << "Points load start." << std::endl;
    _pointsMT = new osg::MatrixTransform();
    osg::Matrix m;
    m.makeScale(osg::Vec3(1000.0,1000.0,1000.0));
    _pointsMT->setMatrix(m);

    _pointMode = PointsNode::POINTS_GL_POINTS;

    std::ifstream infile("/home/aprudhom/trishPans/gal12NdrfFilterFilt-r20.xyb", std::ios::in|std::ios::binary);
    if(!infile.fail())
    {
	infile.seekg(0, std::ios::end);
	std::ifstream::pos_type size = infile.tellg();
	infile.seekg(0, std::ios::beg);

	int num = size / ((sizeof(float) * 3) + (sizeof(float) * 3));

	_pointsNode = new PointsNode(_pointMode,num,1.0f,0.005f,osg::Vec4ub(255,0,0,255));

	osg::Vec3 pos;
	osg::Vec4ub color(255,255,255,255);
	float tempColor[3];
	for(int i = 0; i < num; i++)
	{
	    infile.read((char*)pos.ptr(), sizeof(float) * 3);
	    infile.read((char*)tempColor, sizeof(float) * 3);
	    color.r() = (char)(tempColor[0] * 255.0);
	    color.g() = (char)(tempColor[1] * 255.0);
	    color.b() = (char)(tempColor[2] * 255.0);

	    _pointsNode->setPoint(i,pos,color,0.01f,1.0f + ((float)i) * 0.000003);
	}
    }
    else
    {
	std::cerr << "PluginTest: Unable to open test points file." << std::endl;
	return;
    }

    _pointsMT->addChild(_pointsNode);
    PluginHelper::getObjectsRoot()->addChild(_pointsMT);

    std::cerr << "Points load end." << std::endl;
}

void PluginTest::testMulticast()
{
    int * data = new int[4000];

    if(ComController::instance()->isMaster())
    {
	for(int i = 0; i < 4000; i++)
	{
	    data[i] = i % 10;
	}

	ComController::instance()->sendSlavesMulticast(data,4000*sizeof(int));
    }
    else
    {
	memset(data,0,4000*sizeof(int));
	ComController::instance()->readMasterMulticast(data,4000*sizeof(int));
	std::cerr << "Got data : ";
	bool ok = true;
	for(int i = 0; i < 4000; i++)
	{
	    if(data[i] != i % 10)
	    {
		ok = false;
	    }
	}
	std::cerr << ok << std::endl;
    }

    delete[] data;
}
