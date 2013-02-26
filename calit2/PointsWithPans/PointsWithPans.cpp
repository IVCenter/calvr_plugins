#include "PointsWithPans.h"

#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/PluginHelper.h>
#include <cvrKernel/PluginManager.h>
#include <cvrUtil/PointsNode.h>
#include <PluginMessageType.h>

#include <sstream>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <sys/stat.h>

#include "PanMarkerObject.h"

using namespace cvr;

CVRPLUGIN(PointsWithPans)

PointsWithPans::PointsWithPans()
{
    _loadedSetIndex = -1;
}

PointsWithPans::~PointsWithPans()
{
}

bool PointsWithPans::init()
{

    _pwpMenu = new SubMenu("PointsWithPans");
    PluginHelper::addRootMenuItem(_pwpMenu);

    _setMenu = new SubMenu("Sets");
    _pwpMenu->addItem(_setMenu);

    _removeButton = new MenuButton("Remove");
    _removeButton->setCallback(this);
    _pwpMenu->addItem(_removeButton);

    std::vector<std::string> setTags;
    ConfigManager::getChildren("Plugin.PointsWithPans.Sets",setTags);

    for(int i = 0; i < setTags.size(); i++)
    {
	std::stringstream setss;
	setss << "Plugin.PointsWithPans.Sets." << setTags[i]; 
	PWPSet * set = new PWPSet;
	set->scale = ConfigManager::getFloat("scale",setss.str(),1.0);
	set->pointSize = ConfigManager::getFloat("pointSize",setss.str(),0.005);
	float x,y,z;
	x = ConfigManager::getFloat("x",setss.str(),0);
	y = ConfigManager::getFloat("y",setss.str(),0);
	z = ConfigManager::getFloat("z",setss.str(),0);
	set->offset = osg::Vec3(x,y,z);
	set->file = ConfigManager::getEntry("file",setss.str(),"",NULL);
	set->moveTime = ConfigManager::getFloat("moveTime",setss.str(),4.0);
	set->fadeTime = ConfigManager::getFloat("fadeTime",setss.str(),5.0);
	std::vector<std::string> panTags;
	ConfigManager::getChildren(setss.str(),panTags);
	float radius = ConfigManager::getFloat("sphereRadius",setss.str(),250.0);
	float distance = ConfigManager::getFloat("selectDistance",setss.str(),2500.0);
	for(int j = 0; j < panTags.size(); j++)
	{
	    std::stringstream panss;
	    panss << setss.str() << "." << panTags[j];
	    x = ConfigManager::getFloat("x",panss.str(),0);
	    y = ConfigManager::getFloat("y",panss.str(),0);
	    z = ConfigManager::getFloat("z",panss.str(),0);
	    PWPPan pan;
	    pan.location = osg::Vec3(x,y,z);
	    pan.name = ConfigManager::getEntry("name",panss.str(),"",NULL);
	    pan.rotationOffset = ConfigManager::getFloat("rotationOffset",panss.str(),0);
	    pan.rotationOffset = pan.rotationOffset * M_PI / 180.0;
	    pan.sphereRadius = ConfigManager::getFloat("sphereRadius",panss.str(),radius);
	    pan.selectDistance = ConfigManager::getFloat("selectDistance",panss.str(),distance);
	    pan.textureFile = ConfigManager::getEntry("textureFile",panss.str(),"");
	    set->panList.push_back(pan);
	}
	_setList.push_back(set);
	MenuButton * button = new MenuButton(setTags[i]);
	button->setCallback(this);
	_setMenu->addItem(button);
	_buttonList.push_back(button);
    }

    _activeObject = new PointsObject("Points",true,false,false,true,false);
    PluginHelper::registerSceneObject(_activeObject,"PointsWithPans");

    return true;
}

void PointsWithPans::menuCallback(MenuItem * item)
{
    if(item == _removeButton)
    {
	if(_loadedSetIndex >= 0)
	{
	    _loadedSetIndex = -1;
	    _activeObject->clear();
	}
	return;
    }

    for(int i = 0; i < _buttonList.size(); i++)
    {
	if(item == _buttonList[i])
	{
	    if(_loadedSetIndex >= 0)
	    {
		_activeObject->clear();
	    }

	    if(!PluginManager::instance()->getPluginLoaded("Points"))
	    {
		std::cerr << "PointsWithPans: Error, Points plugin is not loaded." << std::endl;
		return;
	    }

#ifndef USE_POINTS_NODE
	    PointsLoadInfo pli;
	    pli.file = _setList[i]->file;
	    pli.group = new osg::Group();

	    PluginHelper::sendMessageByName("Points",POINTS_LOAD_REQUEST,(char*)&pli);

	    if(!pli.group->getNumChildren())
	    {
		std::cerr << "PointsWithPans: Error, no points loaded for file: " << pli.file << std::endl;
		return;
	    }
#else
	    PointsNode * node = createPointsNode(_setList[i]->file,1.0f,_setList[i]->pointSize);
	    if(!node)
	    {
		std::cerr << "PointsWithPans: Error, no points loaded for file: " << _setList[i]->file << std::endl;
		return;
	    }
#endif

	    _activeObject->attachToScene();
	    _activeObject->setNavigationOn(false);
	    _activeObject->setScale(_setList[i]->scale);
	    _activeObject->setPosition(_setList[i]->offset);
	    _activeObject->setRotation(osg::Quat());
	    _activeObject->setNavigationOn(true);
#ifndef USE_POINTS_NODE
	    _activeObject->addChild(pli.group.get());
#else
	    _activeObject->addChild(node);
#endif
	    _activeObject->setTransitionTimes(_setList[i]->moveTime,_setList[i]->fadeTime);

#ifndef USE_POINTS_NODE
	    _scaleUni = new osg::Uniform("pointScale",1.0f * _setList[i]->pointSize);
	    pli.group->getOrCreateStateSet()->addUniform(_scaleUni);
	    _scaleUni->set((float)_activeObject->getObjectToWorldMatrix().getScale().x());
#endif

	    for(int j = 0; j < _setList[i]->panList.size(); j++)
	    {
		PanMarkerObject * pmo = new PanMarkerObject(_setList[i]->scale,_setList[i]->panList[j].rotationOffset,_setList[i]->panList[j].sphereRadius,_setList[i]->panList[j].selectDistance,_setList[i]->panList[j].name,_setList[i]->panList[j].textureFile,false,false,false,true,false);
		_activeObject->addChild(pmo);

		osg::Matrix m;
		m.makeTranslate(_setList[i]->panList[j].location);
		pmo->setTransform(m);
	    }

	    _loadedSetIndex = i;
	    return;
	}
    }
}

void PointsWithPans::preFrame()
{
#ifndef USE_POINTS_NODE
    if(_loadedSetIndex >= 0 && _scaleUni && _activeObject)
    {
	_scaleUni->set((float)(_activeObject->getObjectToWorldMatrix().getScale().x()*_setList[_loadedSetIndex]->pointSize));
    }
#endif

    if(_loadedSetIndex >= 0)
    {
	_activeObject->update();
    }
}

void PointsWithPans::message(int type, char *&data, bool collaborative)
{
    if(type == PWP_PAN_UNLOADED)
    {
	if(_loadedSetIndex >= 0)
	{
	    float rotation = *((float*)data);
	    _activeObject->panUnloaded(rotation);
	}
    }
}

PointsNode * PointsWithPans::createPointsNode(std::string file, float pointSize, float pointRadius)
{
    size_t pos = file.find_last_of('.');
    if(pos == std::string::npos || pos + 1 == file.length())
    {
	std::cerr << "Unable to determine extension for file: " << file << std::endl;
	return NULL;
    }

    std::string ext = file.substr(pos+1);
    std::transform(ext.begin(),ext.end(),ext.begin(),::tolower);

    osg::ref_ptr<osg::Vec3Array> verts;
    osg::ref_ptr<osg::Vec4ubArray> colors;

    if(ext == "ply")
    {
    }
    else if(ext == "xyz")
    {
    }
    else if(ext == "xyb")
    {
	std::ifstream inFile;
	inFile.open(file.c_str(),std::ios::in|std::ios::binary);
	if(inFile.fail())
	{
	    std::cerr << "Unable to open file: " << file << std::endl;
	    return NULL;
	}

	struct stat fileStat;
	stat(file.c_str(), &fileStat);
	int numPoints = fileStat.st_size / (sizeof(float)*6);
	if(!numPoints)
	{
	    return NULL;
	}

	verts = new osg::Vec3Array(numPoints);
	colors = new osg::Vec4ubArray(numPoints);

	float xyzrgb[6];
	for(int i = 0; i < numPoints; ++i)
	{
	    inFile.read((char*)xyzrgb,sizeof(float)*6);
	    verts->at(i).x() = xyzrgb[0];
	    verts->at(i).y() = xyzrgb[1];
	    verts->at(i).z() = xyzrgb[2];
	    colors->at(i).r() = (unsigned char)(xyzrgb[3]*255.0);
	    colors->at(i).g() = (unsigned char)(xyzrgb[4]*255.0);
	    colors->at(i).b() = (unsigned char)(xyzrgb[5]*255.0);
	    colors->at(i).a() = (unsigned char)255;
	}
    }
    else
    {
	std::cerr << "No point parser for file: " << file << std::endl;
	return NULL;
    }

    if(!verts || !colors)
    {
	return NULL;
    }

    PointsNode * pNode = new PointsNode(PointsNode::POINTS_SHADED_SPHERES,0,pointSize,pointRadius,osg::Vec4(1.0,1.0,1.0,1.0),PointsNode::POINTS_OVERALL,PointsNode::POINTS_OVERALL);
    pNode->setVertexArray(verts);
    pNode->setColorArray(colors);

    return pNode;
}
