#include "PointsWithPans.h"

#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/PluginHelper.h>
#include <cvrKernel/PluginManager.h>
#include <PluginMessageType.h>

#include <sstream>
#include <iostream>

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

	    PointsLoadInfo pli;
	    pli.file = _setList[i]->file;
	    pli.group = new osg::Group();

	    PluginHelper::sendMessageByName("Points",POINTS_LOAD_REQUEST,(char*)&pli);

	    if(!pli.group->getNumChildren())
	    {
		std::cerr << "PointsWithPans: Error, no points loaded for file: " << pli.file << std::endl;
		return;
	    }

	    _activeObject->attachToScene();
	    _activeObject->setNavigationOn(false);
	    _activeObject->setScale(_setList[i]->scale);
	    _activeObject->setPosition(_setList[i]->offset);
	    _activeObject->setNavigationOn(true);
	    _activeObject->addChild(pli.group.get());
	    _activeObject->setTransitionTimes(_setList[i]->moveTime,_setList[i]->fadeTime);

	    /*_sizeUni = pli.group->getOrCreateStateSet()->getUniform("pointSize");
	    if(_sizeUni)
	    {
		_sizeUni->set(_setList[i]->pointSize);
	    }*/
	    _scaleUni = new osg::Uniform("pointScale",1.0f * _setList[i]->pointSize);
	    pli.group->getOrCreateStateSet()->addUniform(_scaleUni);
	    _scaleUni->set((float)_activeObject->getObjectToWorldMatrix().getScale().x());

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
    if(_loadedSetIndex >= 0 && _scaleUni && _activeObject)
    {
	_scaleUni->set((float)(_activeObject->getObjectToWorldMatrix().getScale().x()*_setList[_loadedSetIndex]->pointSize));
    }

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
