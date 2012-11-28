#include "Mugic.h"

#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/SceneManager.h>
#include <cvrMenu/MenuSystem.h>
#include <cvrKernel/PluginHelper.h>
#include <cvrKernel/ComController.h>
#include <cvrUtil/CVRSocket.h>
#include <cvrKernel/SceneObject.h>
#include <cvrKernel/NodeMask.h>

#include <fcntl.h>
#include <iostream>
#include <cstring>

#include <osg/Matrix>
#include <osg/CullFace>
#include <osg/Material>
#include <osg/Group>
#include <osgDB/ReadFile>

using namespace osg;
using namespace std;
using namespace cvr;

CVRPLUGIN( Mugic)

Mugic::Mugic() 
{
}

bool Mugic::init() 
{
	std::cerr << "Mugic init\n";
    _st = NULL;
    _parser = NULL;
	_parserSignal = new OpenThreads::Condition();
    
    _commands = new ThreadQueue<std::string> ();
    _commands->setCondition(_parserSignal);

    std::string routerAddress = ConfigManager::getEntry("value", "Plugin.Mugic.Router", "127.0.0.1");
	_st = new SocketThread(_commands, routerAddress);

    _root = new osg::Group();
    _root->setNodeMask(_root->getNodeMask() & ~DISABLE_FIRST_CULL & ~INTERSECT_MASK);

    // set default stateset to _root node
    osg::StateSet* state = _root->getOrCreateStateSet();
    state->setMode(GL_BLEND, osg::StateAttribute::ON);
    osg::Material* mat = new osg::Material();
    mat->setAlpha(osg::Material::FRONT_AND_BACK, 0.1);
    mat->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
    state->setAttributeAndModes(mat, osg::StateAttribute::ON);

     PluginHelper::getObjectsRoot()->addChild(_root);
	_parser = new CommandParser(_commands, _root);

	return true;
}

Mugic::~Mugic() 
{
    if( _st )
        delete _st;
    _st = NULL;
    
    if( _parser )
        delete _parser;
    _parser = NULL;

    if( _commands )
        delete _commands;
    _commands = NULL;
}

void Mugic::preFrame() 
{
}
