#include "EZVis.h"

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

bool Globals::G_ROTAXIS = 0;

CVRPLUGIN( EZVis)

EZVis::EZVis() 
{
}

bool EZVis::init() 
{
	std::cerr << "EZVis init\n";
    _st = NULL;
    _parser = NULL;
    _router = NULL;
	_parserSignal = new OpenThreads::Condition();
    
    _commands = new ThreadQueue<std::string> ();
    _commands->setCondition(_parserSignal);

    // works by default if just running on head node else needs to be set
    std::string routerAddress = ConfigManager::getEntry("value", "Plugin.EZVis.RouterIP", "127.0.0.1");
	_st = new SocketThread(_commands, routerAddress);

    // global params
    Globals::G_ROTAXIS = (ConfigManager::getEntry("value", "Plugin.EZVis.RotAxis", "no") == "yes");
    float scale = ConfigManager::getFloat("value", "Plugin.EZVis.Scale", 1.0);
   
    // add root node to he scene 
    _root = new MainNode(scale, Globals::G_ROTAXIS);
    _root->setNodeMask(_root->getNodeMask() & ~DISABLE_FIRST_CULL & ~INTERSECT_MASK);
    PluginHelper::getObjectsRoot()->addChild(_root);

    // enable router on headNode
    if(ComController::instance()->isMaster())
    {
        // start router thread
        _router = new Router(ConfigManager::getInt("value", "Plugin.EZVis.RouterPort", 16667));
    }
    
	_parser = new CommandParser(_commands, _root);

	return true;
}

EZVis::~EZVis() 
{
    std::cerr << "EZViz Destructor called\n";
    
    if( _st )
        delete _st;
    _st = NULL;
    
    if( _commands )
        delete _commands;
    _commands = NULL;
    
    if( _parser )
        delete _parser;
    _parser = NULL;
	
    if( _router )
        delete _router;
    _router = NULL;
}
