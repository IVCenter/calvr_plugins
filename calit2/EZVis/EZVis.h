#ifndef _EZVIS_
#define _EZVIS_

#include <cvrKernel/CVRPlugin.h>
#include <cvrMenu/SubMenu.h>
#include <cvrMenu/MenuButton.h>
#include <cvrKernel/PluginHelper.h>
#include <OpenThreads/Condition>

#include "SocketThread.h"
#include "CommandParser.h"
#include "ThreadQueue.h"
#include "MainNode.h"
#include "Router.h"
#include "Globals.h"

#include <sstream>
#include <fstream>
#include <iostream>

using namespace std;
using namespace osg;

class EZVis: public cvr::CVRPlugin
{
    public:        
	    EZVis();
	    virtual ~EZVis();
	    bool init();

    protected:
	    OpenThreads::Condition* _parserSignal;
        MainNode * _root;
	    SocketThread * _st;
	    CommandParser * _parser;
        ThreadQueue<std::string> * _commands;
        Router* _router; // only runs on headnode to process commands
};

#endif
