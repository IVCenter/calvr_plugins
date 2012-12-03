#ifndef _MUGIC_
#define _MUGIC_

#include <cvrKernel/CVRPlugin.h>
#include <cvrMenu/SubMenu.h>
#include <cvrMenu/MenuButton.h>
#include <cvrKernel/PluginHelper.h>
#include <OpenThreads/Condition>

#include "SocketThread.h"
#include "CommandParser.h"
#include "ThreadQueue.h"

#include <sstream>
#include <fstream>
#include <iostream>

using namespace std;
using namespace osg;

class Mugic: public cvr::CVRPlugin
{
public:        
	Mugic();
	virtual ~Mugic();
	bool init();
	void preFrame();

protected:
	OpenThreads::Condition* _parserSignal;
    osg::Group * _root;
	SocketThread * _st;
	CommandParser * _parser;
    ThreadQueue<std::string> * _commands;
};

#endif
