#include "DumpTracking.h"

#include <cstdio>
#include <cvrKernel/PluginHelper.h>
#include <cvrKernel/ComController.h>
#include <cvrConfig/ConfigManager.h>
#include <string>

CVRPLUGIN(DumpTracking)

using namespace std;
using namespace cvr;
using namespace osg;

DumpTracking::DumpTracking()
{
    _stringOut = NULL;
}

DumpTracking::~DumpTracking()
{
    if(ComController::instance()->isMaster())
    {
	if(_file.good())
	{
	    _file.close();
	}

	if(_stringOut)
	{
	    delete[] _stringOut;
	}
    }
}

bool DumpTracking::init()
{
    if(ComController::instance()->isMaster())
    {
	std::string filename = ConfigManager::getEntry("Plugin.DumpTracking.File");
	_file.open(filename.c_str(), fstream::out | fstream::trunc);
	if(_file.fail())
	{
	    std::cerr << "Error opening file: " << std::endl;
	    return false;
	}

	_stringOut = new char[255];
    }
    return true;
}

void DumpTracking::preFrame()
{
    if(ComController::instance()->isMaster())
    {
	_file.seekp(0);
	Vec3 pos = PluginHelper::getHeadMat(0).getTrans();
	Quat rot = PluginHelper::getHeadMat(0).getRotate();

	sprintf(_stringOut, "%8f %8f %8f %8f %8f %8f %8f \n",pos.x(),pos.y(),pos.z(),rot.x(),rot.y(),rot.z(),rot.w());

	//std::cerr << _stringOut;
	_file << _stringOut;

	pos = PluginHelper::getHandMat(0).getTrans();
	rot = PluginHelper::getHandMat(0).getRotate();

	sprintf(_stringOut, "%8f %8f %8f %8f %8f %8f %8f \n",pos.x(),pos.y(),pos.z(),rot.x(),rot.y(),rot.z(),rot.w());

	//std::cerr << _stringOut;
	_file << _stringOut;
    }
}

