#include "Plane.h"
using namespace osg;

#include <string>
using std::string;

#include <config/ConfigManager.h>
using namespace cvr;

Node* makePlane()
{
    string planepath = ConfigManager::getEntry("value", "Plugin.AppSwitcher.planepath", "");
    //return osgDB::readNodeFile("planeIcon.obj", NULL);
    return osgDB::readNodeFile(planepath, NULL);
}

