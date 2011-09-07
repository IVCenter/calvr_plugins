#include "MenuBasics.h"
#include "ComputeBBVisitor.h"

#include <kernel/Navigation.h>
#include <kernel/SceneManager.h>
#include <kernel/CVRViewer.h>
#include <kernel/ScreenConfig.h>
#include <kernel/ComController.h>
#include <input/TrackingManager.h>
#include <menu/MenuSystem.h>
#include <config/ConfigManager.h>

#include <PluginMessageType.h>

#include <osg/Matrix>

#include <algorithm>
#include <iostream>

CVRPLUGIN(MenuBasics)

using namespace cvr;

MenuBasics::MenuBasics()
{
    changeSep = false;
}

MenuBasics::~MenuBasics()
{
}

bool MenuBasics::init()
{
    moveworld = new MenuCheckbox("Move World", false);
    moveworld->setCallback(this);
    scale = new MenuCheckbox("Scale",false);
    scale->setCallback(this);
    drive = new MenuCheckbox("Drive",true);
    drive->setCallback(this);
    fly = new MenuCheckbox("Fly",false);
    fly->setCallback(this);

    activeMode = drive;

    MenuSystem::instance()->addMenuItem(moveworld);
    MenuSystem::instance()->addMenuItem(scale);
    MenuSystem::instance()->addMenuItem(drive);
    MenuSystem::instance()->addMenuItem(fly);

    viewall = new MenuButton("View All");
    viewall->setCallback(this);

    resetview = new MenuButton("Reset View");
    resetview->setCallback(this);

    MenuSystem::instance()->addMenuItem(viewall);
    MenuSystem::instance()->addMenuItem(resetview);

    stopHeadTracking = new MenuCheckbox("Stop Head Tracking", !cvr::TrackingManager::instance()->getUpdateHeadTracking());
    stopHeadTracking->setCallback(this);

    MenuSystem::instance()->addMenuItem(stopHeadTracking);

    bool state = ConfigManager::getBool("value","EyeSeparation",true);

    eyeSeparation = new MenuCheckbox("Eye Separation", state);
    eyeSeparation->setCallback(this);

    MenuSystem::instance()->addMenuItem(eyeSeparation);

    sceneSize = ConfigManager::getFloat("SceneSize",1500.0);

    return true;
}

void MenuBasics::menuCallback(MenuItem * item)
{
    if(item == moveworld)
    {
	if(activeMode != item)
	{
	    activeMode->setValue(false);
	}
	activeMode = moveworld;
	moveworld->setValue(true);
	Navigation::instance()->setPrimaryButtonMode(MOVE_WORLD);
    }
    else if(item == scale)
    {
	if(activeMode != item)
	{
	    activeMode->setValue(false);
	}
	activeMode = scale;
	scale->setValue(true);
	Navigation::instance()->setPrimaryButtonMode(SCALE);
    }
    else if(item == drive)
    {
	if(activeMode != item)
	{
	    activeMode->setValue(false);
	}
	activeMode = drive;
	drive->setValue(true);
	Navigation::instance()->setPrimaryButtonMode(DRIVE);
    }
    else if(item == fly)
    {
	if(activeMode != item)
	{
	    activeMode->setValue(false);
	}
	activeMode = fly;
	fly->setValue(true);
	Navigation::instance()->setPrimaryButtonMode(FLY);
    }
    else if(item == resetview)
    {
	osg::Matrix m;
	SceneManager::instance()->setObjectMatrix(m);
	SceneManager::instance()->setObjectScale(1.0);
    }
    else if(item == viewall)
    {
	osg::Matrix m;
	m.makeIdentity();
	float scale;
	osg::Vec3 center;
	if(ComController::instance()->isMaster())
	{
	    ComputeBBVisitor * compBB = new ComputeBBVisitor(m);
	    SceneManager::instance()->getObjectsRoot()->accept(*compBB);
	    osg::BoundingBox bb = compBB->getBound();
	    delete compBB;

	    center = bb.center();
	    osg::Vec3 distance = (bb.corner(0) - bb.center());

	    float maxSide = std::max(fabs(distance[0]),fabs(distance[1]));
	    maxSide = std::max(maxSide,(float)fabs(distance[2]));
	    maxSide = maxSide * 2.0;
	    maxSide = std::max(maxSide,1.0f);

	    //std::cerr << "Max side " << maxSide << std::endl;

	    scale = sceneSize / maxSide;
	    center = center * scale;

	    std::cerr << "Scale set to " << scale << std::endl;

	    std::cerr << "Center x: " << center.x() << " y: " << center.y() << " z: " << center.z() << std::endl;
	    ComController::instance()->sendSlaves(&scale,sizeof(float));
	    ComController::instance()->sendSlaves(&center,sizeof(osg::Vec3));
	}
	else
	{
	    ComController::instance()->readMaster(&scale,sizeof(float));
	    ComController::instance()->readMaster(&center,sizeof(osg::Vec3));
	}

	SceneManager::instance()->setObjectScale(scale);
	//m = SceneManager::instance()->getObjectTransform()->getMatrix();
	m.makeTranslate(-center);
	SceneManager::instance()->setObjectMatrix(m);
    }
    else if(item == stopHeadTracking)
    {
	TrackingManager::instance()->setUpdateHeadTracking(!stopHeadTracking->getValue());
    }
    else if(item == eyeSeparation)
    {
	float target;
	if(eyeSeparation->getValue())
	{
	    target = 1.0;
	}
	else
	{
	    target = 0.0;
	}

	sepStep = (target - ScreenConfig::instance()->getEyeSeparationMultiplier()) / 40.0;
	changeSep = true;
    }
}

void MenuBasics::preFrame()
{
    if(changeSep)
    {
	float mult = ScreenConfig::instance()->getEyeSeparationMultiplier() + sepStep;
	if(eyeSeparation->getValue())
	{
	    if(mult >= 1.0)
	    {
		mult = 1.0;
		changeSep = false;
	    }
	}
	else
	{
	    if(mult <= 0.0)
	    {
		mult = 0.0;
		changeSep = false;
	    }
	}
	ScreenConfig::instance()->setEyeSeparationMultiplier(mult);
    }
}

void MenuBasics::message(int type, char * & data, bool)
{
    MenuMessageType mt = (MenuMessageType) type;
    switch(mt)
    {
        case MB_HEAD_TRACKING:
        {
            bool b = *((bool*)data);
            if(b == !stopHeadTracking->getValue())
            {
                return;
            }
            stopHeadTracking->setValue(!b);
            menuCallback(stopHeadTracking);
            break;
        }
        case MB_STEREO:
        {
            bool b = *((bool*)data);
            if(b == eyeSeparation->getValue())
            {
                return;
            }
            eyeSeparation->setValue(b);
            menuCallback(eyeSeparation);
            break;
        }
        default:
            break;
    }
}
