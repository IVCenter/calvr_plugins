#include "MultiViewSimulator.h"

#include <iostream>
#include <cvrInput/TrackingManager.h>
#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/InteractionManager.h>
#include <cvrKernel/PluginHelper.h>
#include <cvrKernel/Screens/Screens/ScreenMultiViewer2.h>
#include <cvrMenu/MenuSystem.h>

#include <osg/ShapeDrawable>

CVRPLUGIN(MultiViewSimulator)

using namespace cvr;

MultiViewSimulator::MultiViewSimulator()
{
    std::cerr << "MultiViewSimulator created." << std::endl;
}

MultiViewSimulator::~MultiViewSimulator()
{
    std::cerr << "MultiViewSimulator destroyed." << std::endl;
    delete startSim;
    delete stopSim;
    delete resetSim;
    delete stepSim;
    delete setHead1to0;
    delete delaySim;
    delete headMenu;
    delete linearFunc;
    delete gaussianFunc;
    delete contributionMenu;
    delete zoneRowQuantity;
    delete zoneColumnQuantity;
    delete autoAdjust;
    delete autoAdjustTarget;
    delete autoAdjustOffset;
    delete zoneMenu;
    delete multipleUsers;
    delete mvsMenu;
}

bool MultiViewSimulator::init()
{
    std::cerr << "MultiViewSimulator init()." << std::endl;

/*** Test Scene ***
    osg::ref_ptr<osg::Box> box = new osg::Box(osg::Vec3(0,0,1),1024,2,768);
    osg::ref_ptr<osg::ShapeDrawable> drawable = new osg::ShapeDrawable(box);
    osg::ref_ptr<osg::Geode> geode = new osg::Geode();
    geode->addDrawable(drawable.get());
    PluginHelper::getObjectsRoot()->addChild(geode.get());
/*****************
    if (ConfigManager::getBool("Plugin.MultiViewSimulator.Head0",false))
    {
        osg::ref_ptr<osg::Cone> view0 = new osg::Cone(osg::Vec3(0,0,0),3,10);
        osg::ref_ptr<osg::ShapeDrawable> drawable0 = new osg::ShapeDrawable(view0);
        osg::ref_ptr<osg::Geode> geode0 = new osg::Geode();
        geode0->addDrawable(drawable0.get());
        osg::ref_ptr<osg::MatrixTransform> pointTransform0 = new osg::MatrixTransform();
        pointTransform0->addChild(geode0.get());
        pointTransform0->setMatrix(osg::Matrix::rotate(osg::Vec3(0,0,1),osg::Vec3(0,1,0)));
        viewTransform0 = new osg::MatrixTransform();
        viewTransform0->addChild(pointTransform0.get());
        PluginHelper::getScene()->addChild(viewTransform0.get());
    }

//    if (ConfigManager::getBool("Plugin.MultiViewSimulator.Head1",true))
    {
        osg::ref_ptr<osg::Cone> view1 = new osg::Cone(osg::Vec3(0,0,0),30,10);
        osg::ref_ptr<osg::ShapeDrawable> drawable1 = new osg::ShapeDrawable(view1);
        osg::ref_ptr<osg::Geode> geode1 = new osg::Geode();
        geode1->addDrawable(drawable1.get());
        osg::ref_ptr<osg::MatrixTransform> pointTransform1 = new osg::MatrixTransform();
        pointTransform1->addChild(geode1.get());
        pointTransform1->setMatrix(osg::Matrix::rotate(osg::Vec3(0,0,1),osg::Vec3(0,1,0)));
        viewTransform1 = new osg::MatrixTransform();
        viewTransform1->addChild(pointTransform1.get());
        PluginHelper::getScene()->addChild(viewTransform1.get());
    }
/*** End Scene ***/

    /*** Menu Setup ***/
    mvsMenu = new SubMenu("MultiViewSimulator", "MultiViewSimulator");
    mvsMenu->setCallback(this);

    multipleUsers = new MenuCheckbox("Multiple Users",
            ScreenMultiViewer2::getMultipleUsers());
    multipleUsers->setCallback(this);

    mvsMenu->addItem(multipleUsers);

    headMenu = new SubMenu("Head Control", "Head Control");
    headMenu->setCallback(this);

    startSim = new MenuButton("Start Simulation");
    startSim->setCallback(this);

    stopSim = new MenuButton("Stop Simulation");
    stopSim->setCallback(this);

    resetSim = new MenuButton("Reset Simulation");
    resetSim->setCallback(this);

    stepSim = new MenuButton("Step Simulation Forward");
    stepSim->setCallback(this);

    setHead1to0 = new MenuButton("Set Head1 = Head0");
    setHead1to0->setCallback(this);

    _delay = 0.025;
    delaySim = new MenuRangeValue("Simulation Delay (seconds)", 0.0,5.0,_delay,.01); 
    delaySim->setCallback(this);

    headMenu->addItem(startSim);
    headMenu->addItem(stopSim);
    headMenu->addItem(resetSim);
    headMenu->addItem(stepSim);
    headMenu->addItem(setHead1to0);
    headMenu->addItem(delaySim);
    mvsMenu->addItem(headMenu);

    contributionMenu = new SubMenu("Contribution Control", "Contribution Control");
    contributionMenu->setCallback(this);

    linearFunc = new MenuCheckbox("Linear Contribution Balancing", false);
    linearFunc->setCallback(this);

    gaussianFunc = new MenuCheckbox("Gaussian Contribution Balancing", true);
    gaussianFunc->setCallback(this);

    orientation3d = new MenuCheckbox("3D Orientation Contribution Balancing",
            ScreenMultiViewer2::getOrientation3d());
    orientation3d->setCallback(this);

    contributionMenu->addItem(linearFunc);
    contributionMenu->addItem(gaussianFunc);
    contributionMenu->addItem(orientation3d);
    mvsMenu->addItem(contributionMenu);

    zoneMenu = new SubMenu("Zone Control", "Zone Control");
    zoneMenu->setCallback(this);

    autoAdjust = new MenuCheckbox("AutoAdjust Zones for FPS",
                    ScreenMultiViewer2::getAutoAdjust());
    autoAdjust->setCallback(this);

    zoneRowQuantity = new MenuRangeValue("Zone Row Quantity", 1,
                    ScreenMultiViewer2::getMaxZoneRows(),
                    ScreenMultiViewer2::getZoneRows(), 1);
    zoneRowQuantity->setCallback(this);

    zoneColumnQuantity = new MenuRangeValue("Zone Column Quantity", 1,
                    ScreenMultiViewer2::getMaxZoneColumns(),
                    ScreenMultiViewer2::getZoneColumns(), 1);
    zoneColumnQuantity->setCallback(this);

    autoAdjustTarget = new MenuRangeValue("AutoAdjust FPS Target", 1, 70,
                    ScreenMultiViewer2::getAutoAdjustTarget(), 1);
    autoAdjustTarget->setCallback(this);

    autoAdjustOffset = new MenuRangeValue("AutoAdjust FPS Offset", 0, 10,
                    ScreenMultiViewer2::getAutoAdjustOffset(), 1);
    autoAdjustOffset->setCallback(this);

    zoneMenu->addItem(autoAdjust);
    zoneMenu->addItem(autoAdjustTarget);
    zoneMenu->addItem(autoAdjustOffset);
    mvsMenu->addItem(zoneMenu);

    MenuSystem::instance()->addMenuItem(mvsMenu);
    /*** End Menu Setup ***/

    _run = false;
    _event = 0;

    if (ConfigManager::getBool("Plugin.MultiViewSimulator.Head0",false))
    {
        ScreenMultiViewer2::headMat[0] = new osg::Matrix();
        head0 = ScreenMultiViewer2::headMat[0];

        int rotate = (int) ConfigManager::getFloat("rotate","Plugin.MultiViewSimulator.Head0",0);
        _event = rotate;
        head0->makeRotate(rotate*M_PI/180.0,osg::Vec3(0,0,1));
        head0->setTrans(osg::Vec3(ConfigManager::getFloat("x","Plugin.MultiViewSimulator.Head0",0),
                                ConfigManager::getFloat("y","Plugin.MultiViewSimulator.Head0",-1500),
                                ConfigManager::getFloat("z","Plugin.MultiViewSimulator.Head0",0)));
    }

    if (ConfigManager::getBool("Plugin.MultiViewSimulator.Head1",true))
    {
        ScreenMultiViewer2::headMat[1] = new osg::Matrix();
        head1 = ScreenMultiViewer2::headMat[1];

        int rotate = (int) ConfigManager::getFloat("rotate","Plugin.MultiViewSimulator.Head1",0);
        _event = rotate;
        head1->makeRotate(rotate*M_PI/180.0,osg::Vec3(0,0,1));
        head1->setTrans(osg::Vec3(ConfigManager::getFloat("x","Plugin.MultiViewSimulator.Head1",1000),
                                ConfigManager::getFloat("y","Plugin.MultiViewSimulator.Head1",-1000),
                                ConfigManager::getFloat("z","Plugin.MultiViewSimulator.Head0",0)));
    }

    return true;
}

void MultiViewSimulator::menuCallback(MenuItem * item)
{
    if (item == multipleUsers)
    {
        ScreenMultiViewer2::setMultipleUsers(multipleUsers->getValue());
    }
    else if (item == startSim)
    {
        _run = true;
    }
    else if (item == stopSim)
    {
        _run = false;
    }
    else if (item == resetSim)
    {
        _run = false;
        _event = 0;
    }
    else if (item == stepSim)
    {
        stepEvent();
    }
    else if (item == setHead1to0)
    {
        if (head1 != NULL)
        {
            if (head0 != NULL)
                *head1 = *head0;
            else
                *head1 = TrackingManager::instance()->getHeadMat(0);
            
            if (viewTransform1)
                viewTransform1->setMatrix(*head1);
        }
    }
    else if (item == delaySim)
    {
        _delay = delaySim->getValue();
    }
    else if (item == linearFunc)
    {
        ScreenMultiViewer2::setSetContributionFunc(0);
        linearFunc->setValue(true);
        gaussianFunc->setValue(false);
    }
    else if (item == gaussianFunc)
    {
        ScreenMultiViewer2::setSetContributionFunc(1);
        linearFunc->setValue(false);
        gaussianFunc->setValue(true);
    }
    else if (item == orientation3d)
    {
        ScreenMultiViewer2::setOrientation3d(orientation3d->getValue());
    }
    else if (item == autoAdjust)
    {
        bool adjust = autoAdjust->getValue();
        ScreenMultiViewer2::setAutoAdjust(adjust);

        if (adjust)
        {
            zoneMenu->addItem(autoAdjustTarget);
            zoneMenu->addItem(autoAdjustOffset);
            zoneMenu->removeItem(zoneRowQuantity);
            zoneMenu->removeItem(zoneColumnQuantity);
        }
        else
        {
            zoneMenu->removeItem(autoAdjustTarget);
            zoneMenu->removeItem(autoAdjustOffset);
            zoneMenu->addItem(zoneRowQuantity);
            zoneMenu->addItem(zoneColumnQuantity);
        }
    }
    else if (item == zoneRowQuantity)
    {
        ScreenMultiViewer2::setZoneRows((int)(zoneRowQuantity->getValue()));
    }
    else if (item == zoneColumnQuantity)
    {
        ScreenMultiViewer2::setZoneColumns((int)(zoneColumnQuantity->getValue()));
    }
    else if (item == autoAdjustTarget)
    {
        ScreenMultiViewer2::setAutoAdjustTarget(autoAdjustTarget->getValue());
    }
    else if (item == autoAdjustOffset)
    {
        ScreenMultiViewer2::setAutoAdjustOffset(autoAdjustOffset->getValue());
    }
}

void MultiViewSimulator::preFrame()
{
    static double lastRun = CVRViewer::instance()->getFrameStartTime();
    double delayed = CVRViewer::instance()->getFrameStartTime() - lastRun;
    if (_run && delayed >= _delay)
    {
        lastRun = CVRViewer::instance()->getFrameStartTime();
        stepEvent();
    }
}

void MultiViewSimulator::stepEvent()
{
    if (viewTransform0 != NULL)
    {
        if (head0 != NULL)
            viewTransform0->setMatrix(*head0);
        else
            viewTransform0->setMatrix(TrackingManager::instance()->getHeadMat(0));
    }
    if (head1 != NULL)
    {
        static int sim = ConfigManager::getInt("sim","Plugin.MultiViewSimulator.Head1",0);

        osg::Vec3 pos = head1->getTrans();
        switch (sim)
        {
        case 0:
            head1->makeRotate(_event*M_PI/180.0,osg::Vec3(0,0,1));
            head1->setTrans(pos);
            break;
        case 1:
            osg::Vec3 move = osg::Vec3((_event < 90 || _event >= 270 ? 30 : -30),0,0);
            head1->setTrans(pos+move);
            break;
        }

        if (viewTransform1 != NULL)
            viewTransform1->setMatrix(*head1);
    }
    else
    {
        if (viewTransform1 != NULL)
            viewTransform1->setMatrix(TrackingManager::instance()->getHeadMat(1));
    }

    if (++_event == 360)
        _event = 0;
}
