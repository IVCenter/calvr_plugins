#include "MVSim.h"

#include <iostream>
#include <input/TrackingManager.h>
#include <config/ConfigManager.h>
#include <kernel/InteractionManager.h>
#include <kernel/PluginHelper.h>
#include <kernel/ScreenConfig.h>
#include <menu/MenuSystem.h>

#include <osg/ShapeDrawable>

CVRPLUGIN(MVSim)

using namespace cvr;

MVSim::MVSim()
{
    std::cerr << "MVSim created." << std::endl;
}

MVSim::~MVSim()
{
    std::cerr << "MVSim destroyed." << std::endl;
    delete startSim;
    delete stopSim;
    delete resetSim;
    delete stepSim;
    delete setHead1to0;
    delete delaySim;
    delete mvsMenu;
}

bool MVSim::init()
{
    std::cerr << "MVSim init()." << std::endl;

/*** Test Scene ***
    osg::ref_ptr<osg::Box> box = new osg::Box(osg::Vec3(0,0,1),1024,2,768);
    osg::ref_ptr<osg::ShapeDrawable> drawable = new osg::ShapeDrawable(box);
    osg::ref_ptr<osg::Geode> geode = new osg::Geode();
    geode->addDrawable(drawable.get());
    PluginHelper::getObjectsRoot()->addChild(geode.get());
/*****************
    if (ConfigManager::getBool("Plugin.MVSim.Head0",false))
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

//    if (ConfigManager::getBool("Plugin.MVSim.Head1",true))
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
    mvsMenu = new SubMenu("MVSim", "MVSim");
    mvsMenu->setCallback(this);

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

    mvsMenu->addItem(startSim);
    mvsMenu->addItem(stopSim);
    mvsMenu->addItem(resetSim);
    mvsMenu->addItem(stepSim);
    mvsMenu->addItem(setHead1to0);
    mvsMenu->addItem(delaySim);

    MenuSystem::instance()->addMenuItem(mvsMenu);
    /*** End Menu Setup ***/

    _run = false;
    _event = 0;

    ScreenConfig * sConfig = ScreenConfig::instance();
    for (int i=0; i < sConfig->getNumScreens(); i++)
    {
        _screenMVSim = dynamic_cast<ScreenMVSimulator *> (sConfig->getScreen(i));
        if (_screenMVSim != NULL)
            break;
    }
    if (_screenMVSim == NULL)
    {
        std::cerr<<"Cannot initialize MVSim without running a ScreenMVSimulator screen.\n";
        return false;
    }

    if (ConfigManager::getBool("Plugin.MVSim.Head0",false))
    {
        head0 = new osg::Matrix();

        int rotate = (int) ConfigManager::getFloat("rotate","Plugin.MVSim.Head0",0);
        _event = rotate;
        head0->makeRotate(rotate*M_PI/180.0,osg::Vec3(0,0,1));
        head0->setTrans(osg::Vec3(ConfigManager::getFloat("x","Plugin.MVSim.Head0",0),
                                ConfigManager::getFloat("y","Plugin.MVSim.Head0",-1500),
                                ConfigManager::getFloat("z","Plugin.MVSim.Head0",0)));

        _screenMVSim->setSimulatedHeadMatrix(0,head0);
    }

    if (ConfigManager::getBool("Plugin.MVSim.Head1",true))
    {
        head1 = new osg::Matrix();

        int rotate = (int) ConfigManager::getFloat("rotate","Plugin.MVSim.Head1",0);
        _event = rotate;
        head1->makeRotate(rotate*M_PI/180.0,osg::Vec3(0,0,1));
        head1->setTrans(osg::Vec3(ConfigManager::getFloat("x","Plugin.MVSim.Head1",1000),
                                ConfigManager::getFloat("y","Plugin.MVSim.Head1",-1000),
                                ConfigManager::getFloat("z","Plugin.MVSim.Head0",0)));

        _screenMVSim->setSimulatedHeadMatrix(1,head1);
    }

    return true;
}

void MVSim::menuCallback(MenuItem * item)
{
    if (item == startSim)
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
            *head1 = _screenMVSim->getCurrentHeadMatrix(0);

            if (viewTransform1)
                viewTransform1->setMatrix(*head1);
        }
    }
    else if (item == delaySim)
    {
        _delay = delaySim->getValue();
    }
}

void MVSim::preFrame()
{
    static double lastRun = CVRViewer::instance()->getFrameStartTime();
    double delayed = CVRViewer::instance()->getFrameStartTime() - lastRun;
    if (_run && delayed >= _delay)
    {
        lastRun = CVRViewer::instance()->getFrameStartTime();
        stepEvent();
    }
}

void MVSim::stepEvent()
{
    if (viewTransform0 != NULL)
    {
        viewTransform0->setMatrix(_screenMVSim->getCurrentHeadMatrix(0));
    }
    if (head1 != NULL)
    {
        static int sim = ConfigManager::getInt("sim","Plugin.MVSim.Head1",0);

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

        _screenMVSim->setSimulatedHeadMatrix(1,head1);
    }
    if (viewTransform1 != NULL)
    {
        viewTransform1->setMatrix(_screenMVSim->getCurrentHeadMatrix(1));
    }

    if (++_event == 360)
        _event = 0;
}
