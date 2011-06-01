#include "MVSim.h"

#include <time.h>
#include <sys/types.h>
#include <dirent.h>

#include <iostream>
#include <input/TrackingManager.h>
#include <config/ConfigManager.h>
#include <kernel/ComController.h>
#include <kernel/InteractionManager.h>
#include <kernel/NodeMask.h>
#include <kernel/PluginHelper.h>
#include <kernel/ScreenConfig.h>
#include <menu/MenuSystem.h>

#include <osg/Config>
#include <osg/ShapeDrawable>
#include <osgDB/ReadFile>
#include <osgDB/WriteFile>

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
    delete delaySim;
    delete scene1;
    delete sceneMenu;

    // clear current lists of heads
    std::map<cvr::SubMenu *,osg::Matrix *>::iterator it;
    cvr::MenuItem * item;
    while (!headMats.empty())
    {
        it = headMats.begin();
        while (it->first->getNumChildren() > 0)
        {
            item = it->first->getChild(0);
            it->first->removeItem(item);
            delete item;
        }

        setHeadMenu->removeItem(it->first);
        delete it->first;
        delete it->second;
        headMats.erase(it);
    }

    delete setHeadMenu;
    delete saveHeads;
    delete loadHeads;
    delete showDiagramBox;
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

    _delay = 0.025;
    delaySim = new MenuRangeValue("Simulation Delay (seconds)", 0.0,5.0,_delay,.01); 
    delaySim->setCallback(this);

    sceneMenu = new SubMenu("Scenes", "Scenes");
    sceneMenu->setCallback(this);

    scene1 = new MenuButton("Central Sphere");
    scene1->setCallback(this);

    sceneMenu->addItem(scene1);

    setHeadMenu = new SubMenu("Head Positions", "Heads to Set");
    setHeadMenu->setCallback(this);

    saveHeads = new MenuButton("Save Current Head Positions");
    saveHeads->setCallback(this);

    loadHeads = new MenuButton("Load Head Positions");
    loadHeads->setCallback(this);

    setHeadMenu->addItem(saveHeads);
    setHeadMenu->addItem(loadHeads);

    showDiagramBox = new MenuCheckbox("Show Diagram", ComController::instance()->isMaster());
    showDiagramBox->setCallback(this);

    mvsMenu->addItem(startSim);
    mvsMenu->addItem(stopSim);
    mvsMenu->addItem(resetSim);
    mvsMenu->addItem(stepSim);
    mvsMenu->addItem(delaySim);
    mvsMenu->addItem(sceneMenu);
    mvsMenu->addItem(setHeadMenu);
    mvsMenu->addItem(showDiagramBox);

    MenuSystem::instance()->addMenuItem(mvsMenu);
    /*** End Menu Setup ***/

    scene1switch = NULL;
    diagram = NULL;

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

    showDiagram(ComController::instance()->isMaster());

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
    else if (item == delaySim)
    {
        _delay = delaySim->getValue();
    }
    else if (item == scene1)
    {
        static bool enabled = false;
        // First time? Create the scene
        if (scene1switch == NULL)
        {
            scene1switch = new osg::Switch();
            osg::ref_ptr<osg::Sphere> sphere = new osg::Sphere(osg::Vec3(0,0,0),10);
            osg::ref_ptr<osg::ShapeDrawable> drawable = new osg::ShapeDrawable(sphere);
            osg::ref_ptr<osg::Geode> geode = new osg::Geode();
            geode->addDrawable(drawable.get());
            scene1switch->addChild(geode,enabled);
            PluginHelper::getObjectsRoot()->addChild(scene1switch);
        }

        if (enabled)
            scene1switch->setAllChildrenOff();
        else
            scene1switch->setAllChildrenOn();
        enabled = !enabled;
    }
    else if (item == saveHeads)
    {
        saveCurrentHeadMatrices();
    }
    else if (item == loadHeads)
    {
        // clear current lists of heads
        std::map<cvr::SubMenu *,osg::Matrix *>::iterator it;
        cvr::MenuItem * item;
        while (!headMats.empty())
        {
            it = headMats.begin();
            while (it->first->getNumChildren() > 0)
            {
                item = it->first->getChild(0);
                it->first->removeItem(item);
                delete item;
            }

            setHeadMenu->removeItem(it->first);
            delete it->first;
            delete it->second;
            headMats.erase(it);
        }

        // load head matrices from file and store them as neccessary
        loadHeadMatrices();
    }
    else if (item == showDiagramBox)
    {
        showDiagram(showDiagramBox->getValue());
    }
    else
    {
        std::map<cvr::SubMenu*,osg::Matrix*>::iterator it;
        for (it = headMats.begin(); it != headMats.end(); it++)
        {
            for (int i = 0; i < it->first->getNumChildren(); i++)
            {
                if (item == it->first->getChild(i))
                {
                    cvr::MenuButton * button = dynamic_cast<cvr::MenuButton *> (item);

                    if (button == NULL)
                    {
                        std::cerr<<"Error: Non-MenuButton located in MVSim under matrix sub-menu.\n";
                        return;
                    }

                    if (button->getText() == "Head 0")
                    {
                        if (head0 != NULL)
                        {
                            head0->set(*it->second);
                            _screenMVSim->setSimulatedHeadMatrix(0,head0);
                        }
                        else
                            std::cerr<<"Warning: Load attempted for unsimiluated head (0).\n";
                        return;
                    }

                    else if (button->getText() == "Head 1")
                    {
                        if (head1 != NULL)
                        {
                            head1->set(*it->second);
                            _screenMVSim->setSimulatedHeadMatrix(1,head1);
                        }
                        else
                            std::cerr<<"Warning: Load attempted for unsimiluated head (1).\n";
                        return;
                    }
                }
            }
        }
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

    // update the heads in the diagram
    if(diagram != NULL && showDiagramBox->getValue())
    {
        osg::Matrix mat0 = _screenMVSim->getCurrentHeadMatrix(0);
        osg::Matrix mat1 = _screenMVSim->getCurrentHeadMatrix(1);

        osg::Vec3 pos0 = mat0.getTrans();
        mat0.setTrans(pos0.x(),pos0.y(),0);

        osg::Vec3 pos1 = mat1.getTrans();
        mat1.setTrans(pos1.x(),pos1.y(),0);

        cone0->setMatrix(mat0);
        cone1->setMatrix(mat1);
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

void MVSim::saveCurrentHeadMatrices()
{
    std::string dir = ConfigManager::getEntry("Plugin.MVSim.HeadMatrixDir");
    if (dir == "")
    {
        std::cerr<<"Error: No head matrix directory given by config file.\n";
        return;
    }

    time_t rawtime;
    struct tm * timeinfo;
    char filename0 [512];
    char filename1 [512];
    time(&rawtime);
    timeinfo = localtime(&rawtime);

    std::string head0format = dir+"%Y_%m_%d_%H_%M_%S_head";
    std::string head1format = head0format;
    head0format += "0.osg";
    head1format += "1.osg";
    strftime(filename0,512,head0format.c_str(),timeinfo);
    strftime(filename1,512,head1format.c_str(),timeinfo);
    std::string f0 = filename0;
    std::string f1 = filename1;

    // Using MatrixTransforms since writing RefMatrix's is buggy (empty files)
    osg::MatrixTransform *h0 = new osg::MatrixTransform(_screenMVSim->getCurrentHeadMatrix(0));
    osg::MatrixTransform *h1 = new osg::MatrixTransform(_screenMVSim->getCurrentHeadMatrix(1));

    std::cerr<<"Saving head0...";
    if (osgDB::writeObjectFile(*h0,f0))
        std::cerr<<" success.\n";
    else
        std::cerr<<" failure.\n";

    std::cerr<<"Saving head1...";
    if (osgDB::writeObjectFile(*h1,f1))
        std::cerr<<" success.\n";
    else
        std::cerr<<" failure.\n";
}

void MVSim::loadHeadMatrices()
{
    std::string dir = ConfigManager::getEntry("Plugin.MVSim.HeadMatrixDir");
    if (dir == "")
    {
        std::cerr<<"Error: No head matrix directory given by config file.\n";
        return;
    }
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL) {
        std::cerr<<"Error opening directory("<<dir<<")\n";
        return;
    }

    while ((dirp = readdir(dp)) != NULL) {
        if (dirp->d_name[0] == '.')
            continue;

        std::cerr<<"Going to read "<<dirp->d_name<<"...";
        osg::MatrixTransform * matTrans = dynamic_cast<osg::MatrixTransform *>(osgDB::readNodeFile(dir+dirp->d_name));

        if (matTrans != NULL)
        {
            cvr::SubMenu * menu = new cvr::SubMenu(dirp->d_name,std::string(dirp->d_name));
            menu->setCallback(this);

            cvr::MenuButton * head0 = new cvr::MenuButton("Head 0");
            head0->setCallback(this);
            menu->addItem(head0);
            cvr::MenuButton * head1 = new cvr::MenuButton("Head 1");
            head1->setCallback(this);
            menu->addItem(head1);

            setHeadMenu->addItem(menu);
            headMats[menu] = new osg::Matrix(matTrans->getMatrix());
            std::cerr<<" success.\n";
        }
        else
            std::cerr<<" failure.\n";
    }
    closedir(dp);
}

void MVSim::showDiagram(bool show)
{
    if (ComController::instance()->isMaster())
    {
        static unsigned int objMask = PluginHelper::getObjectsRoot()->getNodeMask();

        ScreenMVMaster * masterScreen = dynamic_cast<ScreenMVMaster *>(_screenMVSim);

        if (masterScreen == NULL)
        {
            std::cerr<<"Cannot show diagram without a ScreenMVMaster.";
            return;
        }

        if (diagram == NULL)
            setupCaveDiagram(masterScreen);

        if (show)
        {
            objMask = PluginHelper::getObjectsRoot()->getNodeMask();
            PluginHelper::getObjectsRoot()->setNodeMask(0);
            PluginHelper::getScene()->addChild(diagram);
        }
        else
        {
            PluginHelper::getObjectsRoot()->setNodeMask(objMask);
            PluginHelper::getScene()->removeChild(diagram);
        }

        masterScreen->showDiagram(show);
    }
}

void MVSim::setupCaveDiagram(ScreenMVMaster * masterScreen)
{
    diagram = new osg::Group();

    float rad = 1468;

    // circle around cave
    osg::ref_ptr<osg::Geode> circle = new osg::Geode();
    osg::ref_ptr<osg::Geometry> circGeo = new osg::Geometry();
    const int CIRCLE_SEGMENTS = 40;
    osg::Vec3Array * circVerts = new osg::Vec3Array(CIRCLE_SEGMENTS);    

    for (int a = 0; a < CIRCLE_SEGMENTS; a++)
    {
        float angle = a * 2 * M_PI / CIRCLE_SEGMENTS + M_PI/2;
        (*circVerts)[a].set(rad*cos(angle),0,rad*sin(angle));
    }
    circGeo->setVertexArray(circVerts);

    osg::Vec4Array * circColors = new osg::Vec4Array;
    circColors->push_back(osg::Vec4(0,0,1,1));
    circGeo->setColorArray(circColors);
    circGeo->setColorBinding(osg::Geometry::BIND_OVERALL);

    circGeo->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::LINE_LOOP,0,CIRCLE_SEGMENTS));
    circle->addDrawable(circGeo);
    diagram->addChild(circle);

    // walls of cave
    osg::ref_ptr<osg::Geode> walls = new osg::Geode();
    osg::ref_ptr<osg::Geometry> wallGeo = new osg::Geometry();
    const int WALL_SEGMENTS = 5;
    osg::Vec3Array * wallVerts = new osg::Vec3Array(WALL_SEGMENTS);    

    for (int a = 0; a < WALL_SEGMENTS; a++)
    {
        float angle = a * 2 * M_PI / WALL_SEGMENTS + M_PI/2;
        (*wallVerts)[a].set(rad*cos(angle),0,rad*sin(angle));
    }
    wallGeo->setVertexArray(wallVerts);

    osg::Vec4Array * wallColors = new osg::Vec4Array;
    wallColors->push_back(osg::Vec4(1,1,0,1));
    wallGeo->setColorArray(wallColors);
    wallGeo->setColorBinding(osg::Geometry::BIND_OVERALL);

    wallGeo->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::LINE_LOOP,0,WALL_SEGMENTS));
    walls->addDrawable(wallGeo);
    diagram->addChild(walls);

    osg::ref_ptr<osg::MatrixTransform> headRot = new osg::MatrixTransform();
    headRot->setMatrix(osg::Matrix::rotate(osg::Vec3(0,1,0),osg::Vec3(0,0,1)));
    diagram->addChild(headRot);

    //cone for user 0
    osg::ref_ptr<osg::Geode> head0Cone = new osg::Geode();

    osg::ref_ptr<osg::Geometry> head0Geo = new osg::Geometry();
    head0Geo->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    osg::Vec3Array * cone0Verts = new osg::Vec3Array(3);
    (*cone0Verts)[0].set(0,0,0);
    (*cone0Verts)[1].set(400,1000,0);
    (*cone0Verts)[2].set(-400,1000,0);
    head0Geo->setVertexArray(cone0Verts);
    osg::Vec4Array * cone0Colors = new osg::Vec4Array;
    cone0Colors->push_back(osg::Vec4(1,0,0,1));
    head0Geo->setColorArray(cone0Colors);
    head0Geo->setColorBinding(osg::Geometry::BIND_OVERALL);
    head0Geo->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::LINE_LOOP,0,3));
    head0Cone->addDrawable(head0Geo);

    osg::ref_ptr<osg::Geometry> num0Geo = new osg::Geometry();
    num0Geo->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    osg::Vec3Array * num0Verts = new osg::Vec3Array(8);
    (*num0Verts)[0].set(0,400,0);
    (*num0Verts)[1].set(70,430,0);
    (*num0Verts)[2].set(100,500,0);
    (*num0Verts)[3].set(70,570,0);
    (*num0Verts)[4].set(0,600,0);
    (*num0Verts)[5].set(-70,570,0);
    (*num0Verts)[6].set(-100,500,0);
    (*num0Verts)[7].set(-70,430,0);
    num0Geo->setVertexArray(num0Verts);
    num0Geo->setColorArray(cone0Colors);
    num0Geo->setColorBinding(osg::Geometry::BIND_OVERALL);
    num0Geo->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::LINE_LOOP,0,8));
    head0Cone->addDrawable(num0Geo);

    cone0 = new osg::MatrixTransform();
    cone0->addChild(head0Cone.get());
    headRot->addChild(cone0);

    //cone for user 1
    osg::ref_ptr<osg::Geode> head1Cone = new osg::Geode();
    osg::ref_ptr<osg::Geometry> head1Geo = new osg::Geometry();
    head1Geo->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    osg::Vec3Array * cone1Verts = new osg::Vec3Array(3);
    (*cone1Verts)[0].set(0,0,0);
    (*cone1Verts)[1].set(400,1000,0);
    (*cone1Verts)[2].set(-400,1000,0);
    head1Geo->setVertexArray(cone1Verts);
    osg::Vec4Array * cone1Colors = new osg::Vec4Array;
    cone1Colors->push_back(osg::Vec4(0,1,0,1));
    head1Geo->setColorArray(cone1Colors);
    head1Geo->setColorBinding(osg::Geometry::BIND_OVERALL);
    head1Geo->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::LINE_LOOP,0,3));
    head1Cone->addDrawable(head1Geo);

    osg::ref_ptr<osg::Geometry> num1Geo = new osg::Geometry();
    num1Geo->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    osg::Vec3Array * num1Verts = new osg::Vec3Array(6);
    (*num1Verts)[0].set(0,400,0);
    (*num1Verts)[1].set(0,600,0);
    (*num1Verts)[2].set(-70,400,0);
    (*num1Verts)[3].set(70,400,0);
    (*num1Verts)[4].set(0,600,0);
    (*num1Verts)[5].set(-70,530,0);
    num1Geo->setVertexArray(num1Verts);
    num1Geo->setColorArray(cone1Colors);
    num1Geo->setColorBinding(osg::Geometry::BIND_OVERALL);
    num1Geo->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::LINES,0,6));
    head1Cone->addDrawable(num1Geo);

    cone1 = new osg::MatrixTransform();
    cone1->addChild(head1Cone.get());
    headRot->addChild(cone1);
}
