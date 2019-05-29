#include <cvrUtil/AndroidStdio.h>
#include <cvrMenu/MenuSystem.h>
#include <osg/ShapeDrawable>
#include <cvrUtil/AndroidHelper.h>
#include <cvrKernel/PluginHelper.h>
#include "VolumeViewer.h"


using namespace osg;
using namespace cvr;

bool VolumeViewer::init() {
    // --------------- create the menu ---------------
    _mainMenu = new SubMenu("Helmsley", "VolumeViewer");
    _mainMenu->setCallback(this);
    MenuSystem::instance()->addMenuItem(_mainMenu);
    initMenuButtons();

    _root = new osg::Group;
    _rootSO = new SceneObject("HelmsleyRoot", false, false, false, false, false);
    _rootSO->addChild(_root);
    PluginHelper::registerSceneObject(_rootSO, "GlesDrawablesPlugin");
    _rootSO->dirtyBounds();
    _rootSO->attachToScene();

    dcm_renderer = new dcmRenderer;
    basis_renderer = new basisRender;
    _root->addChild(dcm_renderer->createDrawableNode());
    _root->addChild(basis_renderer->createBasicRenderer());

    return true;
}
void VolumeViewer::initMenuButtons(){}
void VolumeViewer::menuCallback(cvr::MenuItem *item) {}

void VolumeViewer::postFrame() {
    dcm_renderer->updateOnFrame();
    basis_renderer->updateOnFrame();
}

bool VolumeViewer::processEvent(cvr::InteractionEvent * event){
    return false;
}