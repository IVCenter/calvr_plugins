#include <cvrUtil/AndroidStdio.h>
#include <cvrMenu/MenuSystem.h>
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

    basis_renderer = new basisRender;
    _root->addChild(basis_renderer->createBasicRenderer());
    dcm_renderer = new dcmRenderer;
    _root->addChild(dcm_renderer->createDrawableNode());
    dcm_renderer->setNodeMask(0);
    return true;
}
void VolumeViewer::initMenuButtons(){
    _pointCB = new MenuCheckbox("Show PointCloud", true);
    _pointCB->setCallback(this);
    _mainMenu->addItem(_pointCB);

    _planeCB = new MenuCheckbox("Show Plane", true);
    _planeCB->setCallback(this);
    _mainMenu->addItem(_planeCB);
}
void VolumeViewer::menuCallback(cvr::MenuItem *item) {
    if(item == _pointCB)
        basis_renderer->setPointCloudVisiable(_pointCB->getValue());
    else if(item == _planeCB)
        basis_renderer->setPlaneVisiable(_planeCB->getValue());
}

void VolumeViewer::postFrame() {
    basis_renderer->updateOnFrame();
    osg::Matrixf dcm_modelMat;
    if(ARCoreManager::instance()->getLatestHitAnchorModelMat(dcm_modelMat, true)){
        if(!_dcm_initialized){
            _dcm_initialized = true;
            dcm_renderer->setNodeMask(0xFFFFF);
        }
        dcm_renderer->setPosition(dcm_modelMat);
    }


        dcm_renderer->updateOnFrame();
}

bool VolumeViewer::processEvent(cvr::InteractionEvent * event){
    AndroidInteractionEvent * aie = event->asAndroidEvent();
    osg::Vec2f touchPos = osg::Vec2f(aie->getX(), aie->getY());
    if(aie->getInteraction()==BUTTON_DOUBLE_CLICK){
        ARCoreManager::instance()->updatePlaneHittest(touchPos.x(), touchPos.y());
        return true;
    }
    return false;
}