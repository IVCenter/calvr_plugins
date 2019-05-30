#include <cvrUtil/AndroidStdio.h>
#include <cvrMenu/MenuSystem.h>
#include <cvrUtil/AndroidHelper.h>
#include <cvrKernel/PluginHelper.h>
#include <cvrConfig/ConfigManager.h>
#include "VolumeViewer.h"


using namespace osg;
using namespace cvr;

bool VolumeViewer::init() {
    // --------------- create the menu ---------------
    _mainMenu = new SubMenu("Helmsley", "VolumeViewer");
    _mainMenu->setCallback(this);
    MenuSystem::instance()->addMenuItem(_mainMenu);

    _tuneMenu = new SubMenu("TuneMenu", "Terms to Tune");
    _tuneMenu->setCallback(this);
    _mainMenu->addItem(_tuneMenu);

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

    _cutCB = new MenuCheckbox("Cut Cube", false);
    _cutCB->setCallback(this);
    _mainMenu->addItem(_cutCB);

    _tuneCBs.push_back(new MenuCheckbox("SampleStep", false));
    _tuneCBs.push_back(new MenuCheckbox("Threshold", false));
    _tuneCBs.push_back(new MenuCheckbox("Brightness", false));

    for(auto cb: _tuneCBs){
        cb->setCallback(this);
        _tuneMenu->addItem(cb);
    }

}
void VolumeViewer::menuCallback(cvr::MenuItem *item) {
    if(item == _pointCB)
        basis_renderer->setPointCloudVisiable(_pointCB->getValue());
    else if(item == _planeCB)
        basis_renderer->setPlaneVisiable(_planeCB->getValue());

    auto got = std::find(_tuneCBs.begin(), _tuneCBs.end(), item);
    if( got != _tuneCBs.end() ){
        current_tune_id = std::distance(_tuneCBs.begin(), got);
        for(int i=0;i<_tuneCBs.size();i++){
            if(i != current_tune_id)
                _tuneCBs[i]->setValue(false);
        }
        if(!_tuneCBs[current_tune_id]->getValue())
            current_tune_id = -1;
    }
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

    //start to check single finger interaction
    if(aie->getTouchType() != cvr::LEFT)
        return false;

    if(aie->getInteraction() == MOVE){
        /*if(!_cutCB->getValue()) return false;

        TrackingManager::instance()->getScreenToClientPos(touchPos);
        dcm_renderer->setCuttingPlane(touchPos.x() * 0.5f + 0.5f); // 0-1
*/
        if(current_tune_id == -1) return false;
        TrackingManager::instance()->getScreenToClientPos(touchPos);
        float percent = touchPos.x() * 0.5f + 0.5f;

        dcm_renderer->setTuneParameter(current_tune_id, MAX_VALUE_TUNE[current_tune_id] * percent);
        return true;
    }
    if(aie->getInteraction() == BUTTON_DOWN){
       // _mPreviousPos = touchPos;
        return true;
    }

    return false;
}