#include <cvrUtil/AndroidStdio.h>
#include <osg/LineWidth>
#include <osgUtil/Tessellator>
#include <osg/Texture>
#include <osg/Texture2D>
#include <cvrUtil/AndroidHelper.h>
#include <osg/ShapeDrawable>
#include <cvrKernel/PluginManager.h>

#include "VolumeViewer"
using namespace osg;
using namespace cvr;

bool VolumeViewer:: tackleHitted(osgUtil::LineSegmentIntersector::Intersection result ){
    return true;
}

void VolumeViewer::initMenuButtons() {

    //_lightButton = new MenuCheckbox("Add Light Source", false);
    //_lightButton->setCallback(this);
    //_mainMenu->addItem(_lightButton);
}

bool VolumeViewer::init() {
    // --------------- create the menu ---------------
    _mainMenu = new SubMenu("GlesDrawable", "GlesDrawable");
    _mainMenu->setCallback(this);
    MenuSystem::instance()->addMenuItem(_mainMenu);
    initMenuButtons();

    _root = new Group;
    _objects = new Group;

    SceneManager::instance()->getSceneRoot()->getOrCreateStateSet()->setMode( GL_LIGHTING, osg::StateAttribute::OFF );

    //bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds
    rootSO= new SceneObject("glesRoot", false, false, false, false, false);
    rootSO->addChild(_root);
    objSO = new SceneObject("testBoundObj", false, false, false, false, true);
    rootSO->addChild(objSO);
    objSO->addChild(_objects);
    objSO->dirtyBounds();

    PluginHelper::registerSceneObject(rootSO, "VolumeViewer Plugin");
    rootSO->dirtyBounds();
    rootSO->attachToScene();

    return true;
}

void VolumeViewer::menuCallback(cvr::MenuItem *item) {}

void VolumeViewer::postFrame() {
    /*_pointcloudDrawable->updateOnFrame();
    cvr::planeMap map = ARCoreManager::instance()->getPlaneMap();
    if(_plane_num < map.size()){
        for(int i= _plane_num; i<map.size();i++){
            planeDrawable * pd = new planeDrawable();
            _root->addChild(pd->createDrawableNode());
            _planeDrawables.push_back(pd);
        }
        _plane_num = map.size();
    }
    auto planeIt = map.begin();
    for(int i=0; i<_plane_num; i++,planeIt++)
        _planeDrawables[i]->updateOnFrame(planeIt->first, planeIt->second);


    Vec3f isPoint;
    if(TrackingManager::instance()->getIsPoint(isPoint)){
        _strokeDrawable->updateOnFrame(isPoint);
        _strokeDrawable->getGLNode()->setNodeMask(0xFFFFFF);
    } else
        _strokeDrawable->getGLNode()->setNodeMask(0x0);


    size_t anchor_num = ARCoreManager::instance()->getAnchorSize();
    if( anchor_num != 0){
        if(_objNum < anchor_num){
            for(int i=_objNum; i<anchor_num; i++){
                Matrixf modelMat;
                if(!ARCoreManager::instance()->getAnchorModelMatrixAt(modelMat, i))
                    break;
                if(_add_light){
                    createDebugSphere(_objects, modelMat);
//
                    osg::Vec3f debug = modelMat.getTrans();

                    osg::Vec4f tmp = osg::Vec4f(debug.x(), debug.z(), -debug.y(),1.0) * (*ARCoreManager::instance()->getViewMatrix());
                    _lightPosition = osg::Vec3f(tmp.x() / tmp.w(), -tmp.z()/tmp.w(), tmp.y()/tmp.w());
                            LOGE("===LIGHT: %f, %f, %f", _lightPosition.x(), _lightPosition.y(), _lightPosition.z());
                    _add_light = false;
                    break;
                }
                switch(last_object_select){
                    case 1:
                        createObject(_objects,"models/andy.obj", "textures/andy.png",
                                     modelMat, ONES_SOURCE);
                        break;
                    case 2:
                        createObject(_objects,"models/andy.obj", "textures/andy.png",
                                     modelMat, ONES_SOURCE);
                        break;
                    case 3:
                        createObject(_objects,"models/andy.obj", "textures/andy.png",
                                     modelMat, SPHERICAL_HARMONICS);
                        break;
                    default:
                        createObject(_objects,"models/andy.obj", "textures/andy.png",
                                     modelMat, ARCORE_CORRECTION);
                }

            }

        }
        _objNum = anchor_num;
    }
//    _quadDrawable->updateOnFrame(ARCoreManager::instance()->getCameraTransformedUVs());
*/

}

bool VolumeViewer::processEvent(cvr::InteractionEvent * event){
    return true;
}


void VolumeViewer::createDebugSphere(osg::Group*parent, Matrixf modelMat){
    Transform objectTrans = new MatrixTransform;
    objectTrans->setMatrix(modelMat);
    parent->addChild(objectTrans);

    osg::ref_ptr<osg::ShapeDrawable> shape = new osg::ShapeDrawable;
    shape->setShape(new osg::Sphere(osg::Vec3f(.0,.0,.0), 0.05f));
    shape->setColor(osg::Vec4f(1.0f,.0f,.0f,1.0f));
    osg::ref_ptr<osg::Geode> node = new osg::Geode;
    Program * program = assetLoader::instance()->createShaderProgramFromFile("shaders/lighting.vert","shaders/lighting.frag");

    osg::StateSet * stateSet = shape->getOrCreateStateSet();
    stateSet->setAttributeAndModes(program);

    stateSet->addUniform( new osg::Uniform("lightDiffuse",
                                           osg::Vec4(0.8f, 0.8f, 0.8f, 1.0f)) );
    stateSet->addUniform( new osg::Uniform("lightSpecular",
                                           osg::Vec4(1.0f, 1.0f, 0.4f, 1.0f)) );
    stateSet->addUniform( new osg::Uniform("shininess", 64.0f) );

    stateSet->addUniform( new osg::Uniform("lightPosition", osg::Vec3(0,0,1)));

    Uniform * baseColor = new osg::Uniform("uBaseColor", osg::Vec4f(1.0f, .0f, .0f, 1.0f));
    stateSet->addUniform(baseColor);

    node->addDrawable(shape.get());
    objectTrans->addChild(node);
}
