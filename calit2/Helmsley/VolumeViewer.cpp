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

    renderer = new dcmRenderer;
    _root->addChild(renderer->createDrawableNode());

    return true;
}
void VolumeViewer::initMenuButtons(){}
void VolumeViewer::menuCallback(cvr::MenuItem *item) {}

void VolumeViewer::postFrame() {
    renderer->updateOnFrame();
}

bool VolumeViewer::processEvent(cvr::InteractionEvent * event){
    return false;
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
