#include "PhysxBall.h"
//PhysX
#include <PxPhysicsAPI.h>
#include <extensions/PxDefaultErrorCallback.h>
#include <extensions/PxDefaultAllocator.h>
#include <foundation/Px.h>
#include <extensions/PxShapeExt.h>
#include <foundation/PxMat33.h>
#include <foundation/PxQuat.h>
#include <extensions/PxRigidActorExt.h>
#include <foundation/PxFoundation.h>
#include <osg/ShapeDrawable>
#include <cvrMenu/BoardMenu/BoardMenuGeometry.h>
#include "PhysicsUtils.h"
#include <osg/BlendFunc>

using namespace osg;
using namespace cvr;
using namespace physx;
using namespace osgPhysx;


void UpdateActorCallback::operator()( osg::Node* node, osg::NodeVisitor* nv )
{
    osg::MatrixTransform* mt = (node->asTransform() ? node->asTransform()->asMatrixTransform() : NULL);
    if(update_mat) {
        if (mt && _actor) {
            PxMat44 matrix(_actor->getGlobalPose());
            mt->setMatrix(osgPhysx::toMatrix(matrix));
            //
        }
    }else{
        if (mt && _pd) {
            //Matrix t = glm_to_osg(_pd->_model_mat);
            //mt->setMatrix(t);
            //
        }
    }

    traverse( node, nv );
}

void PhysxBall::preFrame() {
    _phyEngine->update();
    syncPlane();
}

void PhysxBall::initMenuButtons() {
    _bThrowBall = new MenuCheckbox("Throw Ball", false);
    _bThrowBall->setCallback(this);
    _mainMenu->addItem(_bThrowBall);

    _bCreatePlane = new MenuCheckbox("Sync Plane with PhysX", false);
    _bCreatePlane->setCallback(this);
    _mainMenu->addItem(_bCreatePlane);

    _planePButton = new MenuCheckbox("Debug Plane(Polygon)", false);
    _planePButton->setCallback(this);
    _mainMenu->addItem(_planePButton);

    _planeBButton = new MenuCheckbox("Debug Plane(Bounding)", false);
    _planeBButton->setCallback(this);
    _mainMenu->addItem(_planeBButton);

    _forceFactor = new MenuRangeValueCompact("Throw Force",0.01,100.0,0.01,true,4.5);
    _forceFactor->setCallback(this);
    _mainMenu->addItem(_forceFactor);
}
bool PhysxBall::init() {
    // --------------- create the menu ---------------
    _mainMenu = new SubMenu("PhysxBall", "PhysxBall");
    _mainMenu->setCallback(this);

    initMenuButtons();
    MenuSystem::instance()->addMenuItem(_mainMenu);


    //--------------init scene node--------------
    _menu = new Group;
    _scene = new Group;


    //--------------init physx-------------------
    _phyEngine = Engine::instance();
    if(!_phyEngine->init())
        return false;
    _phyEngine->addScene("main");
    _phyEngine->getScene("main")->setBounceThresholdVelocity(0.065 * 9.81);

    //createPlane(_scene, Vec3f(.0f, .0f, -0.5f));
    _uniform_mvps.clear();
    bounding.clear();
    pp_list.clear();
    PxGeo.clear();

    SceneManager::instance()->getSceneRoot()->getOrCreateStateSet()->setMode( GL_LIGHTING, osg::StateAttribute::OFF );


    //bool navigation, bool movable, bool clip, bool contextMenu, bool showBounds
    rootSO= new SceneObject("myPluginRoot", false, false, false, false, false);
    sceneSO= new SceneObject("myPluginScene", false, false, false, false, false);
    menuSo= new SceneObject("myPluginMenu", false, false, false, false, false);

    rootSO->addChild(sceneSO);
    rootSO->addChild(menuSo);
    sceneSO->addChild(_scene);
    menuSo->addChild(_menu);

//    rootSO->addChild(_root);
    PluginHelper::registerSceneObject(rootSO, "PhysxBallSceneObjset");
    rootSO->dirtyBounds();
    rootSO->attachToScene();

    return true;
}

void PhysxBall::menuCallback(cvr::MenuItem *item) {
   if(item == _planeBButton){
        if(_planeBButton->getValue() && !last_state){
            for(auto b : bounding){
                b->setNodeMask(0xFFFFFF);
            }
        }else if (!_planeBButton->getValue() && last_state){
            for(auto b : bounding){
                b->setNodeMask(0);
            }
        }
        last_state = _planeBButton->getValue();
    }
}


bool PhysxBall::processEvent(cvr::InteractionEvent * event){
    AndroidInteractionEvent * aie = event->asAndroidEvent();
    if(aie->getTouchType() != LEFT)
        return false;
    if(aie->getInteraction()== BUTTON_UP){
        throwBall();
        return true;
    }
    return false;
}

void PhysxBall::throwBall() {
    if(_bThrowBall->getValue()) {
        if(!_bFirstPress) {
            createBall(_scene, osg::Vec3(.0f, 0.5, 0.5), 0.014f);
        }else{
            _bFirstPress = false;
        }
    }else{
        _bFirstPress = true;
    }
}

void PhysxBall::createBall(osg::Group* parent,osg::Vec3f pos, float radius) {
    osg::Vec3 eye, at, up;
    _viewer = CVRViewer::instance();
    auto mat = _viewer->getCamera()->getViewMatrix();
    mat.getLookAt( eye, at, up );

    osg::Vec3 viewDir = at - eye;
    viewDir.normalize();

    osg::Vec3 target = eye + viewDir * radius;
    float forceFactor = _forceFactor->getValue();
    viewDir *= forceFactor;

    PxReal density = 1000.0f;
    PxMaterial* mMaterial = _phyEngine->getPhysicsSDK()->createMaterial(10.0,10.0,0.65);

    PxSphereGeometry geometrySphere(radius);
    PxTransform transform(PxVec3(target.x(), target.y(), target.z()), PxQuat(PxIDENTITY()));
    PxRigidDynamic *actor = PxCreateDynamic(* _phyEngine->getPhysicsSDK(),
                                            transform,
                                            geometrySphere,
                                            *mMaterial,
                                            density);

    if(!actor) return;

    actor->setAngularDamping(0.75);
    actor->setLinearDamping(0.02);
    actor->setLinearVelocity(PxVec3(.0f, .0f ,0));
    actor->setSleepThreshold(0.0);
    _phyEngine->addActor("main", actor);

    actor->addForce(PxVec3(viewDir.x(),viewDir.y(),viewDir.z()*1.3), PxForceMode::eIMPULSE);

    osg::ref_ptr<osg::MatrixTransform> sphereTrans = addSphere(parent, pos, radius);
    sphereTrans->addUpdateCallback(new UpdateActorCallback(actor));
}

ref_ptr<MatrixTransform> PhysxBall::addSphere(osg::Group*parent, Vec3f pos, float radius)
{
    osg::ref_ptr<osg::ShapeDrawable> shape = new osg::ShapeDrawable();
    shape->setShape(new osg::Sphere(Vec3f(.0,.0,.0), radius));
    shape->setColor(osg::Vec4f(1.0f,.0f,.0f,1.0f));
    shape->getOrCreateStateSet()->setMode( GL_LIGHTING, osg::StateAttribute::OFF );
    osg::ref_ptr<osg::Geode> node = new osg::Geode;
    ///////use shader
    Program * program =assetLoader::instance()->createShaderProgramFromFile("shaders/lighting.vert","shaders/lighting.frag");
    osg::StateSet * stateSet = shape->getOrCreateStateSet();
    stateSet->setAttributeAndModes(program);

    stateSet->addUniform( new osg::Uniform("lightDiffuse",
                                           osg::Vec4(0.8f, 0.8f, 0.8f, 1.0f)) );
    stateSet->addUniform( new osg::Uniform("lightSpecular",
                                           osg::Vec4(1.0f, 1.0f, 0.4f, 1.0f)) );
    stateSet->addUniform( new osg::Uniform("shininess", 64.0f) );

    stateSet->addUniform( new osg::Uniform("lightPosition", osg::Vec3(0,0,1)));

    node->addDrawable(shape.get());
    ref_ptr<MatrixTransform> sphereTrans = new MatrixTransform;
    Matrixf m;
    m.makeTranslate(pos);
    sphereTrans->setMatrix(m);
    sphereTrans->addChild(node.get());

    parent->addChild(sphereTrans.get());
    return sphereTrans.get();
}


void PhysxBall::createBox(osg::Group *parent, osg::Vec3f extent, osg::Vec3f color, PlaneData* pp) {


    PxReal density = 1000.0f;
    PxMaterial* mMaterial = _phyEngine->getPhysicsSDK()->createMaterial(10,10,0.65);
    PxBoxGeometry geometryBox(PxVec3(extent.x()/2,extent.y()/2,extent.z()/2));
    PxTransform transform(toPhysicsMatrix(pp->_model_mat));
    PxRigidDynamic *actor = PxCreateDynamic(* _phyEngine->getPhysicsSDK(),
                                            transform,
                                            geometryBox,
                                            *mMaterial,
                                            density);

    if(!actor) return;
    actor->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, true);
    actor->setAngularDamping(0.75);
    actor->setLinearDamping(0.02);
    actor->setLinearVelocity(PxVec3(.0f, .0f ,0));
    actor->setSleepThreshold(0.0);
    _phyEngine->addActor("main", actor);
    osg::ref_ptr<osg::MatrixTransform> boxTrans = addBox(parent,extent, color, actor, pp);
    auto cb = new UpdateActorCallback(actor, pp);
    //cb->update_mat = false;
    boxTrans->addUpdateCallback(cb);
}

ref_ptr<MatrixTransform> PhysxBall::addBox(osg::Group *parent, osg::Vec3f extent, osg::Vec3f color, PxRigidDynamic* physx_box,PlaneData* pp)
{
    osg::ref_ptr<osg::ShapeDrawable> shape = new osg::ShapeDrawable();
    shape->setUseDisplayList(false);
    auto box = new osg::Box(Vec3f(0,0,0), extent.x(), extent.y(), extent.z());

    shape->setShape(box);
    shape->setColor(osg::Vec4f(color,0.3f));

    osg::ref_ptr<osg::Geode> node = new osg::Geode;
    ///////use shader
    osg::BlendFunc *func = new osg::BlendFunc();
    func->setFunction(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    Program * program =assetLoader::instance()->createShaderProgramFromFile("shaders/wall.vert","shaders/wall.frag");
    osg::StateSet * stateSet = shape->getOrCreateStateSet();
    stateSet->setAttributeAndModes(program,osg::StateAttribute::ON| osg::StateAttribute::OVERRIDE);
    stateSet->setMode(GL_DEPTH_TEST, osg::StateAttribute::ON| osg::StateAttribute::OVERRIDE);
    stateSet->setRenderBinDetails(500, "transparent");
    stateSet->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
    stateSet->setAttributeAndModes(func, osg::StateAttribute::ON| osg::StateAttribute::OVERRIDE);
    stateSet->addUniform( new osg::Uniform("color",
                                           osg::Vec4(color, 0.3f)) );


    node->addDrawable(shape.get());
    ref_ptr<MatrixTransform> boxTrans = new MatrixTransform;

    boxTrans->addChild(node.get());
    bounding.push_back(node);
    if(!_planeBButton->getValue()){
        node->setNodeMask(0x0);
    }
    parent->addChild(boxTrans.get());
    physx_osg_pair.push_back(std::tuple(physx_box, boxTrans.get()));
    return boxTrans.get();
}

void PhysxBall::syncPlane() {
    _session = ARCoreManager::instance()->getArSession();
    const auto plane_ptrs = ARCoreManager::instance()->getPlanePointers();
    auto plane_color_map = ARCoreManager::instance()->getPlaneMap();
    _viewer = CVRViewer::instance();
    if(_bCreatePlane->getValue()) {
        osg::Vec3 eye, at, up;
        auto mat = _viewer->getCamera()->getViewMatrix();
        mat.getLookAt( eye, at, up );
        for (int i = 0; i < plane_ptrs.size(); i++) {
            auto arPlane = plane_ptrs[i];
            PlaneData *pp = new PlaneData;
            //get model matrix
            ArPose *arPose;
            ArPose_create(_session, nullptr, &arPose);
            ArPlane_getCenterPose(_session, arPlane, arPose);
            ArPose_getMatrix(_session, arPose, pp->_model_mat.ptr());

            pp->_model_mat(3,1) = -pp->_model_mat(3,1);

            osg::Matrixf swap_mat;

            swap_mat(1,1) = 0;
            swap_mat(1,2) = -1;
            swap_mat(2,1) = -1;
            swap_mat(2,2) = 0;

            pp->_model_mat = swap_mat * pp->_model_mat * swap_mat;

            float raw_center_pose[7] = {.0f};
            ArPose_getPoseRaw(_session, arPose, raw_center_pose);

            osg::Quat q(raw_center_pose[0], raw_center_pose[2], raw_center_pose[1], raw_center_pose[3]);

            //float extentXZ[2];
            ArPlane_getExtentX(_session, arPlane, &pp->_extentXZ[0]);
            ArPlane_getExtentZ(_session, arPlane, &pp->_extentXZ[1]);

            //int planeType;
            ArPlane_getType(_session, arPlane, (ArPlaneType *) &pp->_planeType);
            int debug_size = pp_list.size();
            if (i >= pp_list.size()) {
                pp_list.push_back(pp);
                auto color =  plane_color_map[arPlane];

                // create box for the plane
                osg::Vec3 dimensions;

                dimensions = osg::Vec3(pp->_extentXZ[0]*1.1, pp->_extentXZ[1]*1.1, 0.001);
                osg::Vec3 pos;//pp->_centerPose[0],-pp->_centerPose[2],pp->_centerPose[1]);
                createBox(_scene, dimensions, color, pp);
            } else {
                delete(pp_list[i]);
                pp_list[i] = pp;
                if(count>=delay) {
                    physx::PxRigidDynamic *actor;
                    osg::ref_ptr<osg::MatrixTransform> boxTrans;
                    std::tie(actor, boxTrans) = physx_osg_pair[i];

                    auto geo_p = static_cast<osg::Geode *>(boxTrans->getChild(0));
                    auto draw_p = static_cast<osg::ShapeDrawable *>(geo_p->getChild(0));

                    auto shape_p = static_cast<osg::Box *>(draw_p->getShape());
                    osg::Vec3f extent;

                    extent = osg::Vec3(pp->_extentXZ[0] * 1.1/2, pp->_extentXZ[1] * 1.1/2, 0.001/2);

                    shape_p->setHalfLengths(extent);

                    draw_p->dirtyDisplayList();
                    draw_p->build();

                    PxShape * buf;

                    actor->getShapes(&buf,1);

                    PxBoxGeometry box(PxVec3(extent.x(),extent.y(),extent.z()));
                    (buf[0]).setGeometry(box);
                    //PxQuat(q.x(),q.y(),q.z(),q.w())
                    PxTransform transform(osgPhysx::toPhysicsMatrix(pp->_model_mat));
                    actor->setGlobalPose(transform);
                    count = 0;
                }else{
                    count++;
                }
            }
        }
    }
}