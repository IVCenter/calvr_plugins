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
#include <cvrUtil/ARCoreHelper.h>
#include <osg/BlendFunc>

using namespace osg;
using namespace cvr;
using namespace physx;
using namespace osgPhysx;

class modelCallback:public osg::UniformCallback{
    protected: physx::PxRigidDynamic * _actor;
public:
    modelCallback(PxRigidDynamic * actor):_actor(actor){}
    virtual void operator()(Uniform *uf, NodeVisitor *nv){
        PxMat44 matrix( _actor->getGlobalPose() );
        uf->set(osgPhysx::toMatrix(matrix));
        uf->dirty();
    }
};

//class mvpCallback:public osg::UniformCallback{
//public:
//    virtual void operator()(Uniform *uf, NodeVisitor *nv){
//        uf->set(Matrixf(glm::value_ptr(ARcoreHelper::instance()->getMVPMatrix())));
//        uf->dirty();
//    }
//};

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
void PhysxBall::createPlane(osg::Group* parent, osg::Vec3f pos) {
    PxMaterial* mMaterial = _phyEngine->getPhysicsSDK()->createMaterial(0.1, 0.2, 0.5);
    PxTransform pose = PxTransform(PxVec3(.0f, pos.z(), .0f),
                                   PxQuat(PxHalfPi, PxVec3(0.0f, 0.0f, 1.0f)));
    PxRigidStatic* plane = PxCreateStatic(*_phyEngine->getPhysicsSDK(), pose, PxPlaneGeometry(), *mMaterial);
    _phyEngine->addActor("main", plane);
//    _planeHeight = pos[1];
//    addBoard(parent, pos, Vec3f(1.0f, .0f,.0f), PI_2f);
}

void PhysxBall::preFrame() {
    _phyEngine->update();
    syncPlane();

//    if(_planeTurnedOn){
//        _planeTurnedOn = false;
//        std::vector<osg::Vec3f> planes = ARcoreHelper::instance()->getPlaneCenters();
//        if(planes.size()){
//            float x=0, y=0, z=0;
//            for(int i=0;i<planes.size(); i++){
//                x+=planes[i][0]; x+=planes[i][1];x+=planes[i][2];
//            }
//            createPlane(_scene, osg::Vec3f(x/planes.size(), y/planes.size(), z/planes.size()));
//            for(int i=0;i<5;i++)
//                createBall(_scene, osg::Vec3(x/planes.size() + std::rand()%10 * 0.01f-0.05f, y/planes.size() + std::rand()%10 * 0.1f, 0.5), z/planes.size() + 0.01f);
//        }
//
//    }

}
//void PhysxBall::postFrame() {
    //use ar controller to render
//    const float* pointCloudData;
//    int32_t num_of_points = 0;
//    if(!cvr::ARcoreHelper::instance()->getActiveState())
//        return;
//
//    cvr::ARcoreHelper::instance()->getPointCloudData(pointCloudData, num_of_points);
//    if(!pointCloudData)
//        return;
//    _pointcloudDrawable->updateVertices(pointCloudData, num_of_points);
//    _pointcloudDrawable->updateARMatrix(cvr::ARcoreHelper::instance()->getMVPMatrix());
//}

void PhysxBall::initMenuButtons() {
    _addButton = new MenuButton("Add Ball");
    _addButton->setCallback(this);
    _mainMenu->addItem(_addButton);

    _pointButton = new MenuButton("Show PointCloud");
    _pointButton->setCallback(this);
    _mainMenu->addItem(_pointButton);

    _planeButton = new MenuButton("Do Plane Detection");
    _planeButton->setCallback(this);
    _mainMenu->addItem(_planeButton);

    _addAndyButton = new MenuButton("Add a Cow");
    _addAndyButton->setCallback(this);
    _mainMenu->addItem(_addAndyButton);

    _delAndyButton = new MenuButton("Remove a Cow");
    _delAndyButton->setCallback(this);
    _mainMenu->addItem(_delAndyButton);

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
    _planeTurnedOn = ARcoreHelper::instance()->getPlaneStatus();
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
    if(item == _addButton)
        createBall(_scene, osg::Vec3f(.0f, 0.5f, 0.5f), 0.02f);
//    {
//        glm::vec3 featurePoint = ARcoreHelper::instance()->getRandomPointPos();
//        if(planeCreated)
//            createBall(_scene, glm::vec3(featurePoint[0], _planeHeight + 0.1f, featurePoint[2]), 0.5f);
//        else{
//            planeCreated = true;
//            createPlane(_scene, featurePoint);
//            createBall(_scene, featurePoint + glm::vec3(.0f, 0.01f, .0f), 0.01f);
//        }
//
//
//    }
//        createBall(_scene, osg::Vec3(std::rand() % 10 * 0.05f-0.25f, std::rand() % 10 * 0.1f, 0.5), 0.01f);
    else if(item == _pointButton)
        ARcoreHelper::instance()->changePointCloudStatus();
    else if(item == _planeButton){
        ARcoreHelper::instance()->turnOnPlane();
        _planeTurnedOn = true;
    }
    else if(item == _addAndyButton){
//        std::vector<glm::vec3> planes = ARcoreHelper::instance()->getPlaneCenters();
//        for(int i=0;i<planes.size();i++)
//            createObject(_scene, planes[i]);
        //ARCoreManager::instance()->
    }else if(item == _addButton) {
        //createBall(_scene, osg::Vec3(.0f, 0.5, 0.5), 0.01f);
    }
    else if(item == _planeBButton){
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
void PhysxBall::createPointCloud(osg::Group *parent) {
//    _pointcloudDrawable = new pointDrawable();
//    parent->addChild(_pointcloudDrawable->createDrawableNode(_assetHelper, &_glStateStack));
}
ref_ptr<Geometry> PhysxBall::_makeQuad(float width, float height, osg::Vec4f color, osg::Vec3 pos) {
    ref_ptr<Geometry> geo = new osg::Geometry();
    geo->setUseDisplayList(false);
    geo->setUseVertexBufferObjects(true);

    osg::Vec3Array* verts = new osg::Vec3Array();
    verts->push_back(pos);
    verts->push_back(pos + osg::Vec3(0, 0, height));
    verts->push_back(pos + osg::Vec3(width, 0, height));
    verts->push_back(pos + osg::Vec3(width,0,0));

    geo->setVertexArray(verts);

    geo->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUADS,0,4));

    osg::Vec4Array* colors = new osg::Vec4Array;
    colors->push_back(color);
    geo->setColorArray(colors);
    geo->setColorBinding(osg::Geometry::BIND_OVERALL);

    osg::Vec2Array* texcoords = new osg::Vec2Array;
    texcoords->push_back(osg::Vec2(0,0));
    texcoords->push_back(osg::Vec2(0, 1));
    texcoords->push_back(osg::Vec2(1, 1));
    texcoords->push_back(osg::Vec2(1,0));

    geo->setTexCoordArray(0,texcoords);

    return geo.get();
}

void PhysxBall::addBoard(Group* parent, osg::Vec3f pos, osg::Vec3f rotAxis, float rotAngle) {
    float boardWidth = 50, boardHeight = 50;

    ref_ptr<MatrixTransform> nodeTrans = new MatrixTransform();
    Matrixf transMat;
    transMat.makeTranslate(-boardWidth/2, 0, -boardHeight/2);
    transMat.makeTranslate(pos);
    osg::Geode * geode = new osg::Geode();

    geode->addDrawable(
            _makeQuad(boardWidth, boardHeight, Vec4f(.0f,.0f,.0f,1.0f), Vec3f(.0f,.0f,.0f)));
    if(rotAngle){
        Matrixf rotMat;
        rotMat.makeRotate(rotAngle, rotAxis);
        nodeTrans->setMatrix(rotMat * transMat);
    } else
        nodeTrans->setMatrix(transMat);
    nodeTrans->addChild(geode);
    parent->addChild(nodeTrans);
}
osg::Texture2D* createTexture( const std::string& filename )
{
    osg::ref_ptr<osg::Texture2D> texture = new osg::Texture2D;
    texture->setImage( osgDB::readImageFile(filename) );
    texture->setWrap( osg::Texture2D::WRAP_S, osg::Texture2D::REPEAT );
    texture->setWrap( osg::Texture2D::WRAP_T, osg::Texture2D::REPEAT );
    texture->setFilter( osg::Texture::MIN_FILTER, osg::Texture::LINEAR_MIPMAP_LINEAR );
    texture->setFilter( osg::Texture::MAG_FILTER, osg::Texture::LINEAR );
    return texture.release();
}
void PhysxBall::createObject(osg::Group *parent, Vec3f pos) {
    osg::ref_ptr<osg::MatrixTransform> objectTrans = new MatrixTransform;

    std::string fhead(getenv("CALVR_RESOURCE_DIR"));
    osg::ref_ptr<Node> objNode = osgDB::readNodeFile(fhead + "models/cow.osgt");

    ///////use shader
    Program * program =assetLoader::instance()->createShaderProgramFromFile("shaders/lightingOSG_test.vert","shaders/osgLightTexture.frag");
    osg::StateSet * stateSet = objNode->getOrCreateStateSet();
    stateSet->setAttributeAndModes(program);

    stateSet->setTextureAttributeAndModes(1, createTexture(fhead+"models/andy.png"), osg::StateAttribute::ON);

    stateSet->addUniform( new osg::Uniform("lightDiffuse",
                                           osg::Vec4(0.8f, 0.8f, 0.8f, 1.0f)) );
    stateSet->addUniform( new osg::Uniform("lightSpecular",
                                           osg::Vec4(1.0f, 1.0f, 0.4f, 1.0f)) );
    stateSet->addUniform( new osg::Uniform("shininess",
                                           64.0f) );
    stateSet->addUniform( new osg::Uniform("lightPosition",
                                           osg::Vec3(0,0,1)));
    stateSet->addUniform(new osg::Uniform("uScale", 0.1f));

    osg::Uniform* _uniform_sampler =new osg::Uniform(osg::Uniform::SAMPLER_2D, "uSampler");
    _uniform_sampler->set(1);
    stateSet->addUniform(_uniform_sampler);

    Uniform * mvpUniform = new Uniform(Uniform::FLOAT_MAT4, "uarMVP");
    mvpUniform->setUpdateCallback(new mvpCallback);
    stateSet->addUniform(mvpUniform);

    //uModel
    Uniform * modelUniform = new Uniform(Uniform::FLOAT_MAT4, "uModel");
//    modelUniform->set(Matrixf(glm::value_ptr(glm::translate(glm::mat4(), pos ))) );
    stateSet->addUniform(modelUniform);
    objectTrans->addChild(objNode.get());
    parent->addChild(objectTrans.get());
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