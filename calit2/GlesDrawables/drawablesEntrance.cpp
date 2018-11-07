#include <cvrUtil/AndroidStdio.h>
#include <osg/LineWidth>
#include <osgUtil/Tessellator>
#include <osg/Texture>
#include "drawablesEntrance.h"
#include "planeDrawable.h"
#include <osg/Texture2D>
#include <cvrUtil/AndroidHelper.h>
#include <osg/ShapeDrawable>
#include <cvrKernel/PluginManager.h>

using namespace osg;
using namespace cvr;

bool GlesDrawables:: tackleHitted(osgUtil::LineSegmentIntersector::Intersection result ){
//    LOGE("==== parent Num: %d", result.drawable->getNumParents());
    osg::Node* parent = dynamic_cast<Node*>(result.drawable->getParent(0));
    if(_map.empty() || _map.find(parent) ==_map.end()){
//        MatrixTransform *transform =
//        Matrixf mat = transform->getMatrix();
//        Vec3f pos = mat.getTrans();
//        LOGE("====getpos: %f, %f, %f", pos.x(), pos.y(), pos.z());
        isectObj obj;
        obj.uTexture = parent->getOrCreateStateSet()->getUniform("uTextureChoice");
        obj.matrixTrans = dynamic_cast<MatrixTransform *>(parent->getParent(0));

        _map[parent] = obj;
        bool textureChoice;
        _map[parent].uTexture->get(textureChoice);
        _map[parent].uTexture->set(!textureChoice);
        PluginManager::setCallBackRequest("popButtons");
        return true;
    }
    return false;
}

void GlesDrawables::initMenuButtons() {
    _pointButton = new MenuButton("Show PointCloud");
    _pointButton->setCallback(this);
    _mainMenu->addItem(_pointButton);

    _planeButton = new MenuButton("Show Detected Planes");
    _planeButton->setCallback(this);
    _mainMenu->addItem(_planeButton);

    _strokeButton = new MenuButton("Show Stroke Ray");
    _strokeButton->setCallback(this);
    _mainMenu->addItem(_strokeButton);
}

bool GlesDrawables::init() {
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

    PluginHelper::registerSceneObject(rootSO, "GlesDrawablesPlugin");
    rootSO->dirtyBounds();
    rootSO->attachToScene();

    _strokeDrawable = new strokeDrawable;
    _root->addChild(_strokeDrawable->createDrawableNode(.0f,-0.8f));

    _pointcloudDrawable = new pointDrawable;
    _root->addChild(_pointcloudDrawable->createDrawableNode());

//    createObject(_objects,
//                 "models/andy.obj", "textures/andy.png",
//                 Matrixf::rotate(PI_2f, Vec3f(.0,.0,1.0)) * Matrixf::translate(Vec3f(.0f, 0.5f, .0f)));
//    _objects->addUpdateCallback(new )
    return true;
}

void GlesDrawables::menuCallback(cvr::MenuItem *item) {
}

void GlesDrawables::postFrame() {
    _pointcloudDrawable->updateOnFrame();
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
                createObject(_objects,"models/andy-origin.obj", "textures/andy.png", modelMat);
            }

        }
        _objNum = anchor_num;
    }
}

bool GlesDrawables::processEvent(cvr::InteractionEvent * event){
    AndroidInteractionEvent * aie = event->asAndroidEvent();
    if(aie->getTouchType() == TRANS_BUTTON){

        return true;
    }
    if(aie->getTouchType() == ROT_BUTTON){
        return true;
    }

    if(aie->getTouchType() != LEFT)
        return false;

    Vec2f touchPos = Vec2f(aie->getX(), aie->getY());
    if(aie->getInteraction()==BUTTON_DOUBLE_CLICK){
        ARCoreManager::instance()->updatePlaneHittest(touchPos.x(), touchPos.y());
        return true;
    }

    if(aie->getInteraction()== BUTTON_DOWN){
        Matrixf vpMat =ARCoreManager::instance()->getMVPMatrix();
        vpMat = Matrixf::inverse(vpMat);

        osg::Vec3 pointerStart, pointerEnd;
        pointerStart = TrackingManager::instance()->getHandMat(0).getTrans();
        TrackingManager::instance()->getScreenToClientPos(touchPos);
        Vec4f vIn(touchPos.x(),touchPos.y(),-1, 1.0f);
        Vec4f pos = vIn * vpMat;
        float inv_w = 1.0f / pos.w();// * ConfigManager::UNIT_ALIGN_FACTOR;


        Vec4f testScreen = Vec4f(pos.x()*inv_w, pos.y()*inv_w, pos.z()*inv_w, 1.0) *ARCoreManager::instance()->getMVPMatrix();
        pointerEnd = Vec3f(pos.x() * inv_w, -pos.z()*inv_w, pos.y()*inv_w);
        Vec3f dir = pointerEnd-pointerStart;
        float t = (10-pointerStart.y())/dir.y();
        pointerEnd = Vec3f(pointerStart.x() + t*dir.x(), 10.0f, pointerStart.z() + t*dir.z());

        osg::ref_ptr<osgUtil::LineSegmentIntersector> handseg = new osgUtil::LineSegmentIntersector(pointerStart, pointerEnd);

        osgUtil::IntersectionVisitor iv(handseg.get());
        _objects->accept( iv );
        if ( handseg->containsIntersections()){
            _map.clear();
            for(auto itr=handseg->getIntersections().begin(); itr!=handseg->getIntersections().end(); itr++){
                if(tackleHitted(*itr))
                    break;
            }
        }

        return true;
    }
    return false;
}

void GlesDrawables::createObject(osg::Group *parent,
                                 const char* obj_file_name, const char* png_file_name,
                                 Matrixf modelMat) {
    Transform objectTrans = new MatrixTransform;
    objectTrans->setMatrix(modelMat);

    ref_ptr<Geometry>_geometry = new osg::Geometry();
    ref_ptr<Geode> _node = new osg::Geode;
    _node->addDrawable(_geometry.get());

    ref_ptr<Vec3Array> vertices = new Vec3Array();
    ref_ptr<Vec3Array> normals = new Vec3Array();

    ref_ptr<Vec2Array> uvs = new Vec2Array();

    std::vector<GLfloat> _vertices;
    std::vector<GLfloat > _uvs;
    std::vector<GLfloat > _normals;
    std::vector<GLushort > _indices;

    assetLoader::instance()->LoadObjFile(obj_file_name, &_vertices, &_normals, &_uvs, &_indices);


    //REstore in OSG Coord or pass REAL_TO_OSG_COORD matrix to shader to flip
    for(int i=0; i<_uvs.size()/2; i++){
        vertices->push_back(Vec3f(_vertices[3*i], -_vertices[3*i+2], _vertices[3*i+1]));
        normals->push_back(Vec3f(_normals[3*i], -_normals[3*i+2], _normals[3*i+1]));
        uvs->push_back(Vec2f(_uvs[2*i], _uvs[2*i+1]));
    }

    _geometry->setVertexArray(vertices.get());
    _geometry->setNormalArray(normals.get());
    _geometry->setTexCoordArray(0, uvs.get());
    _geometry->addPrimitiveSet(new DrawElementsUShort(GL_TRIANGLES, (unsigned int)_indices.size(), _indices.data()));
    _geometry->setUseVertexBufferObjects(true);
    _geometry->setUseDisplayList(false);

    std::string fhead(getenv("CALVR_RESOURCE_DIR"));

    Program * program =assetLoader::instance()->createShaderProgramFromFile("shaders/objectOSG.vert","shaders/objectOSG.frag");
    osg::StateSet * stateSet = _node->getOrCreateStateSet();
    stateSet->setAttributeAndModes(program);

    stateSet->addUniform( new osg::Uniform("lightDiffuse",
                                           osg::Vec4(0.8f, 0.8f, 0.8f, 1.0f)) );
    stateSet->addUniform( new osg::Uniform("lightSpecular",
                                           osg::Vec4(1.0f, 1.0f, 0.4f, 1.0f)) );
    stateSet->addUniform( new osg::Uniform("shininess", 64.0f) );
    stateSet->addUniform( new osg::Uniform("lightPosition",
                                           osg::Vec3(0,0,1)));

    Uniform * envColorUniform = new Uniform(Uniform::FLOAT_VEC4, "uColorCorrection");
    envColorUniform->setUpdateCallback(new envLightCallback);
    stateSet->addUniform(envColorUniform);

    osg::ref_ptr<osg::Texture2D> texture = new osg::Texture2D;
    texture->setImage( osgDB::readImageFile(fhead+png_file_name) );
    texture->setWrap( osg::Texture2D::WRAP_S, osg::Texture2D::REPEAT );
    texture->setWrap( osg::Texture2D::WRAP_T, osg::Texture2D::REPEAT );
    texture->setFilter( osg::Texture::MIN_FILTER, osg::Texture::LINEAR_MIPMAP_LINEAR );
    texture->setFilter( osg::Texture::MAG_FILTER, osg::Texture::LINEAR );

    osg::ref_ptr<osg::Texture2D> changeTexture = new osg::Texture2D;
    changeTexture->setImage( osgDB::readImageFile(fhead+"textures/andy-change.png") );
    changeTexture->setWrap( osg::Texture2D::WRAP_S, osg::Texture2D::REPEAT );
    changeTexture->setWrap( osg::Texture2D::WRAP_T, osg::Texture2D::REPEAT );
    changeTexture->setFilter( osg::Texture::MIN_FILTER, osg::Texture::LINEAR_MIPMAP_LINEAR );
    changeTexture->setFilter( osg::Texture::MAG_FILTER, osg::Texture::LINEAR );


    stateSet->setTextureAttributeAndModes(1, texture.get());
    stateSet->addUniform(new osg::Uniform("uSampler", 1));

    stateSet->setTextureAttributeAndModes(2, changeTexture.get());
    stateSet->addUniform(new osg::Uniform("uSamplerC", 2));

    stateSet->addUniform(new osg::Uniform("uTextureChoice", true));

    objectTrans->addChild(_node.get());
    parent->addChild(objectTrans.get());
}

void GlesDrawables::createObject(osg::Group *parent,
                                 const char* obj_file_name, const char* png_file_name,
                                 Matrixf modelMat, bool opengl) {
    Transform objectTrans = new MatrixTransform;
    objectTrans->setMatrix(modelMat);
    ref_ptr<Geometry>_geometry = new osg::Geometry();
    ref_ptr<Geode> _node = new osg::Geode;
    _node->addDrawable(_geometry.get());

    ref_ptr<Vec3Array> vertices = new Vec3Array();
    ref_ptr<Vec3Array> normals = new Vec3Array();

    ref_ptr<Vec2Array> uvs = new Vec2Array();

    std::vector<GLfloat> _vertices;
    std::vector<GLfloat > _uvs;
    std::vector<GLfloat > _normals;
    std::vector<GLushort > _indices;

    assetLoader::instance()->LoadObjFile(obj_file_name, &_vertices, &_normals, &_uvs, &_indices);


    for(int i=0; i<_uvs.size()/2; i++){
        vertices->push_back(Vec3f(_vertices[3*i], _vertices[3*i+1], _vertices[3*i+2]));
        normals->push_back(Vec3f(_normals[3*i], _normals[3*i+1], _normals[3*i+2]));
        uvs->push_back(Vec2f(_uvs[2*i], _uvs[2*i+1]));
    }

    _geometry->setVertexArray(vertices.get());
    _geometry->setNormalArray(normals.get());
    _geometry->setTexCoordArray(0, uvs.get());
    _geometry->addPrimitiveSet(new DrawElementsUShort(GL_TRIANGLES, (unsigned int)_indices.size(), _indices.data()));
    _geometry->setUseVertexBufferObjects(true);
    _geometry->setUseDisplayList(false);

    std::string fhead(getenv("CALVR_RESOURCE_DIR"));

    Program * program =assetLoader::instance()->createShaderProgramFromFile("shaders/object.vert","shaders/object.frag");
    osg::StateSet * stateSet = _node->getOrCreateStateSet();
    stateSet->setAttributeAndModes(program);

    stateSet->addUniform( new osg::Uniform("lightDiffuse",
                                           osg::Vec4(0.8f, 0.8f, 0.8f, 1.0f)) );
    stateSet->addUniform( new osg::Uniform("lightSpecular",
                                           osg::Vec4(1.0f, 1.0f, 0.4f, 1.0f)) );
    stateSet->addUniform( new osg::Uniform("shininess", 64.0f) );
    stateSet->addUniform( new osg::Uniform("lightPosition",
                                           osg::Vec3(0,0,1)));

    Uniform * envColorUniform = new Uniform(Uniform::FLOAT_VEC4, "uColorCorrection");
    envColorUniform->setUpdateCallback(new envLightCallback);
    stateSet->addUniform(envColorUniform);

    Uniform * projUniform = new Uniform(Uniform::FLOAT_MAT4, "uProj");
    projUniform->setUpdateCallback(new projMatrixCallback);
    stateSet->addUniform(projUniform);

    Uniform * modelViewUniform = new Uniform(Uniform::FLOAT_MAT4, "uModelView");
    modelViewUniform->setUpdateCallback(new modelViewCallBack(modelMat));
    stateSet->addUniform(modelViewUniform);

    osg::ref_ptr<osg::Texture2D> texture = new osg::Texture2D;
    texture->setImage( osgDB::readImageFile(fhead+png_file_name) );
    texture->setWrap( osg::Texture2D::WRAP_S, osg::Texture2D::REPEAT );
    texture->setWrap( osg::Texture2D::WRAP_T, osg::Texture2D::REPEAT );
    texture->setFilter( osg::Texture::MIN_FILTER, osg::Texture::LINEAR_MIPMAP_LINEAR );
    texture->setFilter( osg::Texture::MAG_FILTER, osg::Texture::LINEAR );


    stateSet->setTextureAttributeAndModes(1, texture.get());
    stateSet->addUniform(new osg::Uniform("uSampler", 1));

    objectTrans->addChild(_node.get());
    parent->addChild(objectTrans.get());
}
