#include <cvrInput/TrackingManager.h>
#include "basisRenderer.h"

osg::Group* basisRender::createBasicRenderer(){
    _root = new osg::Group;
    _pointcloudDrawable = new pointDrawable;
    _root->addChild(_pointcloudDrawable->createDrawableNode());

    _strokeDrawable = new strokeDrawable;
    _root->addChild(_strokeDrawable->createDrawableNode(.0f,-0.8f));
    return _root;
}
void basisRender::updateOnFrame(){
    //point
    _pointcloudDrawable->updateOnFrame();
    //stroke
    osg::Vec3f isPoint;
    if(cvr::TrackingManager::instance()->getIsPoint(isPoint)){
        _strokeDrawable->updateOnFrame(isPoint);
        _strokeDrawable->getGLNode()->setNodeMask(0xFFFFFF);
    } else
        _strokeDrawable->getGLNode()->setNodeMask(0x0);
    //plane
    cvr::planeMap map = cvr::ARCoreManager::instance()->getPlaneMap();
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
}