#ifndef PLUGIN_BASIS_RENDERER_H
#define PLUGIN_BASIS_RENDERER_H

#include "pointDrawable.h"
#include "strokeDrawable.h"
#include "planeDrawable.h"

class basisRender{
public:
    osg::Group* createBasicRenderer();
    void updateOnFrame();

    void setPointCloudVisiable(bool on){
        if(on)
            _pointcloudDrawable->getGLNode()->setNodeMask(0xFFFFFF);
        else
            _pointcloudDrawable->getGLNode()->setNodeMask(0);
    }
    void setPlaneVisiable(bool on){
        if(on){
            for(auto p:_planeDrawables)
                p->setNodeMask(0xFFFFFF);
        }else{
            for(auto p:_planeDrawables)
                p->setNodeMask(0);
        }
        plane_visiable = on;
    }
    void reset();
protected:
    osg::Group* _root;
    pointDrawable* _pointcloudDrawable;
    std::vector<planeDrawable*> _planeDrawables;
    strokeDrawable* _strokeDrawable;
    bool plane_visiable = true;
private:
    int _plane_num = 0;
};
#endif
