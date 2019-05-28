#ifndef VOLUME_VIEWER_H
#define VOLUME_VIEWER_H

#include <cvrKernel/CVRPlugin.h>
#include <cvrMenu/MenuItem.h>
#include <osg/MatrixTransform>
#include <cvrMenu/SubMenu.h>
#include <cvrKernel/SceneObject.h>

#include "dcmRenderer.h"

class VolumeViewer : public cvr::CVRPlugin, public cvr::MenuCallback{
typedef osg::ref_ptr<osg::MatrixTransform> Transform;

protected:
    cvr::SubMenu *_mainMenu;

    osg::Group* _root;
    cvr::SceneObject *_rootSO;

    void initMenuButtons();
    void createDebugSphere(osg::Group *, osg::Matrixf);

public:
    bool init();
    void menuCallback(cvr::MenuItem * item);
    void postFrame();
    bool processEvent(cvr::InteractionEvent * event);

private:
    dcmRenderer* renderer;
};

#endif
