#ifndef VOLUME_VIEWER_H
#define VOLUME_VIEWER_H

#include <osg/MatrixTransform>

#include <cvrKernel/CVRPlugin.h>
#include <cvrKernel/SceneObject.h>
//cvr menu
#include <cvrMenu/SubMenu.h>
#include <cvrMenu/MenuCheckbox.h>
#include <cvrMenu/MenuItem.h>

#include "dcmRenderer.h"
#include "basicDrawables/basisRenderer.h"

class VolumeViewer : public cvr::CVRPlugin, public cvr::MenuCallback{
typedef osg::ref_ptr<osg::MatrixTransform> Transform;

protected:
    cvr::SubMenu *_mainMenu;

    osg::Group* _root;
    cvr::SceneObject *_rootSO;

    cvr::MenuCheckbox *_pointCB, *_planeCB;

    void initMenuButtons();

public:
    bool init();
    void menuCallback(cvr::MenuItem * item);
    void postFrame();
    bool processEvent(cvr::InteractionEvent * event);

private:
    dcmRenderer* dcm_renderer = nullptr;
    basisRender* basis_renderer = nullptr;

    bool _dcm_initialized = false;
};

#endif
