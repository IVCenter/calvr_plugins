#ifndef VOLUME_VIEWER_H
#define VOLUME_VIEWER_H

#include <osg/MatrixTransform>

#include <cvrKernel/CVRPlugin.h>
#include <cvrKernel/SceneObject.h>
//cvr menu
#include <cvrMenu/SubMenu.h>
#include <cvrMenu/MenuCheckbox.h>
#include <cvrMenu/MenuItem.h>

//#include "dcmRenderer.h"
#include "dcmRenderer_osg.h"
#include "basicDrawables/basisRenderer.h"

class VolumeViewer : public cvr::CVRPlugin, public cvr::MenuCallback{
typedef osg::ref_ptr<osg::MatrixTransform> Transform;

protected:
    cvr::SubMenu *_mainMenu, *_tuneMenu;

    osg::Group* _root, *_scene;
    cvr::SceneObject *_rootSO, *_sceneSO;

    cvr::MenuCheckbox *_pointCB, *_planeCB, *_cutCB, *_osgCB;
    std::vector<cvr::MenuCheckbox*> _tuneCBs;

    void initMenuButtons();

public:
    bool init();
    void menuCallback(cvr::MenuItem * item);
    void postFrame();
    bool processEvent(cvr::InteractionEvent * event);

private:
//    dcmRenderer* dcm_renderer = nullptr;
    dcmRendererOSG* dcmRenderer_OSG;
    basisRender* basis_renderer = nullptr;

    bool _dcm_initialized = false;

    //interaction
    int current_tune_id = -1;
    const float MAX_VALUE_TUNE[4] = {800.0f, 2.0f, 500.0f, 1.0f};

};

#endif
