#ifndef DRAWABLE_ENTRANCE_H
#define DRAWABLE_ENTRANCE_H

// STD
#include <queue>
#include <vector>
#include <sys/types.h>
#include <sys/stat.h>

// CVR
#include <cvrKernel/CVRPlugin.h>
#include <cvrKernel/ScreenBase.h>
#include <cvrKernel/SceneManager.h>
#include <cvrKernel/SceneObject.h>
#include <cvrKernel/Navigation.h>
#include <cvrKernel/PluginHelper.h>
#include <cvrKernel/ComController.h>
#include <cvrMenu/MenuSystem.h>
#include <cvrMenu/SubMenu.h>
#include <cvrMenu/MenuButton.h>
#include <cvrMenu/MenuCheckbox.h>
#include <cvrMenu/MenuRangeValue.h>
#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/FileHandler.h>
#include <cvrUtil/AndroidHelper.h>
// OSG
#include <osg/Group>
#include <osg/Vec3>
#include <osg/Uniform>
#include <osgDB/ReadFile>
#include <osgDB/FileUtils>

#include "pointDrawable.h"
#include "planeDrawable.h"
#include "strokeDrawable.h"
#include "quadDrawable.h"
#include "pano.h"
typedef struct IntersetctObj{
    osg::Uniform * uTexture;
    osg::MatrixTransform * matrixTrans;
    osg::Uniform * modelMatUniform;
}isectObj;
enum sceneState{
    FREE = 0,
    ROTATE,
    TRANSLATE
};
enum LightingType{
    ARCORE_CORRECTION = 0,
    ONES_SOURCE,
    SPHERICAL_HARMONICS

};
class GlesDrawables : public cvr::CVRPlugin, public cvr::MenuCallback
{
typedef osg::ref_ptr<osg::MatrixTransform> Transform;
private:
//    const float ENV_QUAD_COORDS[6][12]  = {
//            { -0.4f, -0.6f, 0.0f, -0.4f, -0.4f, 0.0f, -0.1f, -0.6f, 0.0f, -0.1f, -0.4f, 0.0f },
//            { -1.0f, -0.6f, 0.0f, -1.0f, -0.4f, 0.0f, -0.7f, -0.6f, 0.0f, -0.7f, -0.4f, 0.0f },
//            { -0.7f, -0.4f, 0.0f, -0.7f, -0.2f, 0.0f, -0.4f, -0.4f, 0.0f, -0.4f, -0.2f, 0.0f },
//            { -0.7f, -0.8f, 0.0f, -0.7f, -0.6f, 0.0f, -0.4f, -0.8f, 0.0f, -0.4f, -0.6f, 0.0f },
//            { -0.1f, -0.6f, 0.0f, -0.1f, -0.4f, 0.0f, 0.2f, -0.6f, 0.0f, 0.2f, -0.4f, 0.0f },
//            { -0.7f, -0.6f, 0.0f, -0.7f, -0.4f, 0.0f, -0.4f, -0.6f, 0.0f, -0.4f, -0.4f, 0.0f }
//    };
    const float ENV_QUAD_COORDS[6][12]  = {
            { -0.4f, -0.4f, 0.0f,  -0.1f, -0.4f, 0.0f, -0.1f, -0.6f, 0.0f, -0.4f, -0.6f, 0.0f},
            { -1.0f, -0.4f, 0.0f, -0.7f, -0.4f, 0.0f, -0.7f, -0.6f, 0.0f,-1.0f, -0.6f, 0.0f},
            { -0.7f, -0.2f, 0.0f, -0.4f, -0.2f, 0.0f, -0.4f, -0.4f, 0.0f,-0.7f, -0.4f, 0.0f},
            {  -0.7f, -0.6f, 0.0f,-0.4f, -0.6f, 0.0f,-0.4f, -0.8f, 0.0f,-0.7f, -0.8f, 0.0f},
            { -0.1f, -0.4f, 0.0f, 0.2f, -0.4f, 0.0f, 0.2f, -0.6f, 0.0f,-0.1f, -0.6f, 0.0f},
            { -0.7f, -0.4f, 0.0f, -0.4f, -0.4f, 0.0f,-0.4f, -0.6f, 0.0f,-0.7f, -0.6f, 0.0f}
    };

protected:
    cvr::SubMenu *_mainMenu;

    cvr::MenuCheckbox *_pointButton, *_planeButton, *_quadButton;
    cvr::MenuCheckbox* _obj1Button, *_obj2Button, *_obj3Button, *_lightButton;

    osg::Group *_root, *_objects;
    cvr::SceneObject *rootSO, *objSO;

    osg::ref_ptr<pointDrawable> _pointcloudDrawable;
    std::vector<quadDrawable*> _quadDrawables;
    int _plane_num = 0, _objNum = 0;
    std::vector<planeDrawable*> _planeDrawables;
    osg::ref_ptr<strokeDrawable> _strokeDrawable;
    std::unordered_map<osg::Node*, isectObj> _map;
    osg::Node* _selectedNode = nullptr;
    sceneState _selectState = FREE;
    osg::Vec2f _mPreviousPos;
    bool last_state_plane = true, last_state_point=true, last_state_quad = true, _add_light=false;
    int last_object_select = 1;
    panoStitcher* stitcher;


    void initMenuButtons();
    void createDebugSphere(osg::Group *, osg::Matrixf);
    void createObject(osg::Group *, const char*, const char*, osg::Matrixf, LightingType);
    void createObject(osg::Group *, const char*, const char*, osg::Matrixf, bool opengl);
    bool tackleHitted(osgUtil::LineSegmentIntersector::Intersection result );
public:
    bool init();
    void menuCallback(cvr::MenuItem * item);
    void postFrame();
    bool processEvent(cvr::InteractionEvent * event);
};

#endif
