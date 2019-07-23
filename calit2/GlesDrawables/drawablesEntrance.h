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
//self
#include "basicDrawables/basisRenderer.h"

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
enum cbTypes{
    CB_POINT = 0,
    CB_PLANE,
    CB_LIGHT
};
private:
    std::unordered_map<int, bool> cb_map = {{CB_POINT, true}, {CB_PLANE, true}, {CB_LIGHT, true}, {CB_LIGHT+1, false}, {CB_LIGHT+2, false}};
protected:
    cvr::SubMenu *_mainMenu;

//    cvr::MenuCheckbox *_pointButton, *_planeButton, *_quadButton;
//    cvr::MenuCheckbox* _obj1Button, *_obj2Button, *_obj3Button, *_lightButton;

    std::vector<cvr::MenuCheckbox*> _vCheckBox;

    osg::Group *_root, *_objects;
    cvr::SceneObject *rootSO, *objSO;

    basisRender* basis_renderer = nullptr;

    std::unordered_map<osg::Node*, isectObj> _map;
    osg::Node* _selectedNode = nullptr;
    sceneState _selectState = FREE;
    osg::Vec2f _mPreviousPos;

    //_add_light=false;
    int last_object_select = 1;


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
