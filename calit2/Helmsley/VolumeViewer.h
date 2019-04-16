#ifndef VOLUME_VIEWER_H
#define VOLUME_VIEWER_H

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


class VolumeViewer : public cvr::CVRPlugin, public cvr::MenuCallback
{
typedef osg::ref_ptr<osg::MatrixTransform> Transform;

protected:
    cvr::SubMenu *_mainMenu;

    osg::Group *_root, *_objects;
    cvr::SceneObject *rootSO, *objSO;

    void initMenuButtons();
    void createDebugSphere(osg::Group *, osg::Matrixf);
    //void createObject(osg::Group *, const char*, const char*, osg::Matrixf, LightingType);
    //void createObject(osg::Group *, const char*, const char*, osg::Matrixf, bool opengl);
    bool tackleHitted(osgUtil::LineSegmentIntersector::Intersection result );
public:
    bool init();
    void menuCallback(cvr::MenuItem * item);
    void postFrame();
    bool processEvent(cvr::InteractionEvent * event);
};

#endif
