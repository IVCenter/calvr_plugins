#ifndef _PHYSXBALL_H
#define _PHYSXBALL_H

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
#include <cvrMenu/MenuRangeValueCompact.h>

// OSG
#include <osg/Group>
#include <osg/Vec3>
#include <osg/Uniform>
#include <osgDB/ReadFile>
#include <osgDB/FileUtils>

#include <osg/PositionAttitudeTransform> 	// the transform of objects
#include <foundation/PxVec3.h>
#include <osgText/Text>
#include <PxRigidActor.h>
#include <osg/ShapeDrawable>
#include "Engine.h"

/** The callback to update the actor, should be applied to a matrix transform node */

class PlaneData{
public:
    float _extentXZ[2];
    int _planeType;
    osg::Matrixf _model_mat;
};


/** The callback to update the actor, should be applied to a matrix transform node */
class UpdateActorCallback : public osg::NodeCallback
{
public:
    UpdateActorCallback( physx::PxRigidActor* a=0, PlaneData* pd = 0) : _actor(a), _pd(pd) {}

    UpdateActorCallback( const UpdateActorCallback& copy, const osg::CopyOp& op=osg::CopyOp::SHALLOW_COPY )
            : osg::NodeCallback(copy, op), _actor(copy._actor) {}

//    META_Object( osgPhysics, UpdateActorCallback );

    virtual void operator()( osg::Node* node, osg::NodeVisitor* nv );

protected:
    physx::PxRigidActor* _actor;
    PlaneData * _pd;
public:
    bool update_mat = true;
};

class PhysxBall : public cvr::CVRPlugin, public cvr::MenuCallback
{
private:
    void createBall(osg::Group* parent, osg::Vec3f pos, float radius);


    osg::ref_ptr<osg::MatrixTransform> addSphere(osg::Group*parent, osg::Vec3f pos, float radius);

    void createBox(osg::Group* parent, osg::Vec3f extent, osg::Vec3f color, PlaneData* pp);
    osg::ref_ptr<osg::MatrixTransform> addBox(osg::Group*parent, osg::Vec3f extent, osg::Vec3f color, physx::PxRigidDynamic* physx_box, PlaneData* pp);

    void initMenuButtons();
protected:
    osgPhysx::Engine * _phyEngine;
    cvr::SubMenu *_mainMenu;
    osg::Group* _menu, *_scene;
    cvr::SceneObject *rootSO, *sceneSO, *menuSo;
    std::vector<osg::Uniform *> _uniform_mvps;

//_uniform_mvps
//bounding
//pp_list
//PxGeo
    std::vector<osg::ref_ptr<osg::Geode>> bounding;
    ArSession * _session;
    cvr::CVRViewer * _viewer;
    cvr::MenuRangeValueCompact * _forceFactor;
    std::vector<PlaneData*> pp_list;
    std::vector<physx::PxBoxGeometry*> PxGeo;
    bool _bFirstPress = false;
    bool last_state = true;

    int count = 0;
    int delay = 150;


public:
    std::vector<std::tuple<physx::PxRigidDynamic*, osg::ref_ptr<osg::MatrixTransform>>> physx_osg_pair;
    cvr::MenuCheckbox * _bThrowBall;
    cvr::MenuCheckbox * _bCreatePlane, *_planePButton, *_planeBButton;
    bool processEvent(cvr::InteractionEvent * event);
    void throwBall();
    void syncPlane();
    bool init();
    void menuCallback(cvr::MenuItem * item);
    void preFrame();
//    void postFrame();
};

#endif
