/***************************************************************
* File Name: DesignObjectHandler.h
*
* Class Name: DesignObjectHandler
*
***************************************************************/

#ifndef _DESIGN_OBJECT_HANDLER_H_
#define _DESIGN_OBJECT_HANDLER_H_


// C++
#include <string.h>

// Open scene graph
#include <osg/Group>
#include <osg/Switch>

// local includes
#include "SnapLevelController.h"
#include "VirtualScenicHandler.h"
#include "DesignObjects/DOGeometryCollector.h"
#include "DesignObjects/DOGeometryCreator.h"
#include "DesignObjects/DOGeometryEditor.h"
#include "DesignObjects/DOGeometryCloner.h"


/***************************************************************
* Class: DesignObjectHandler
***************************************************************/
class DesignObjectHandler
{
  public:
    DesignObjectHandler(osg::Group* rootGroup);
    ~DesignObjectHandler();

    void setActive(bool flag);
    void inputDevMoveEvent();
    bool inputDevPressEvent(const osg::Vec3 &pointerOrg, const osg::Vec3 &pointerPos);
    bool inputDevReleaseEvent();
    void update();

    /* access handler pointers which can be passed to 'DesignStateHandler' */
    osg::Group *getNonIntersectableSceneGraphPtr() { return mNonIntersectableSceneGraphPtr; }
    osg::Group *getIntersectableSceneGraphPtr() { return mIntersectableSceneGraphPtr; }
    osg::Switch *getCAVEShapeSwitchPtr() { return mCAVEShapeSwitch; }
    VirtualScenicHandler *getScenicHandlerPtr() { return mVirtualScenicHandler; }

    DOGeometryCollector *getDOGeometryCollectorPtr() { return mDOGeometryCollector; }
    DOGeometryCreator *getDOGeometryCreatorPtr() { return mDOGeometryCreator; }
    DOGeometryEditor *getDOGeometryEditorPtr() { return mDOGeometryEditor; }

  protected:

    /* most upper level of scene graph structure: takes all design objects that contained
       in world space, with the exception of menu items that usually comes with viewer  */
    osg::Group *mDesignObjectRoot;
    osg::Group *mNonIntersectableSceneGraphPtr;		// non-intersectable objects
    osg::Group *mIntersectableSceneGraphPtr;		// intersectable objects
    osg::Group *mRoot;

    // switches under 'mIntersectableSceneGraphPtr' that control different types of CAVEGeode objects
    osg::Switch *mCAVEShapeSwitch;
    osg::Switch *mCAVEIconSurfaceSwitch;
    osg::Switch *mCAVEIconToolkitSwitch;

    VirtualScenicHandler *mVirtualScenicHandler;

    /* design object pointers controlled by 'DesignObjectHandler', these pointers are
       linked to design states that handles geometry creation and editting. */
    DOGeometryCollector *mDOGeometryCollector;
    DOGeometryCreator *mDOGeometryCreator;
    DOGeometryEditor *mDOGeometryEditor;
    DOGeometryCloner *mDOGeometryCloner;
};


#endif

