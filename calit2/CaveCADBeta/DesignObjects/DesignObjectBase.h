/***************************************************************
* File Name: DesignObjectBase.h
*
* Description: Base class of design object tools
*
***************************************************************/

#ifndef _DESIGN_OBJECT_BASE_H_
#define _DESIGN_OBJECT_BASE_H_


// C++
#include <string.h>

// Open scene graph
#include <osg/Group>
#include <osg/Switch>


/***************************************************************
* Class: DesignObjectBase
***************************************************************/
class DesignObjectBase
{
  public:
    DesignObjectBase();

    virtual ~DesignObjectBase() {}

    /* virtual function calls as initialization process when instances are created,
       make sure that 'initSceneGraphPtr' is called ahead of 'initDesignObjects'
    */
    virtual void initSceneGraphPtr( osg::Group *nonInterSCPtr, osg::Group *interSCPtr,
				    osg::Switch *shapeSwitch, osg::Switch *surfaceIconSwitch, 
				    osg::Switch *toolkitIconSwitch);
    virtual void initDesignObjects() = 0;

  protected:

    /* intersectable and non-intersectable root groups passed from 'DesignObjectHandler' */
    osg::Group *mNonIntersectableSceneGraphPtr;
    osg::Group *mIntersectableSceneGraphPtr;

    osg::Switch *mDOShapeSwitch;
    osg::Switch *mDOIconSurfaceSwitch;
    osg::Switch *mDOIconToolkitSwitch;
};


#endif

