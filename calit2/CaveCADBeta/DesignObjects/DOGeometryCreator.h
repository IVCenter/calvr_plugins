/***************************************************************
* File Name: DOGeometryCreator.h
*
* Description:
*
***************************************************************/

#ifndef _DO_GEOMETRY_CREATOR_H_
#define _DO_GEOMETRY_CREATOR_H_


// C++
#include <string.h>

// Open scene graph
#include <osg/Group>
#include <osg/Switch>

// local includes
#include "DesignObjectBase.h"

#include "../Geometry/CAVEGroupReference.h"
#include "../Geometry/CAVEGroupShape.h"
#include "../Geometry/CAVEGeodeShape.h"
#include "../Geometry/CAVEGeodeSnapWireframe.h"
#include "../Geometry/CAVEGeodeSnapSolidshape.h"
#include "../Geometry/CAVEGeometry.h"

#include "../AnimationModeler/ANIMGeometryCreator.h"


/***************************************************************
* Class: DOGeometryCreator
***************************************************************/
class DOGeometryCreator: public DesignObjectBase
{
  public:
    DOGeometryCreator();

    virtual void initDesignObjects();

    // 'DSGeometryCreator' functions called by 'DSGeometryCreator'
    void setWireframeActiveID(const int &idx);
    void setSolidshapeActiveID(const int &idx);
    void setWireframeInitPos(const osg::Vec3 &initPos);
    void setSolidshapeInitPos(const osg::Vec3 &initPos, bool snap = true);
    void resetWireframeGeodes(const osg::Vec3 &centerPos);
    void setReferencePlaneMasking(bool flagXY, bool flagXZ, bool flagYZ);
    void setReferenceAxisMasking(bool flag);
    void setScalePerUnit(const float &scalePerUnit, const std::string &infoStr);
    void updateReferenceAxis();
    void updateReferencePlane(const osg::Vec3 &center, bool noSnap = false);
    void setPointerDir(const osg::Vec3 &pointerDir);
    void setSnapPos(const osg::Vec3 &snapPos, bool snap = true);
    void setResize(const float &s);
    void setResize(bool snap = true);
    void registerSolidShape();

  protected:

    /* In 'DOGeometryCreator' four set of scene graph objects are attached to
      'mNonIntersectableSceneGraphPtr': 1) mSnapWireframeSwitch; 2) mSnapSolidshapeSwitch; 
      3) mCAVEGroupRefPlane; 4) mCAVEGroupRefAxis; */
    int mWireframeActiveID, mSolidShapeActiveID;
    osg::Switch *mSnapWireframeSwitch, *mSnapSolidshapeSwitch;

    // CAVE objects that show up during geometry creation
    CAVEGroupReferencePlane *mCAVEGroupRefPlane;
    CAVEGroupReferenceAxis *mCAVEGroupRefAxis;
    CAVEGeodeSnapWireframe *mWireframeGeode;
    CAVEGeodeSnapSolidshape *mSolidshapeGeode;
};

#endif

