/***************************************************************
* File Name: DOGeometryCloner.h
*
* Description:
*
***************************************************************/

#ifndef _DO_GEOMETRY_CLONER_H_
#define _DO_GEOMETRY_CLONER_H_

// C++
#include <stdlib.h>

// Open scene graph
#include <osg/Group>
#include <osg/Switch>

// local includes
#include "DesignObjectBase.h"
#include "DOGeometryCollector.h"

#include "../Geometry/CAVEGroupShape.h"
#include "../Geometry/CAVEGeodeShape.h"


/***************************************************************
* Class: DOGeometryCloner
***************************************************************/
class DOGeometryCloner: public DesignObjectBase
{
  public:
    DOGeometryCloner();

    virtual void initDesignObjects() {}
    void setDOGeometryCollectorPtr(DOGeometryCollector *geomCollectorPtr);

    /* function called by 'DOGeometryEditor' */
    bool isToolkitEnabled() { return mEnabledFlag; }
    void setToolkitEnabled(bool flag) { mEnabledFlag = flag; }
    void pushClonedObjects();
    void popClonedObjects();

  protected:
    /* enabled flag: clone function is performed only when this flag is 'true' */
    bool mEnabledFlag;

    /* reference pointer of 'DOGeometryCollector' */
    DOGeometryCollector *mDOGeometryCollectorPtr;

    /* Temporary geode/geometry vector that tracks original objects that to be cloned, these vectors refer to
       the objects that selected before 'CLONE' state is activated */
    CAVEGeodeShapeVector mDupGeodeShapeVector;
    CAVEGeometryVector mDupGeometryVector;

};


#endif
