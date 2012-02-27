/***************************************************************
* File Name: DOGeometryCollector.h
*
* Description:
*
***************************************************************/

#ifndef _DO_GEOMETRY_COLLECTOR_H_
#define _DO_GEOMETRY_COLLECTOR_H_


// C++
#include <iostream>
#include <string.h>

// Open scene graph
#include <osg/Group>
#include <osg/Vec3>

// local includes
#include "DesignObjectBase.h"

#include "../Geometry/CAVEGeodeIconSurface.h"
#include "../Geometry/CAVEGeodeIconToolkit.h"
#include "../Geometry/CAVEGeodeShape.h"
#include "../Geometry/CAVEGeometry.h"

#include "../AnimationModeler/ANIMGeometryCollector.h"


/***************************************************************
* Class: DOGeometryCollector
***************************************************************/
class DOGeometryCollector: public DesignObjectBase
{
    /* allow 'DOGeometryEditor' and 'DOGeometryCloner' to make geometry modifications to existing collections */
    friend class DOGeometryEditor;
    friend class DOGeometryCloner;

  public:
    DOGeometryCollector();

    virtual void initDesignObjects();

    /* called before geometry toggle functions: only during the first time when a 
      'CAVEGeodeShape' object is hit in 'DSGeometryCollector'. */
    void setSurfaceCollectionHints(CAVEGeodeShape *shapeGeode, const osg::Vec3 &iconCenter);

    /* collection function called by 'DSGeometryEditor' to update collection vector */
    void toggleCAVEGeodeShape(CAVEGeodeShape *shapeGeode);
    void toggleCAVEGeodeIconSurface(CAVEGeodeIconSurface *iconGeodeSurface);

    bool isGeodeShapeVectorEmpty();
    bool isGeometryVectorEmpty();
    void clearGeometryVector();
    void clearGeodeShapeVector();
    void setAllIconSurfaceHighlightsNormal();
    void setAllIconSurfaceHighlightsUnselected();

  protected:

    /* two level collection modes: only accessible by 'this' and 'DOGeometryEditor' */
    enum Mode
    {
	COLLECT_NONE,
	COLLECT_GEODE,
	COLLECT_GEOMETRY
    };
    Mode mMode;

    /* 'mShapeCenter' is the center position of the first hit 'CAVEGeodeShape' object.
      'mIconCenter' is the same as that of 'DesignStateBase', which is also center 
       for all 'CAVEGeodeIconToolkit' objects, and is the reference origin point that
      'CAVEGeodeIconSurface' objects are generated. */
    osg::Vec3 mShapeCenter, mIconCenter;

    /* reference switch for 'CAVEGeodeShape' objects. The switch is in parallel with 'mDOShapeSwitch'
       but only turned on within editting state 'START_EDITTING' */
    osg::Switch *mDOShapeRefSwitch;
    osg::MatrixTransform *mDOShapeRefMatrixTrans;

    /* reference switch for 'CAVEGroupEditWireframe' objects. */
    osg::Switch *mDOGeodeWireframeSwitch;
    osg::Switch *mDOGeometryWireframeSwitch;

    /* 'mGroupIconSurfaceVector' is in parallel with 'mGeodeShapeVector', each 'CAVEGroupIconSurface' in
       the vector holds all surfaces related to one 'CAVEGeodeShape'. The two vectors have the same sizes. */
    CAVEGroupIconSurfaceVector mGroupIconSurfaceVector;

    /* two types of wireframe vectors that used in Geode level and Geometry level editting */
    CAVEGroupEditGeodeWireframeVector mEditGeodeWireframeVector;
    CAVEGroupEditGeometryWireframeVector mEditGeometryWireframeVector;

    /* Storage system of collector works both in Geode level and Geometry level
       since 'CAVEGroupIconSurface' objects under 'mDOIconSurfaceSwitch' and 'CAVEGeodeShape' objects 
       in 'mGeodeShapeVector' have the same index sequence and same size, surface groups can be tracked
       with reference to index number of 'CAVEGeodeShape' */
    CAVEGeodeShapeVector mGeodeShapeVector;
    CAVEGeometryVector mGeometryVector;

    /* private references to geometry collection: only accessed by friend class 'DOGeometryEditor' */
    CAVEGroupIconSurfaceVector &getGroupIconSurfaceVector() { return mGroupIconSurfaceVector; }
    CAVEGroupEditGeodeWireframeVector &getEditGeodeWireframeVector() { return mEditGeodeWireframeVector; }
    CAVEGroupEditGeometryWireframeVector &getEditGeometryWireframeVector() { return mEditGeometryWireframeVector; }

    CAVEGeodeShapeVector &getGeodeShapeVector() { return mGeodeShapeVector; }
    CAVEGeometryVector &getGeometryVector() { return mGeometryVector; }

    /* update vertex masking vector for each 'CAVEGeodeShape' according to current selection,
       this function is called after 'mMode' is modified or in toggle functions. */
    void updateVertexMaskingVector();
};


#endif


