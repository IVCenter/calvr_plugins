/***************************************************************
* File Name: CAVEGeodeShape.h
*
* Description: Derived class from CAVEGeode
*
***************************************************************/

#ifndef _CAVE_GEODE_SHAPE_H_
#define _CAVE_GEODE_SHAPE_H_

// C++
#include <iostream>
#include <list>
#include <vector>

// Open scene graph
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Material>
#include <osg/StateSet>
#include <osg/Texture2D>
#include <osg/ShapeDrawable>

#include <osgDB/ReadFile>

// local
#include "CAVEGeodeSnapWireframe.h"
#include "CAVEGeometry.h"


class CAVEGeodeShape;
typedef std::vector<CAVEGeodeShape*>	CAVEGeodeShapeVector;
typedef std::vector<bool>		VertexMaskingVector;


/***************************************************************
* Class: CAVEGeodeShape
***************************************************************/
class CAVEGeodeShape: public CAVEGeode
{
    /* allow class 'DOGeometryCollector' to change its private vector index values */
    friend class DOGeometryCollector;

    /* allow class 'CAVEGroupIconSurface' to get access to geometry list and date */
    friend class CAVEGroupIconSurface;

  public:
    /* definition of two initial shape types */
    enum Type
    {
        BOX,
        CYLINDER
    };

    /***************************************************************
    * Class 'EditorInfo': information panel that keeps editting info
    ***************************************************************/
    class EditorInfo
    {
      public:

        enum ActiveTypeMasking
        {
            NONE = 0x00,
            MOVE = 0x01,
            ROTATE = 0x02,
            SCALE = 0x04,
            MANIPULATE = 0x08
        };

        void reset()
        {
            mTypeMasking = NONE;
            mMoveOffset = osg::Vec3(0, 0, 0);
            mRotateCenter = osg::Vec3(0, 0, 0);
            mRotateAxis = osg::Vec3(0, 0, 1);
            mRotateAngle = 0;
            mScaleCenter = osg::Vec3(0, 0, 0);
            mScaleVect = osg::Vec3(1, 1, 1);
        }

        /* default constructor */
        EditorInfo() { reset(); }

        /* parameter update functions */
        void setMoveUpdate(const osg::Vec3 &offset)
        {
            mMoveOffset = offset;
            mTypeMasking = MOVE;
        }

        void setRotateUpdate(const osg::Vec3 &center, const osg::Vec3 &axis, const float &angle)
        {
            mRotateCenter = center;
            mRotateAxis = axis;
            mRotateAngle = angle;
            mTypeMasking = ROTATE;
        }

        void setScaleUpdate(const osg::Vec3 &center, const osg::Vec3 &scalevect)
        {
            mScaleCenter = center;
            mScaleVect = scalevect;
            mTypeMasking = SCALE;
        }

        /* parameter access functions */
        const osg::Vec3 &getMoveOffset() { return mMoveOffset; }
        const osg::Vec3 &getRotateCenter() { return mRotateCenter; }
        const osg::Vec3 &getRotateAxis() { return mRotateAxis; }
        const osg::Vec3 &getScaleCenter() { return mScaleCenter; }
        const osg::Vec3 &getScaleVect() { return mScaleVect; }
        const float &getRotateAngle() { return mRotateAngle; }
        const ActiveTypeMasking &getTypeMasking() { return mTypeMasking; }

      protected:

        ActiveTypeMasking mTypeMasking;
        osg::Vec3 mMoveOffset;
        osg::Vec3 mRotateCenter, mRotateAxis;
        float mRotateAngle;
        osg::Vec3 mScaleCenter, mScaleVect;
    };

    /* 'CAVEGeodeShape' constructors & destructor */
    CAVEGeodeShape(const Type &typ, const osg::Vec3 &initVect, const osg::Vec3 &sVect);
    CAVEGeodeShape(CAVEGeodeShape *geodeShapeRef);
    ~CAVEGeodeShape();

    virtual void movedon() {}
    virtual void movedoff() {}
    virtual void pressed() {}
    virtual void released() {}

    /* update vertex masking vector based on selected geometries, or apply an existing vector with the same size */
    void updateVertexMaskingVector(bool flag);
    void updateVertexMaskingVector(const VertexMaskingVector &vertMaskingVector);
    void updateVertexMaskingVector();

    /* apply editting changes to vertex, normal & texcoord arrays */
    void applyEditorInfo(EditorInfo **infoPtr);
    void applyEditorInfo(EditorInfo **infoPtr, CAVEGeodeShape *refGeodePtr);

    const osg::Vec3 &getCAVEGeodeCenter() { return mCenterVect; }
    CAVEGeometryVector &getCAVEGeometryVector() { return mGeometryVector; }
    const VertexMaskingVector &getVertexMaskingVector() { return mVertexMaskingVector; }

    /* static function that implements 'EditorInfo' changes */
    static void applyEditorInfo(osg::Vec3Array **vertexArrayPtr, osg::Vec3Array **normalArrayPtr, 
				osg::Vec3Array **udirArrayPtr, osg::Vec3Array **vdirArrayPtr, 
				osg::Vec2Array **texcoordArrayPtr, 
				const osg::Vec3Array *refVertexArrayPtr, const osg::Vec3Array *refNormalArrayPtr, 
				const osg::Vec3Array *refUDirArrayPtr, const osg::Vec3Array *refVDirArrayPtr, 
				const osg::Vec2Array *refTexcoordArrayPtr, 
				const int &nVerts, EditorInfo **infoPtr, const VertexMaskingVector &vertMaskingVector);

    bool snapToVertex(const osg::Vec3 point, osg::Vec3 *ctr);
    void hideSnapBounds();


  protected:

    /* vector index that indicates the selection status of the Geode, ONLY accessed by 'DOGeometryCollector' */
    int mDOCollectorIndex;

    /* all CAVEGeometry objects share the same 'VertexArray', 'NormalArray' and 'TexcoordArray' 
       normaly each entry of 'mGeometryVector' contains one instance of CAVEGeometry */
    int mNumVertices, mNumNormals, mNumTexcoords;
    osg::Vec3Array* mVertexArray;
    osg::Vec3Array* mNormalArray;
    osg::Vec3Array* mUDirArray;
    osg::Vec3Array* mVDirArray;
    osg::Vec2Array* mTexcoordArray;
    CAVEGeometryVector mGeometryVector;
    
    std::vector<osg::Sphere*> mVertBoundingSpheres;
    std::map<osg::Sphere*, osg::ShapeDrawable*> mShapeDrawableMap;

    std::vector<osg::Cylinder*> mEdgeBoundingCylinder;
    std::map<osg::Cylinder*, osg::ShapeDrawable*> mEdgeDrawableMap;
    std::map<osg::Cylinder*, osg::Geode*>  mEdgeGeodeMap;
    /* center vector is normally the average of all vertices, which will be used for generating surface icons */
    osg::Vec3 mCenterVect;

    /* bool vector with the same size as 'mVertexArray' that indicates active editing states of each vertex */
    VertexMaskingVector mVertexMaskingVector;

    /* these functions are only called when the shape is being created and give no specific type 
      'sticker' of the shape, since it can be anything after modification. */
    void initGeometryBox(const osg::Vec3 &initVect, const osg::Vec3 &sVect);
    void initGeometryCylinder(const osg::Vec3 &initVect, const osg::Vec3 &sVect);

    /* actual side length of each texture image pattern in geometry */
    static const float gTextureTileSize;
};


#endif

