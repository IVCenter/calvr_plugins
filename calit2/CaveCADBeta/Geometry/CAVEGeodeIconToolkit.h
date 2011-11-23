/***************************************************************
* File Name: CAVEGeodeIconToolkit.h
*
* Description: Derived class from CAVEGeodeIcon
*
***************************************************************/

#ifndef _CAVE_GEODE_ICON_TOOLKIT_H_
#define _CAVE_GEODE_ICON_TOOLKIT_H_


// C++
#include <iostream>
#include <list>

// Open scene graph
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Material>
#include <osg/MatrixTransform>
#include <osg/ShapeDrawable>
#include <osg/StateSet>

// local
#include "CAVEGeodeIcon.h"


/***************************************************************
* Class: CAVEGeodeIconToolkit
***************************************************************/
class CAVEGeodeIconToolkit: public CAVEGeodeIcon
{
  public:

    /* types of editting operations */
    enum Type
    {
	MOVE,
	CLONE,
	ROTATE,
	MANIPULATE
    };

    /* constructors */
    CAVEGeodeIconToolkit();
    ~CAVEGeodeIconToolkit();

    virtual void movedon() {}
    virtual void movedoff() {}
    virtual void pressed();
    virtual void released();

    /* 'matTrans' it the 'MatrixTransform' node between 'CAVEGroupIconToolkit' and
       'CAVEGeodeIconToolkit', which decides the local rotation and offset of this
        geode. Set matrix value based on the specific type of 'CAVEGroupIconToolkit' */
    virtual void setMatrixTrans(osg::MatrixTransform *matTrans) = 0;

    const Type &getType() { return mType; }

  protected:

    /* basic type of toolkit icon */
    Type mType;

    /* static members that decides transparency of different color & texture states
      'gAlphaNormal': normal alpha value when 'CAVEGeodeIconToolkit' is activated
      'gAlphaSelected': alpha value when a specific 'CAVEGeodeIconToolkit' is being used
      'gAlphaUnselected': alpha value when a specific 'CAVEGeodeIconSurface' is not used
    */
    static const float gAlphaNormal;
    static const float gAlphaSelected;
    static const float gAlphaUnselected;

    static const osg::Vec3 gDiffuseColor;
    static const osg::Vec3 gSpecularColor;
};


/***************************************************************
* Class: CAVEGeodeIconToolkitMove
***************************************************************/
class CAVEGeodeIconToolkitMove: public CAVEGeodeIconToolkit
{
  public:

    /* enumeration of face orientations */
    enum FaceOrientation
    {
	FRONT_BACK,
	LEFT_RIGHT,
	UP_DOWN
    };

    CAVEGeodeIconToolkitMove() {}
    CAVEGeodeIconToolkitMove(const FaceOrientation &typ);

    const FaceOrientation &getFaceOrientation() { return mFaceOrient; }
    virtual void setMatrixTrans(osg::MatrixTransform *matTrans);

  protected:
    FaceOrientation mFaceOrient;

    void initBaseGeometry();

    /* osg geometry objects and parameters */
    osg::Cone *mConeUp, *mConeDown;
    osg::Cylinder *mCylinderUp, *mCylinderDown;
    osg::ShapeDrawable *mConeDrawableUp, *mCylinderDrawableUp, *mConeDrawableDown, *mCylinderDrawableDown;

    static const float gBodyRadius;
    static const float gBodyLength;
    static const float gArrowRadius;
    static const float gArrowLength;
};


/***************************************************************
* Class: CAVEGeodeIconToolkitClone
***************************************************************/
class CAVEGeodeIconToolkitClone: public CAVEGeodeIconToolkitMove
{
  public:
    CAVEGeodeIconToolkitClone(const FaceOrientation &typ);

  protected:
    osg::Cone *mExConeUp, *mExConeDown;
    osg::ShapeDrawable *mExConeDrawableUp, *mExConeDrawableDown;
};


/***************************************************************
* Class: CAVEGeodeIconToolkitRotate
***************************************************************/
class CAVEGeodeIconToolkitRotate: public CAVEGeodeIconToolkit
{
  public:

    /* enumeration of axis orientations */
    enum AxisAlignment
    {
	X_AXIS,
	Y_AXIS,
	Z_AXIS
    };

    CAVEGeodeIconToolkitRotate(const AxisAlignment &typ);

    const AxisAlignment &getAxisAlignment() { return mAxisAlignment; }
    virtual void setMatrixTrans(osg::MatrixTransform *matTrans);

  protected:
    AxisAlignment mAxisAlignment;

    /* osg geometry objects and parameters */
    osg::Cylinder *mPanelCylinder;
    osg::ShapeDrawable *mPanelDrawable;

    static const float gPanelRadius;
    static const float gPanelThickness;
};


/***************************************************************
* Class: CAVEGeodeIconToolkitManipulate
***************************************************************/
class CAVEGeodeIconToolkitManipulate: public CAVEGeodeIconToolkit
{
  public:

    /* enumeration of control point types */
    enum CtrlPtType
    {
	CORNER,
	EDGE,
	FACE
    };

    /* each manipulation control point is initialized with scaling direction */
    CAVEGeodeIconToolkitManipulate(const osg::Vec3 &scalingDir, const CtrlPtType &ctrlPtTyp);

    /* scaling direction is used as a hint to set offset values of the translation matrix on upper level */
    const osg::Vec3 &getScalingDir() { return mScalingDir; }

    /* called by 'DOGeometryEditor' as reference sizes to apply scaling function */
    const osg::Vec3 &getBoundingVect() { return mBoundingVect; }

    /* 'updateBoundingVect' is called before 'setMatrixTrans' to gurantee the right sizes of bounding box */
    void updateBoundingVect(const osg::Vec3 &boundingVect) { mBoundingVect = boundingVect; }
    virtual void setMatrixTrans(osg::MatrixTransform *matTrans); 

  protected:
    /* osg geometry objects and parameters */
    osg::Box *mCtrlPtBox;
    osg::ShapeDrawable *mCtrlPtBoxDrawable;

    /* directional vector of scaling, and bounding sizes */
    osg::Vec3 mScalingDir, mBoundingVect;

    CtrlPtType mCtrlPtType;

    /* radius of the manipulation control point */
    static const float gCtrlPtSize;
};


#endif



