/***************************************************************
* File Name: DOGeometryEditor.h
*
* Description:
*
***************************************************************/

#ifndef _DO_GEOMETRY_EDITOR_H_
#define _DO_GEOMETRY_EDITOR_H_

// C++
#include <stdlib.h>

// Open scene graph
#include <osg/Group>
#include <osg/Switch>

// local includes
#include "DesignObjectBase.h"
#include "DOGeometryCollector.h"
#include "DOGeometryCloner.h"

#include "../Geometry/CAVEGroupShape.h"
#include "../Geometry/CAVEGeodeShape.h"
#include "../Geometry/CAVEGeodeEditWireframe.h"

#include "../SnapLevelController.h"

#include "../AnimationModeler/ANIMGeometryEditor.h"


/***************************************************************
* Class: DOGeometryEditor
***************************************************************/
class DOGeometryEditor: public DesignObjectBase
{
  public:
    DOGeometryEditor();

    virtual void initDesignObjects();

    /* Function 'setToolkitEnabled()': reset forward/backward animation of toolkit objects */
    void setToolkitEnabled(bool flag);

    /* Function 'setSubToolkitEnabled()': switch on/off sub editing states indexed with 'mToolkitActiveIdx' */
    void setSubToolkitEnabled(bool flag);

    /* functions calls controlled by DSGeometryEditor substate switch actions */
    void setPrevToolkit();
    void setNextToolkit();
    void setScalePerUnit(SnapLevelController **snapLevelControllerRefPtr);

    void setSnapStarted(const osg::Vec3 &pos);
    void setSnapUpdate(const osg::Vec3 &pos);
    void setSnapFinished();

    void setPointerDir(const osg::Vec3 &pointerDir);
    void setActiveIconToolkit(CAVEGeodeIconToolkit *iconToolkit);
    void setDOGeometryCollectorPtr(DOGeometryCollector *geomCollectorPtr);
    void setDOGeometryClonerPtr(DOGeometryCloner *geomClonerPtr);

  protected:
    /* actual length, rotational angle and scale value represented by each snapping grid */
    float mLengthPerUnit, mAnglePerUnit, mScalePerUnit;
    std::string mLengthPerUnitInfo, mAnglePerUnitInfo, mScalePerUnitInfo;

    osg::Vec3 mInitSnapPos, mInitRadiusVector;
    CAVEGeodeShape::EditorInfo *mEditorInfo;

    /* root matrix transform under 'mDOIconToolkitSwitch' that handles offset of toolkit 
       to central position of design sphere */
    osg::MatrixTransform *mMatrixTrans;

    /* total number of types of editing toolkits, active index and switch entry arrays, active icon pointer */
    int mNumToolkits, mToolkitActiveIdx;		
    CAVEAnimationModeler::ANIMIconToolkitSwitchEntry **mIconToolkitSwitchEntryArray;
    CAVEGeodeIconToolkit *mActiveIconToolkit;

    /* reference pointer of 'DOGeometryCollector' and 'DOGeometryCloner' */
    DOGeometryCollector *mDOGeometryCollectorPtr;
    DOGeometryCloner *mDOGeometryClonerPtr;

    /* function calls within 'this' class that take effect of geometry changes */
    void applyTranslation(const osg::Vec3 &offset);
    void applyRotation(const osg::Vec3 &axis, const osg::Vec3 &offset);
    void applyManipulation(const osg::Vec3 &dir, const osg::Vec3 &bound, const osg::Vec3 &offset);
    void applyGridUnitUpdates();
    void applyEditingUpdates();
    void applyEditingFinishes();
};


#endif















