/***************************************************************
* File Name: TrackballController.h
*
* Class Name: TrackballController
*
***************************************************************/

#ifndef _TRACKBALL_CONTROLLER_H_
#define _TRACKBALL_CONTROLLER_H_

// C++
#include <iostream>

// Open scene graph
#include <osg/Vec3>
#include <osg/Matrixd>


/***************************************************************
* Class: TrackballController
***************************************************************/
class TrackballController
{
  public:
    TrackballController();

    void setAxis(const osg::Vec3 &axis) { mAxis = axis; }
    void setActive(bool flag) { mActiveFlag = flag; }
    void triggerInitialPick();
    void updateCtrPoint(const osg::Vec3 &ctrPoint);

    float getAngularOffset();

  protected:
    bool mActiveFlag, mInitialPickFlag;
    osg::Vec3 mAxis;
    osg::Vec3 mCtrPoint, mRefPoint;
};


#endif
