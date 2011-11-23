/***************************************************************
* File Name: SnapLevelController.h
*
* Class Name: SnapLevelController
*
***************************************************************/

#ifndef _SNAP_LEVEL_CONTROLLER_H_
#define _SNAP_LEVEL_CONTROLLER_H_


// C++
#include <iostream>
#include <list>
#include <string>

// Open scene graph
#include <osg/Vec3>


/***************************************************************
* Class: SnapLevelController
***************************************************************/
class SnapLevelController
{
  public:
    SnapLevelController();

    void switchToLowerLevel();
    void switchToUpperLevel();
    void switchSnapMetrics();

    inline const float &getSnappingLength() { return mSnapLengthLevels[mLevelIdx]; }
    const std::string getSnappingLengthInfo();

    inline const float &getSnappingAngle() { return mSnapAngleLevels[mLevelIdx]; }
    const std::string getSnappingAngleInfo();

    inline const float &getSnappingScale() { return mSnapScaleLevels[mLevelIdx]; }
    const std::string getSnappingScaleInfo();

    static const float getInitSnappingLength();
    static const float getInitSnappingAngle();
    static const float getInitSnappingScale();

    /* metrics used for snapping */
    enum SnapMetrics
    {
	METERS,
	FEET
    };

  protected:
    int mLevelIdx, mNumLevels;
    float *mSnapLengthLevels, *mSnapAngleLevels, *mSnapScaleLevels;
    SnapMetrics mMetrics;
};


#endif




