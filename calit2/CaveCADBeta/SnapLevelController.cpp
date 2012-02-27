/***************************************************************
* File Name: SnapLevelController.cpp
*
* Description:  
*
* Written by ZHANG Lelin on Nov 22, 2010
*
***************************************************************/
#include "SnapLevelController.h"


using namespace std;
using namespace osg;

//Constructor
SnapLevelController::SnapLevelController(): mLevelIdx(2), mNumLevels(7), mMetrics(FEET)
{
    mSnapLengthLevels = new float[mNumLevels];
    mSnapLengthLevels[0] = 0.0508;
    mSnapLengthLevels[1] = 0.1524;
    mSnapLengthLevels[2] = 0.3048;
    mSnapLengthLevels[3] = 0.6096;
    mSnapLengthLevels[4] = 1.524;
    mSnapLengthLevels[5] = 3.048;
    mSnapLengthLevels[6] = 6.096;

    mSnapAngleLevels = new float[mNumLevels];
    mSnapAngleLevels[0] = 1 * M_PI / 180.f;
    mSnapAngleLevels[1] = 2 * M_PI / 180.f;
    mSnapAngleLevels[2] = 5 * M_PI / 180.f;
    mSnapAngleLevels[3] = 10 * M_PI / 180.f;
    mSnapAngleLevels[4] = 15 * M_PI / 180.f;
    mSnapAngleLevels[5] = 30 * M_PI / 180.f;
    mSnapAngleLevels[6] = 90 * M_PI / 180.f;

    mSnapScaleLevels = new float[mNumLevels];
    mSnapScaleLevels[0] = 0.05;
    mSnapScaleLevels[1] = 0.10;
    mSnapScaleLevels[2] = 0.20;
    mSnapScaleLevels[3] = 0.25;
    mSnapScaleLevels[4] = 0.50;
    mSnapScaleLevels[5] = 1.00;
    mSnapScaleLevels[6] = 2.00;
}


/***************************************************************
* Function: switchToLowerLevel()
***************************************************************/
void SnapLevelController::switchToLowerLevel()
{
    if (--mLevelIdx < 0) mLevelIdx = mNumLevels - 1;
}


/***************************************************************
* Function: switchToUpperLevel()
***************************************************************/
void SnapLevelController::switchToUpperLevel()
{
    if (++mLevelIdx >= mNumLevels) mLevelIdx = 0;
}


/***************************************************************
* Function: switchSnapMetrics()
***************************************************************/
void SnapLevelController::switchSnapMetrics()
{
    if (mMetrics == FEET)
    {
	mMetrics = METERS;

	mSnapLengthLevels[0] = 0.05;
	mSnapLengthLevels[1] = 0.10;
	mSnapLengthLevels[2] = 0.20;
	mSnapLengthLevels[3] = 0.50;
	mSnapLengthLevels[4] = 1.00;
	mSnapLengthLevels[5] = 2.00;
	mSnapLengthLevels[6] = 5.00;
    }
    else if (mMetrics == METERS)
    {
	mMetrics = FEET;

	mSnapLengthLevels[0] = 0.0508;
	mSnapLengthLevels[1] = 0.1524;
	mSnapLengthLevels[2] = 0.3048;
	mSnapLengthLevels[3] = 0.6096;
	mSnapLengthLevels[4] = 1.524;
	mSnapLengthLevels[5] = 3.048;
	mSnapLengthLevels[6] = 6.096;
    }
}


/***************************************************************
* Function: getSnappingLengthInfo()
***************************************************************/
const string SnapLevelController::getSnappingLengthInfo()
{
    if (mMetrics == FEET)
    {
	switch (mLevelIdx)
	{
	    case 0: return string("2 inch");
	    case 1: return string("6 inch");
	    case 2: return string("1 ft");
	    case 3: return string("2 ft");
	    case 4: return string("5 ft");
	    case 5: return string("10 ft");
	    case 6: return string("20 ft");
	    default: break;
	}
    }
    else if (mMetrics == METERS)
    {
	switch (mLevelIdx)
	{
	    case 0: return string("5 cm");
	    case 1: return string("10 cm");
	    case 2: return string("20 cm");
	    case 3: return string("50 cm");
	    case 4: return string("1 m");
	    case 5: return string("2 m");
	    case 6: return string("5 m");
	    default: break;
	}
    }
    return string("length info");
}


/***************************************************************
* Function: getSnappingAngleInfo()
***************************************************************/
const string SnapLevelController::getSnappingAngleInfo()
{
    switch (mLevelIdx)
    {
	case 0: return string("1 deg");
	case 1: return string("2 deg");
	case 2: return string("5 deg");
	case 3: return string("10 deg");
	case 4: return string("15 deg");
	case 5: return string("30 deg");
	case 6: return string("90 deg");
	default: break;
    }
    return string("angle info");
}


/***************************************************************
* Function: getSnappingScaleInfo()
***************************************************************/
const string SnapLevelController::getSnappingScaleInfo()
{
    switch (mLevelIdx)
    {
	case 0: return string("0.05");
	case 1: return string("0.10");
	case 2: return string("0.20");
	case 3: return string("0.25");
	case 4: return string("0.50");
	case 5: return string("1.00");
	case 6: return string("2.00");
	default: break;
    }
    return string("scale info");
}


/***************************************************************
* Static Function: getInitSnappingLength()
***************************************************************/
const float SnapLevelController::getInitSnappingLength() { return 0.3048f; }


/***************************************************************
* Static Function: getInitSnappingAngle()
***************************************************************/
const float SnapLevelController::getInitSnappingAngle() { return M_PI / 36.f; }


/***************************************************************
* Static Function: getInitSnappingScale()
***************************************************************/
const float SnapLevelController::getInitSnappingScale() { return 0.2f; }










