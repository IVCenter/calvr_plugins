/***************************************************************
* File Name: DSVirtualEarth.h
*
* Description: Derived class from DesignStateBase
*
***************************************************************/

#ifndef _DS_VIRTUAL_EARTH_H_
#define _DS_VIRTUAL_EARTH_H_


// Local include
#include "DesignStateBase.h"
#include "../TrackballController.h"
#include "../VirtualScenicHandler.h"
#include "../AnimationModeler/ANIMVirtualEarth.h"


/***************************************************************
* Class: DSVirtualEarth
***************************************************************/
class DSVirtualEarth: public DesignStateBase
{
  public:
    DSVirtualEarth();

    /* virtual functions inherited from base class */
    virtual void setObjectEnabled(bool flag);
    virtual void switchToPrevSubState();
    virtual void switchToNextSubState();

    void inputDevMoveEvent(const osg::Vec3 &pointerOrg, const osg::Vec3 &pointerPos);
    bool inputDevPressEvent(const osg::Vec3 &pointerOrg, const osg::Vec3 &pointerPos);
    bool inputDevReleaseEvent();
    void update() {}
    void updateVSParameters(const osg::Vec3 &viewDir, const osg::Vec3 &viewPos);

    void setScenicHandlerPtr(VirtualScenicHandler *vsHandlerPtr) { mVirtualScenicHandler = vsHandlerPtr; }

    /* sub states definition */
    enum DSVirtualEarthState
    {
        SET_NULL = 0,
        SET_PIN,
        SET_TIME,
        SET_DATE
    };

  protected:

    /* descendents of 'this' Switch other than mPATransFwd and mPATransBwd */
    osg::MatrixTransform *mEclipticTrans, *mTiltAxisTrans, *mEquatorTrans;
    osg::MatrixTransform *mFixedPinIndicatorTrans, *mFixedPinTrans;
    osg::Switch *mEclipticSwitch, *mEquatorSwitch;
    osg::Geode *mEarthGeode, *mSeasonsMapGeode;

    DSVirtualEarthState mState;
    float mLongi, mLati, mTime, mDate;
    float mTimeOffset, mTimeLapseSpeed;
    osg::Matrixd mUnitspaceMat;

    TrackballController *mTrackballController;
    VirtualScenicHandler *mVirtualScenicHandler;

    void stateSwitchHandler();
    void updateEstimatedTime();

    void setFixedPinPos(const float &longi, const float &lati);
    void setFixedPinPos(const Vec3 &pinPos);
    void setPinIndicatorPos(const float &lati);
    void setEclipticPos(const float &date);
    void setEquatorPos(const float &t);

    /* space conversion functions: all input vectors are assumed to be normal vector */
    void transcoordWorldToEquator(const osg::Vec3 &world, osg::Vec3 &local);
    void transcoordWorldToEcliptic(const osg::Vec3 &world, osg::Vec3 &local);
    void transcoordEclipticToWorld(const osg::Vec3 &local, osg::Vec3 &world);
    void transcoordEquatorToWorld(const osg::Vec3 &local, osg::Vec3 &world);

    /* conversion between geographical info and coordinates */
    void longilatiToCartesian(const float &longi, const float &lati, osg::Vec3 &coord);
    void cartesianToLongilati(const osg::Vec3 &coord, float &longi, float &lati);

    static const float gDateOffset;
};


#endif
