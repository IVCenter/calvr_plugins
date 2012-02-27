/***************************************************************
* File Name: DSVirtualEarth.cpp
*
* Description: 
*
* Written by ZHANG Lelin on Oct 5, 2010
*
***************************************************************/
#include "DSVirtualEarth.h"

using namespace std;
using namespace osg;


const float DSVirtualEarth::gDateOffset(0.551f);	// mapped to Jun 20 of the year


//Constructor
DSVirtualEarth::DSVirtualEarth(): mState(SET_NULL), mLongi(-117.17f), mLati(32.75f), mTime(8.f), mDate(0.551f),
				  mTimeOffset(0.0f), mTimeLapseSpeed(0.0f)
{
    mEclipticTrans = new MatrixTransform();
    mTiltAxisTrans = new MatrixTransform();
    mEquatorTrans = new MatrixTransform();
    mEclipticSwitch = new Switch;
    mEquatorSwitch = new Switch;

    addChild(mEclipticTrans);
    mEclipticTrans->addChild(mEclipticSwitch);
    mEclipticTrans->addChild(mTiltAxisTrans);
    mTiltAxisTrans->addChild(mEquatorTrans);
    mEquatorTrans->addChild(mEquatorSwitch);

    Matrixd tiltaxisMat;
    float angle = 23.5f / 180.0f * M_PI;
    tiltaxisMat.makeRotate(Vec3(0, 0, 1), Vec3(sin(angle), 0, cos(angle)));
    mTiltAxisTrans->setMatrix(tiltaxisMat);

    /* Level 1: Seasons map and Ecliptic ruler */
    Switch *thisPtr = dynamic_cast <Switch*> (this);
    CAVEAnimationModeler::ANIMLoadVirtualEarthReferenceLevel(&thisPtr, &mSeasonsMapGeode);

    /* Level 2: Wired sphere, Meridian ruler, earth axis, equator ruler*/
    CAVEAnimationModeler::ANIMLoadVirtualEarthEclipticLevel(&mEclipticSwitch);

    /* Level 3: Wired sphere, fwd & bwd animation, pin indicator, fixed pin trans */
    CAVEAnimationModeler::ANIMLoadVirtualEarthEquatorLevel(&mEquatorSwitch, &mEarthGeode, 
	&mPATransFwd, &mPATransBwd, &mFixedPinIndicatorTrans, &mFixedPinTrans);

    /* set initial geographical info/ time / date */
    setFixedPinPos(mLongi, mLati);
    setEclipticPos(mDate);
    setEquatorPos(mTime);
    setPinIndicatorPos(mLongi);
    setAllChildrenOff();

    /* transform matrix that flip shader world plain to vertical earth model */
    mUnitspaceMat = Matrixd::inverse(Matrixd( 0,  0, -1,  0, -1,  0,  0,  0,		
			     		      0,  1,  0,  0,  0,  0,  0,  1));

    /* create instance of intersector */
    mDSIntersector = new DSIntersector();
    mTrackballController = new TrackballController();
    mVirtualScenicHandler = NULL; 
}


/***************************************************************
* Function: setObjectEnabled()
*
* Description:
*
***************************************************************/
void DSVirtualEarth::setObjectEnabled(bool flag)
{
    mObjEnabledFlag = flag;
    mDSParticleSystemPtr->setEmitterEnabled(flag);

    /* set geometry switches when state is enabled / diabled */
    if (flag) stateSwitchHandler();
    else 
    {
	mState = SET_NULL;
	this->setSingleChildOn(0);
	mEclipticSwitch->setAllChildrenOff();
	mTrackballController->setActive(false);
	mDSIntersector->loadRootTargetNode(NULL, NULL);
    }
    if (!mPATransFwd || !mPATransBwd) return;

    AnimationPathCallback* animCallback = NULL;
    if (flag)
    {
	mEquatorSwitch->setSingleChildOn(1);	//  child #1: Load Forward Animation
	animCallback = dynamic_cast <AnimationPathCallback*> (mPATransFwd->getUpdateCallback());
    } else {
	mEquatorSwitch->setSingleChildOn(2);	//  child #2: Load Backward Animation
	animCallback = dynamic_cast <AnimationPathCallback*> (mPATransBwd->getUpdateCallback());
    }
    if (animCallback) animCallback->reset();
}


/***************************************************************
* Function: switchToPrevSubState()
***************************************************************/
void DSVirtualEarth::switchToPrevSubState()
{
    /* prev state look up */
    switch (mState)
    {
	case SET_NULL: mState = SET_DATE; break;
	case SET_PIN:  mState = SET_NULL; break;
	case SET_TIME: mState = SET_PIN;  break;
	case SET_DATE: mState = SET_TIME; break;
	default: break;
    }
    stateSwitchHandler();
    mDSParticleSystemPtr->setEmitterEnabled(false);
}


/***************************************************************
* Function: switchToNextSubState()
***************************************************************/
void DSVirtualEarth::switchToNextSubState()
{
    /* next state look up */
    switch (mState)
    {
	case SET_NULL: mState = SET_PIN; break;
	case SET_PIN:  mState = SET_TIME; break;
	case SET_TIME: mState = SET_DATE;  break;
	case SET_DATE: mState = SET_NULL; break;
	default: break;
    }
    stateSwitchHandler();
    mDSParticleSystemPtr->setEmitterEnabled(false);
}


/***************************************************************
* Function: inputDevMoveEvent()
*
* Description: This function is called within every 'preFrame'
*
***************************************************************/
void DSVirtualEarth::inputDevMoveEvent(const osg::Vec3 &pointerOrg, const osg::Vec3 &pointerPos)
{
    if (mDevPressedFlag)
    {
	if (mDSIntersector->test(pointerOrg, pointerPos))
	{
	    /* use 'hitNormalWorld' as input to track ball controller in order to avoid shaking effects */
	    Vec3 hitNormalWorld = mDSIntersector->getWorldHitNormal();
	    mTrackballController->updateCtrPoint(hitNormalWorld);
	    float offset = mTrackballController->getAngularOffset();

	    /* translate offset value to time & date changes and apply them to earth object */
	    if (mState == SET_PIN)
	    {
		/* convert world hit normal to localized vector before setting pin position */
		Vec3 hitNormalLocal;
		transcoordWorldToEquator(hitNormalWorld, hitNormalLocal);

		setFixedPinPos(hitNormalLocal * ANIM_VIRTUAL_SPHERE_RADIUS);
		cartesianToLongilati(hitNormalLocal * ANIM_VIRTUAL_SPHERE_RADIUS, mLongi, mLati);
		setPinIndicatorPos(mLongi);
		updateEstimatedTime();
	    }
	    else if (mState == SET_TIME)
	    {
		mTimeLapseSpeed =  offset * 12.f / M_PI;
		mTime += mTimeLapseSpeed;
		setEquatorPos(mTime);
	    }
	    else if (mState == SET_DATE)
	    {
		mDate += offset / (2 * M_PI);
		setEclipticPos(mDate);

		updateEstimatedTime();
	    }
	}

    } else {

	/* update self rotation around earth axis */
	mTime += mTimeLapseSpeed;
	setEquatorPos(mTime);
    } 
}


/***************************************************************
* Function: inputDevPressEvent()
***************************************************************/
bool DSVirtualEarth::inputDevPressEvent(const osg::Vec3 &pointerOrg, const osg::Vec3 &pointerPos)
{
    mDevPressedFlag = true;
    mTrackballController->triggerInitialPick();

    /* switch on instant highlight geometries */
    if (mState == SET_TIME)
    {
	mEquatorSwitch->setValue(0, true);
	mEquatorSwitch->setValue(3, true);
    }
    else if (mState == SET_DATE)
    {
	mEclipticSwitch->setValue(0, true);
	mEclipticSwitch->setValue(1, true);
    }

    if (mState == SET_NULL) return false;
    else return true;
}


/***************************************************************
* Function: inputDevReleaseEvent()
***************************************************************/
bool DSVirtualEarth::inputDevReleaseEvent()
{
    mDevPressedFlag = false;

    if (mState == SET_TIME)
    {
	mEquatorSwitch->setValue(0, false);
	mEquatorSwitch->setValue(3, false);
    }
    else if (mState == SET_DATE)
    {
	mEclipticSwitch->setValue(0, false);
	mEclipticSwitch->setValue(1, false);
    }

    if (mState == SET_NULL) return false;
    else return true;
}


/***************************************************************
* Function: updateVSParameters()
***************************************************************/
void DSVirtualEarth::updateVSParameters(const Vec3 &viewDir, const Vec3 &viewPos)
{
    if (!mVirtualScenicHandler) return;

    /*  compute sun direction in world space: apply transforms resulted by viewer's orientation change, 
	guarantee that from the viewer's position, the virtual earth is always half illuminated. */
    Matrixd baserotMat;
    baserotMat.makeRotate(Vec3(0, 1, 0), gDesignStateFrontVect);
    Vec3 sunDirWorld = (CAVEAnimationModeler::ANIMVirtualEarthLightDir()) * baserotMat;
    StateSet *stateset = mEarthGeode->getStateSet();
    if (stateset)
    {
	Uniform *lightposUniform = stateset->getOrCreateUniform("LightPos", Uniform::FLOAT_VEC4);
	lightposUniform->set(Vec4(sunDirWorld, 0.0));
    }

    /* compute matrix combination that transforms a vector from shader space into world space */
    Matrixd latiMat;
    latiMat.makeRotate(mLati / 180.f * M_PI, Vec3(0, 1, 0));
    Matrixd equatorMat;   
    equatorMat.makeRotate((mTimeOffset / 12.f + mLongi / 180.f) * M_PI, Vec3(0, 0, 1));  
    Matrixd tiltaxisMat = mTiltAxisTrans->getMatrix();
    Matrixd eclipticMat = mEclipticTrans->getMatrix();
    Matrixd transMat = mUnitspaceMat * latiMat * equatorMat * tiltaxisMat * eclipticMat * baserotMat;

    /* updata environment rendering by passing parameters to VirtualScenicHandler */
    mVirtualScenicHandler->updateVSParameters(transMat, sunDirWorld, viewDir, viewPos);
}


/***************************************************************
* Function: stateSwitchHandler()
*
* This function adapts geometries when new state is changed to
* by setting children of the following two switches:
*
*				I-Test -0--1--2--3-
* 'this' Switch
* child #0: Ecliptic trans		Y  Y  Y  Y
* child #1: Seasons map			N  N  N  Y
* child #2: Ecliptic ruler		N  N  N  Y	
*
* mEclipticSwitch:
* child #0: Wired geode 	  T3	-  -  -  -
* child #1: Season indicator	  T3	-  -  -  -
* child #2: Earth axis			N  N  Y  Y
* child #3: Equator ruler		N  N  Y  N
*
* mEquatorSwitch:
* child #0: Wired geode	  	  T2	-  -  -  -
* child #1: Forward animation		Y  Y  Y  Y
* child #2: Backward animation		-  -  -  -
* child #3: Fixed pin indicator	  T2	-  -  -  -
* child #4: Fixed pin trans		N  Y  Y  Y
*
***************************************************************/
void DSVirtualEarth::stateSwitchHandler()
{
    /* switch on/off visible geometries' */
    switch (mState)
    {
	case SET_NULL:
	{
	    this->setSingleChildOn(0);
	    mEclipticSwitch->setAllChildrenOff();
	    mEquatorSwitch->setSingleChildOn(1);
	    break;
	}
	case SET_PIN:
	{
	    this->setSingleChildOn(0);
	    mEclipticSwitch->setAllChildrenOff();
	    mEquatorSwitch->setAllChildrenOff();
	    mEquatorSwitch->setValue(1, true);
	    mEquatorSwitch->setValue(4, true);
	    break;
	}
	case SET_TIME:
	{
	    this->setSingleChildOn(0);
	    mEclipticSwitch->setAllChildrenOff();
	    mEclipticSwitch->setValue(2, true);
	    mEclipticSwitch->setValue(3, true);
	    mEquatorSwitch->setAllChildrenOff();
	    mEquatorSwitch->setValue(1, true);
	    mEquatorSwitch->setValue(4, true);
	    break;
	}
	case SET_DATE:
	{
	    this->setAllChildrenOn();
	    mEclipticSwitch->setAllChildrenOff();
	    mEclipticSwitch->setValue(2, true);
	    mEquatorSwitch->setAllChildrenOff();
	    mEquatorSwitch->setValue(1, true);
	    mEquatorSwitch->setValue(4, true);
	    break;
	}
	default: break;
    }

    /* reset intersection properties & trackball axis */
    if (mState == SET_NULL)
    {
	mDSIntersector->loadRootTargetNode(NULL, NULL);

	mTrackballController->setActive(false);
    }
    else if (mState == SET_PIN)
    {
	mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, mEarthGeode);

	mTrackballController->setActive(false);
    }
    else if (mState == SET_TIME)
    {
	mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, mEarthGeode);

	Vec3 equatorAxis;
	transcoordEquatorToWorld(Vec3(0, 0, 1), equatorAxis);
	mTrackballController->setAxis(equatorAxis);
	mTrackballController->setActive(true);
    }
    else if (mState == SET_DATE)
    {
	mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, mSeasonsMapGeode);

	Vec3 eclipticAxis;
	transcoordEclipticToWorld(Vec3(0, 0, 1), eclipticAxis);
	mTrackballController->setAxis(eclipticAxis);
	mTrackballController->setActive(true);
    }
}


/***************************************************************
* Function: updateEstimatedTime()
*
* Description: This function updates 'mTime' use existing state
* of 'mDate', 'mLongi' and 'mLati'
*
***************************************************************/
void DSVirtualEarth::updateEstimatedTime()
{
    /* get virtual time based on longi/lati/date */
    float vtime = (mLongi + 180.f) / 15.f + 12 + (mDate - gDateOffset) * 24 + mTimeOffset;
    mTime = vtime - (int)(vtime / 24.f) * 24.f;
}


/***************************************************************
* Function: setFixedPinPos()
***************************************************************/
void DSVirtualEarth::setFixedPinPos(const float &longi, const float &lati)
{
    Vec3 pinPos;
    longilatiToCartesian(longi, lati, pinPos);

    Matrixd transMat, rotMat;
    transMat.makeTranslate(pinPos);
    rotMat.makeRotate(Vec3(0, 0, 1), pinPos);
    mFixedPinTrans->setMatrix(rotMat * transMat);
}


void DSVirtualEarth::setFixedPinPos(const Vec3 &pinPos)
{
    Matrixd transMat, rotMat;
    transMat.makeTranslate(pinPos);
    rotMat.makeRotate(Vec3(0, 0, 1), pinPos);
    mFixedPinTrans->setMatrix(rotMat * transMat);
}


/***************************************************************
* Function: setPinIndicatorPos()
***************************************************************/
void DSVirtualEarth::setPinIndicatorPos(const float &lati)
{
    Matrixd rotMat;
    rotMat.makeRotate(lati * M_PI / 180.0f, Vec3(0, 0, 1));
    mFixedPinIndicatorTrans->setMatrix(rotMat);
}


/***************************************************************
* Function: setEclipticPos()
***************************************************************/
void DSVirtualEarth::setEclipticPos(const float &date)
{
    Matrixd rotMat;
    rotMat.makeRotate((date - gDateOffset) * M_PI * 2, Vec3(0, 0, 1));
    mEclipticTrans->setMatrix(rotMat);
}


/***************************************************************
* Function: setEquatorPos()
***************************************************************/
void DSVirtualEarth::setEquatorPos(const float &t)
{
    /* get virtual time based on longi/lati/date */
    float vtime = (mLongi + 180.f) / 15.f + 12 + (mDate - gDateOffset) * 24 + mTimeOffset;
    vtime = vtime - (int)(vtime / 24.f) * 24.f;

    mTimeOffset += (t - vtime);

    Matrixd rotMat;   
    rotMat.makeRotate(mTimeOffset / 12.f * M_PI, Vec3(0, 0, 1));
    mEquatorTrans->setMatrix(rotMat);
}


/***************************************************************
* Function: transcoordWorldToEquator()
*
* Description: Using inverse matrix combinations to transform
* either normal or position from world space to equator space.
*
***************************************************************/
void DSVirtualEarth::transcoordWorldToEquator(const osg::Vec3 &world, osg::Vec3 &local)
{
    Matrixd eclipticMat = mEclipticTrans->getMatrix();
    Matrixd tiltaxisMat = mTiltAxisTrans->getMatrix();
    Matrixd equatorMat = mEquatorTrans->getMatrix();
    Matrixd invMat = Matrixd::inverse(equatorMat * tiltaxisMat * eclipticMat * gDesignStateBaseRotMat);

    local = world * invMat;
}


/***************************************************************
* Function: transcoordWorldToEcliptic()
*
* Description: Using inverse matrix combinations to transform
* either normal or position from world space to ecliptic space.
*
***************************************************************/
void DSVirtualEarth::transcoordWorldToEcliptic(const osg::Vec3 &world, osg::Vec3 &local)
{
    Matrixd eclipticMat = mEclipticTrans->getMatrix();
    Matrixd invMat = Matrixd::inverse(eclipticMat * gDesignStateBaseRotMat);

    local = world * invMat;
}


/***************************************************************
* Function: transcoordEclipticToWorld()
***************************************************************/
void DSVirtualEarth::transcoordEclipticToWorld(const osg::Vec3 &local, osg::Vec3 &world)
{
    world = local * gDesignStateBaseRotMat;
}


/***************************************************************
* Function: transcoordEquatorToWorld()
***************************************************************/
void DSVirtualEarth::transcoordEquatorToWorld(const osg::Vec3 &local, osg::Vec3 &world)
{
    Matrixd eclipticMat = mEclipticTrans->getMatrix();
    Matrixd tiltaxisMat = mTiltAxisTrans->getMatrix();

    world = local * tiltaxisMat * eclipticMat * gDesignStateBaseRotMat;
}


/***************************************************************
* Function: longilatiToCartesian()
***************************************************************/
void DSVirtualEarth::longilatiToCartesian(const float &longi, const float &lati, Vec3 &coord)
{
    const float rad = ANIM_VIRTUAL_SPHERE_RADIUS;
    const float theta = lati / 180.f * M_PI;
    coord.x() = - cos(longi / 180.f * M_PI) * rad * cos(theta);
    coord.y() = - sin(longi / 180.f * M_PI) * rad * cos(theta);
    coord.z() = sin(theta) * rad;
}


/***************************************************************
* Function: cartesianToLongilati()
***************************************************************/
void DSVirtualEarth::cartesianToLongilati(const Vec3 &coord, float &longi, float &lati)
{
    const float rad = ANIM_VIRTUAL_SPHERE_RADIUS;
    longi = acos(-coord.x() / sqrt(coord.x() * coord.x() + coord.y() * coord.y())) / M_PI * 180.f;
    if (coord.y() > 0) longi = - longi;
    lati = asin(coord.z() / rad) / M_PI * 180.f;
}


























