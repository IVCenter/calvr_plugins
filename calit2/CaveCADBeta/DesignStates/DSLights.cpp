/***************************************************************
* File Name: DSLights.cpp
*
* Description: 
*
* Written by Cathy Hughes 25 Sept 2012
*
***************************************************************/
#include "DSLights.h"

using namespace std;
using namespace osg;


// Constructor
DSLights::DSLights()
{
    CAVEAnimationModeler::ANIMCreateLights(&mFwdVec, &mBwdVec);
    for (int i = 0; i < mFwdVec.size(); ++i)
    {
        this->addChild(mFwdVec[i]);
        this->addChild(mBwdVec[i]);
    }

    // create both instances of intersector
    mDSIntersector = new DSIntersector();
    mDOIntersector = new DOIntersector();
    mDSIntersector->loadRootTargetNode(NULL, NULL);
    mDOIntersector->loadRootTargetNode(NULL, NULL);

    setAllChildrenOff();
    mDevPressedFlag = false;
    mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, mFwdVec[0]->getChild(0));

    prevGeode = NULL;
    mDrawingState = IDLE;
}


// Destructor
DSLights::~DSLights()
{

}


/***************************************************************
* Function: setObjectEnabled()
*
* Description:
*
***************************************************************/
void DSLights::setObjectEnabled(bool flag)
{
    mDrawingState = IDLE;

    mObjEnabledFlag = flag;
    AnimationPathCallback* animCallback = NULL;
    setAllChildrenOff();
    if (flag && !mIsOpen) // open menu
    {
        setSingleChildOn(0);
        for (int i = 0; i < mFwdVec.size(); ++i)
        {
            setChildValue(mFwdVec[i], true);
            animCallback = dynamic_cast <AnimationPathCallback*> (mFwdVec[i]->getUpdateCallback());
            mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, mFwdVec[i]->getChild(0));

            if (animCallback)
            {
                animCallback->reset();
            }
        }
        mIsOpen = true;
    }
    else // close menu
    {
        setSingleChildOn(0);
        for (int i = 1; i < mBwdVec.size(); ++i)
        {
            setChildValue(mBwdVec[i], true);
            animCallback = dynamic_cast <AnimationPathCallback*> (mBwdVec[i]->getUpdateCallback());
            mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, mBwdVec[i]);

            if (animCallback)
            {
                animCallback->reset();
            }
        }
        mIsOpen = false;
    }

    mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, mFwdVec[0]->getChild(0));
}


/***************************************************************
* Function: switchToPrevSubState()
***************************************************************/
void DSLights::switchToPrevSubState()
{
    if (mDrawingState == IDLE)
    {

    }
    else
    {

    }
}


/***************************************************************
* Function: switchToNextSubState()
***************************************************************/
void DSLights::switchToNextSubState()
{
    if (mDrawingState == IDLE)
    {

    }
    else
    {

    }
}


/***************************************************************
* Function: inputDevMoveEvent()
***************************************************************/
void DSLights::inputDevMoveEvent(const osg::Vec3 &pointerOrg, const osg::Vec3 &pointerPos)
{	
    if (mDevPressedFlag)
    {
        if (mDrawingState == PLACE_OBJECT && _activeObject)
        {
            osg::Matrixf invScale;
            invScale.makeScale(osg::Vec3(.001, .001, .001));
            osg::MatrixTransform * mt = new osg::MatrixTransform();
            mt->setMatrix(invScale);

            osg::Vec3 pos = osg::Vec3(0, 4000, 0) * cvr::PluginHelper::getHandMat()
                * cvr::PluginHelper::getWorldToObjectTransform();
            _activeObject->setPosition(pos);//pointerPos); 
            std::cout << pos[0] << " " << pos[1] << " " << pos[2] << std::endl;
        }
    }
    if (!mDevPressedFlag)
    {
        if (mDrawingState == PLACE_OBJECT && _activeObject)
        {
            osg::Matrixf invScale;
            invScale.makeScale(osg::Vec3(.001, .001, .001));
            osg::MatrixTransform * mt = new osg::MatrixTransform();
            mt->setMatrix(invScale);

            osg::Vec3 pos = osg::Vec3(0, 4000, 0) * cvr::PluginHelper::getHandMat()
                * cvr::PluginHelper::getWorldToObjectTransform() * invScale;
            _activeObject->setPosition(pos);//pointerPos); 
            std::cout << pos[0] << " " << pos[1] << " " << pos[2] << std::endl;
        }
    }
}


/***************************************************************
* Function: inputDevPressEvent()
***************************************************************/
bool DSLights::inputDevPressEvent(const osg::Vec3 &pointerOrg, const osg::Vec3 &pointerPos)
{
//std::cout << "ObjectPlacer press" << std::endl;
    mDevPressedFlag = true;

    for (int i = 0; i < mFwdVec.size(); ++i)
    {
        mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, mFwdVec[i]->getChild(0));
        //setChildValue(fwdVec[i], true);

        if (mDSIntersector->test(pointerOrg, pointerPos))
        {
            if (i > 0) // not the top menu item
            {
                osg::Box * box = new osg::Box(osg::Vec3(), .2);
                osg::Cone * cone = new osg::Cone(osg::Vec3(), .2, .2);
                osg::ShapeDrawable * sd;

                osg::Geode * geode = new osg::Geode(); 
                sd = new osg::ShapeDrawable(box);
                geode->addDrawable(sd);
                sd = new osg::ShapeDrawable(cone);
                geode->addDrawable(sd);

                geode->getOrCreateStateSet()->setRenderingHint(StateSet::TRANSPARENT_BIN);

                osg::LightSource * lightSource = new LightSource();
                osg::Light * light = new Light();

                light->setLightNum(3);
                light->setPosition(Vec4(0.0f, 0.0f, 0.0f, 1.0f));
                light->setDiffuse( Vec4(0.5f, 0.0f, 0.0f, 1.0f));
                light->setSpecular(Vec4(1.0f, 0.0f, 0.0f, 1.0f));
                light->setAmbient( Vec4(0.0f, 0.0f, 0.0f, 1.0f));
                light->setSpotExponent(50);
                light->setSpotCutoff(45);
                light->setDirection(osg::Vec3(0.0f, 0.0f, 1.0f));

                lightSource->setLight(light);
                lightSource->setLocalStateSetModes(StateAttribute::ON);
                osg::StateSet * stateset = gDesignObjectRootGroup->getOrCreateStateSet();
                lightSource->setStateSetModes(*stateset, StateAttribute::ON);

                _activeObject = new osg::PositionAttitudeTransform();

                _activeObject->addChild(lightSource);
                _activeObject->addChild(geode);
                //cvr::PluginHelper::getObjectsRoot()->addChild(_activeObject); 
                gDesignObjectRootGroup->addChild(_activeObject);
                mDrawingState = PLACE_OBJECT;
                return true;
            }
        }
    }

    mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, mFwdVec[0]->getChild(0));


    if (mDrawingState == PLACE_OBJECT)
    {
        if (_activeObject)
        {
            _activeObject = NULL;
            mDrawingState = IDLE;
        }
        return true;
    }
    return false;
}


/***************************************************************
* Function: inputDevReleaseEvent()
***************************************************************/
bool DSLights::inputDevReleaseEvent()
{
    mDevPressedFlag = false;

    if (mDrawingState == PLACE_OBJECT)
    {
        //mDrawingState = IDLE;
        return true;
    }
    return false;
}


/***************************************************************
* Function: update()
***************************************************************/
void DSLights::update()
{
}


/***************************************************************
* Function: setDesignObjectHandlerPtr()
***************************************************************/
void DSLights::setDesignObjectHandlerPtr(DesignObjectHandler *designObjectHandler)
{
    mDOGeometryCollector = designObjectHandler->getDOGeometryCollectorPtr();
    mDOGeometryCreator = designObjectHandler->getDOGeometryCreatorPtr();
}


/***************************************************************
* Function: DrawingStateTransitionHandle()
*
* Take actions on transition between drawing states
*
***************************************************************/
void DSLights::DrawingStateTransitionHandle(const DrawingState& prevState, const DrawingState& nextState)
{

}


void DSLights::setHighlight(bool isHighlighted, const osg::Vec3 &pointerOrg, const osg::Vec3 &pointerPos) 
{
    int idx = -1;
    mIsHighlighted = isHighlighted;

    mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, mSphereExteriorGeode);
    mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, mFwdVec[0]->getChild(0));
}
 
