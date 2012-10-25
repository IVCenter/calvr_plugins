/***************************************************************
* File Name: DSGeometryCreator.cpp
*
* Description: 
*
* Written by ZHANG Lelin on Nov 10, 2010
*
***************************************************************/
#include "DSGeometryCreator.h"

using namespace std;
using namespace osg;


// Constructor
DSGeometryCreator::DSGeometryCreator(): mShapeSwitchIdx(0), mNumShapeSwitches(0), mDrawingState(IDLE),
					mAudioConfigHandler(NULL)
{
    CAVEAnimationModeler::ANIMLoadGeometryCreator(&mPATransFwd, &mPATransBwd, &mSphereExteriorSwitch, &mSphereExteriorGeode,
							mNumShapeSwitches, &mShapeSwitchEntryArray);
    this->addChild(mPATransFwd);
    this->addChild(mPATransBwd);

    // create both instances of intersector
    mDSIntersector = new DSIntersector();
    mDOIntersector = new DOIntersector();
    mDSIntersector->loadRootTargetNode(NULL, NULL);
    mDOIntersector->loadRootTargetNode(NULL, NULL);

    mSnapLevelController = new SnapLevelController();

    setAllChildrenOff();
    mDevPressedFlag = false;

    prevGeode = NULL;
}


// Destructor
DSGeometryCreator::~DSGeometryCreator()
{

}


/***************************************************************
* Function: setObjectEnabled()
*
* Description:
*
***************************************************************/
void DSGeometryCreator::setObjectEnabled(bool flag)
{
    mObjEnabledFlag = flag;
    mDrawingState = IDLE;

    if (flag) 
        setAllChildrenOn();

    if (!mPATransFwd || !mPATransBwd) 
        return;

    AnimationPathCallback* animCallback = NULL;
    if (flag)
    {
        if (!mIsOpen)
        {
            this->setSingleChildOn(0);
            animCallback = dynamic_cast <AnimationPathCallback*> (mPATransFwd->getUpdateCallback());

            // load intersection root and targets when state is enabled, no need to change till disabled 
            mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, mSphereExteriorGeode);
            mDOIntersector->loadRootTargetNode(gDesignObjectRootGroup, NULL);
            
            for (int i = 0; i < mNumShapeSwitches; ++i)
            {
                mShapeSwitchEntryArray[i]->mSwitch->setSingleChildOn(0);
                mShapeSwitchEntryArray[i]->mFlipUpFwdAnim->reset();
            }
            mIsOpen = true;
        }
        else 
        {
            for (int i = 0; i < mNumShapeSwitches; ++i)
            {
                mShapeSwitchEntryArray[i]->mSwitch->setSingleChildOn(2);
                mShapeSwitchEntryArray[i]->mFlipUpBwdAnim->reset();
            }
            mIsOpen = false;
            mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, mSphereExteriorGeode);
        }
    } 
    else 
    {
        for (int i = 0; i < mNumShapeSwitches; ++i)
        {
            mShapeSwitchEntryArray[i]->mSwitch->setSingleChildOn(2);
            mShapeSwitchEntryArray[i]->mFlipUpBwdAnim->reset();
        }
        mIsOpen = false;
        mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, mSphereExteriorGeode);
    }

    mDrawingState = IDLE;
    if (animCallback) 
        animCallback->reset();
}


/***************************************************************
* Function: switchToPrevSubState()
***************************************************************/
void DSGeometryCreator::switchToPrevSubState()
{
    if (mDrawingState == IDLE)
    {
        mShapeSwitchEntryArray[mShapeSwitchIdx]->mSwitch->setSingleChildOn(3);
        mShapeSwitchEntryArray[mShapeSwitchIdx]->mFlipDownBwdAnim->reset();

        if (--mShapeSwitchIdx < 0) 
        {
            mShapeSwitchIdx = mNumShapeSwitches - 1;
        }

        mShapeSwitchEntryArray[mShapeSwitchIdx]->mSwitch->setSingleChildOn(2);
        mShapeSwitchEntryArray[mShapeSwitchIdx]->mFlipUpBwdAnim->reset();

        mDOGeometryCreator->setWireframeActiveID(-1);
        mDOGeometryCreator->setResize(0.0f);
    }
    else
    {
        mSnapLevelController->switchToUpperLevel();
        mDOGeometryCreator->setScalePerUnit(mSnapLevelController->getSnappingLength(),
                            mSnapLevelController->getSnappingLengthInfo());
    }
}


/***************************************************************
* Function: switchToNextSubState()
***************************************************************/
void DSGeometryCreator::switchToNextSubState()
{
    if (mDrawingState == IDLE)
    {
        mShapeSwitchEntryArray[mShapeSwitchIdx]->mSwitch->setSingleChildOn(1);
        mShapeSwitchEntryArray[mShapeSwitchIdx]->mFlipDownFwdAnim->reset();

        if (++mShapeSwitchIdx >= mNumShapeSwitches) 
        {
            mShapeSwitchIdx = 0;
        }

        mShapeSwitchEntryArray[mShapeSwitchIdx]->mSwitch->setSingleChildOn(0);
        mShapeSwitchEntryArray[mShapeSwitchIdx]->mFlipUpFwdAnim->reset();

        mDOGeometryCreator->setWireframeActiveID(-1);
        mDOGeometryCreator->setResize(0.0f);
    }
    else
    {
        mSnapLevelController->switchToLowerLevel();
        mDOGeometryCreator->setScalePerUnit(  mSnapLevelController->getSnappingLength(), 
                            mSnapLevelController->getSnappingLengthInfo());
    }
}


/***************************************************************
* Function: inputDevMoveEvent()
***************************************************************/
void DSGeometryCreator::inputDevMoveEvent(const osg::Vec3 &pointerOrg, const osg::Vec3 &pointerPos)
{
    if (mDevPressedFlag)
    {
        if (mDrawingState == START_DRAWING)
        {
            bool snap = false;
            osg::Vec3 center = osg::Vec3();

            if (mDOIntersector->test(pointerOrg, pointerPos))
            {
                osg::Vec3 hit = mDOIntersector->getWorldHitPosition();

                CAVEGeodeShape *hitCAVEGeode = dynamic_cast <CAVEGeodeShape*>(mDOIntersector->getHitNode());
                if (hitCAVEGeode)
                {
                    if (hitCAVEGeode->snapToVertex(hit, &center))
                    {
                        snap = true;
                        //mDOGeometryCreator->setSnapPos(center, false);
                    }
                    /*else
                    {
                        mDOGeometryCreator->setSnapPos(pointerPos);
                    }
                }
                else
                {
                    mDOGeometryCreator->setSnapPos(pointerPos);
                }*/
                
                }
            }

            osg::Matrixf w2o = cvr::PluginHelper::getWorldToObjectTransform();
            osg::Matrixd scaleMat;
            scaleMat.makeScale(osg::Vec3(1.5, 1.5, 1.5));
            //scaleMat.invert(scaleMat);
            osg::Matrixd o2cad = scaleMat;

            if (snap)
            {
                center = center * o2cad;

                mDOGeometryCreator->setSnapPos(center, false);
                std::cout << "snap " << center[0] << " " << center[1] << " " << center[2] << std::endl;
            }
            else
            {
                osg::Vec3 pos = pointerPos;
                pos = pos * o2cad;

                mDOGeometryCreator->setSnapPos(pos);
                std::cout << "no snap " << pos[0] << " " << pos[1] << " " << pos[2] << std::endl;
            }
            mDOGeometryCreator->updateReferenceAxis();
        }
    }
    if (!mDevPressedFlag)
    {
        if (prevGeode)
        {
            prevGeode->hideSnapBounds();
        }

        if (mDrawingState == READY_TO_DRAW)
        {
            if (mDOIntersector->test(pointerOrg, pointerPos))
            {
                osg::Vec3 hit = mDOIntersector->getWorldHitPosition();

                CAVEGeodeShape *hitCAVEGeode = dynamic_cast <CAVEGeodeShape*>(mDOIntersector->getHitNode());
                if (hitCAVEGeode)
                { 
                    osg::Vec3 center = osg::Vec3();
                    if (hitCAVEGeode->snapToVertex(hit, &center))
                    {
                        mDOGeometryCreator->updateReferencePlane(center, true);
                        mDOGeometryCreator->setReferencePlaneMasking(false, false, false);
                    }
                    else
                    {
                        mDOGeometryCreator->updateReferencePlane(hit);
                        mDOGeometryCreator->setReferencePlaneMasking(true, true, true);
                    }
                    prevGeode = hitCAVEGeode;
                }
                else
                {
                    mDOGeometryCreator->setReferencePlaneMasking(true, true, true);
                    mDOGeometryCreator->updateReferencePlane(hit);
                }
            }
            else 
            {
                mDOGeometryCreator->setReferencePlaneMasking(false, false, false);
            }
        }
    }
}


/***************************************************************
* Function: inputDevPressEvent()
***************************************************************/
bool DSGeometryCreator::inputDevPressEvent(const osg::Vec3 &pointerOrg, const osg::Vec3 &pointerPos)
{
    mDevPressedFlag = true;

    if (mDrawingState == IDLE)
    {
        // test for sub shape intersection
        bool hit = false;
        for (int i = 0; i < mNumShapeSwitches; i++)
        {
            mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, 
                ((osg::PositionAttitudeTransform*)(mShapeSwitchEntryArray[i]->mSwitch->getChild(0)))->getChild(0));

            if (mDSIntersector->test(pointerOrg, pointerPos))
            {
                mShapeSwitchIdx = i;
                hit = true;
                break;
            }

            if (((osg::PositionAttitudeTransform*)(mShapeSwitchEntryArray[i]->mSwitch->getChild(0)))->getNumChildren()
                 < 2)
                continue;

            mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, 
                ((osg::PositionAttitudeTransform*)(mShapeSwitchEntryArray[i]->mSwitch->getChild(0)))->getChild(1));

            if (mDSIntersector->test(pointerOrg, pointerPos))
            {
                mShapeSwitchIdx = i;
                hit = true;
                break;
            }
        }

        mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, mSphereExteriorGeode);
        if (hit)
        {
            mDrawingState = READY_TO_DRAW;
            DrawingStateTransitionHandle(IDLE, READY_TO_DRAW);

            // initialize wireframe geode attached to 'DesignObjectHandler' root
            mDOGeometryCreator->setReferenceAxisMasking(false);
            mDOGeometryCreator->setWireframeActiveID(mShapeSwitchIdx);
            mDOGeometryCreator->resetWireframeGeodes(gDesignStateCenterPos);
        }

        // switching to lower state 'DSGeometryEditor' only happens in IDLE state, to be specific,
        // the state changes only if a CAVEGeodeShape object is intersected
        else if (mDOIntersector->test(pointerOrg, pointerPos))
        {
            CAVEGeodeShape *hitCAVEGeode = dynamic_cast <CAVEGeodeShape*>(mDOIntersector->getHitNode());
            if (hitCAVEGeode)
            {
                mDOGeometryCollector->setSurfaceCollectionHints(hitCAVEGeode, gDesignStateCenterPos);
                mDOGeometryCollector->toggleCAVEGeodeShape(hitCAVEGeode);

                switchToLowerDesignState(0);
                return false;
            }
        }
    }

    else if (mDrawingState == READY_TO_DRAW)
    {
        if (mDOIntersector->test(pointerOrg, pointerPos))
        {
            mDrawingState = START_DRAWING;
            DrawingStateTransitionHandle(READY_TO_DRAW, START_DRAWING);
     
            mDOGeometryCreator->setWireframeInitPos(pointerPos);
            mDOGeometryCreator->setSolidshapeActiveID(mShapeSwitchIdx);
            mDOGeometryCreator->setPointerDir(pointerPos - pointerOrg);
            mDOGeometryCreator->setScalePerUnit(mSnapLevelController->getSnappingLength(),
                              mSnapLevelController->getSnappingLengthInfo());

            mDOGeometryCreator->setResize(0.0f);
            mDOGeometryCreator->setReferenceAxisMasking(true);
            
            // Vertex/edge snapping 
            CAVEGeodeShape *hitCAVEGeode = dynamic_cast <CAVEGeodeShape*>(mDOIntersector->getHitNode());
            if (hitCAVEGeode)
            {
                osg::Vec3 center = osg::Vec3();
                if (hitCAVEGeode->snapToVertex(mDOIntersector->getWorldHitPosition(), &center))
                {
                    mDOGeometryCreator->setSolidshapeInitPos(center, false);
                } 
                else
                {
                    mDOGeometryCreator->setSolidshapeInitPos(mDOIntersector->getWorldHitPosition());
                }
            }
            else
            {
                mDOGeometryCreator->setSolidshapeInitPos(mDOIntersector->getWorldHitPosition());
            }
            mDOGeometryCreator->setSnapPos(pointerPos);
            mDOGeometryCreator->setReferencePlaneMasking(true, true, true);
        } 
        else 
        {
            mDrawingState = IDLE;
            DrawingStateTransitionHandle(READY_TO_DRAW, IDLE);

            mDOGeometryCreator->setReferencePlaneMasking(false, false, false);
            mDOGeometryCreator->setReferenceAxisMasking(false);
            mDOGeometryCreator->setWireframeActiveID(-1);
        }
    }

    if (mDrawingState == START_DRAWING) 
        return true;
    else 
        return false;
}


/***************************************************************
* Function: inputDevReleaseEvent()
***************************************************************/
bool DSGeometryCreator::inputDevReleaseEvent()
{
    mDevPressedFlag = false;

    if (mDrawingState == START_DRAWING)
    {
        mDrawingState = IDLE;
        DrawingStateTransitionHandle(START_DRAWING, IDLE);

        // finish with Design Object handlers
        mDOGeometryCreator->setReferencePlaneMasking(false, false, false);
        mDOGeometryCreator->setReferenceAxisMasking(false);
        mDOGeometryCreator->registerSolidShape();
        mDOGeometryCreator->setSolidshapeActiveID(-1);
        mDOGeometryCreator->setWireframeActiveID(-1);

        // update audio parameters
        mAudioConfigHandler->updateShapes();

        return true;
    }
    return false;
}


/***************************************************************
* Function: update()
***************************************************************/
void DSGeometryCreator::update()
{

}


/***************************************************************
* Function: setDesignObjectHandlerPtr()
***************************************************************/
void DSGeometryCreator::setDesignObjectHandlerPtr(DesignObjectHandler *designObjectHandler)
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
void DSGeometryCreator::DrawingStateTransitionHandle(const DrawingState& prevState, const DrawingState& nextState)
{
    if (prevState == IDLE && nextState == READY_TO_DRAW)
    {
        setLocked(true);
    }

    else if (prevState == READY_TO_DRAW && nextState == START_DRAWING)
    {

    }

    else if ((prevState == READY_TO_DRAW && nextState == IDLE) ||
	         (prevState == START_DRAWING && nextState == IDLE) ||
	         (prevState == IDLE          && nextState == IDLE))
    {
        mSphereExteriorSwitch->setAllChildrenOn();
        mShapeSwitchEntryArray[mShapeSwitchIdx]->mSwitch->setSingleChildOn(0);
        mShapeSwitchEntryArray[mShapeSwitchIdx]->mFlipUpFwdAnim->reset();

        setLocked(false);
    }
}


void DSGeometryCreator::setHighlight(bool isHighlighted, const osg::Vec3 &pointerOrg, const osg::Vec3 &pointerPos) 
{
    int idx = -1;
    mIsHighlighted = isHighlighted;

    for (int i = 0; i < mNumShapeSwitches; i++)
    {
        ((osg::PositionAttitudeTransform*)(mShapeSwitchEntryArray[i]->mSwitch->getChild(0)))->removeChild(mHighlightGeode);
        mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, 
            ((osg::PositionAttitudeTransform*)(mShapeSwitchEntryArray[i]->mSwitch->getChild(0)))->getChild(0));

        if (mDSIntersector->test(pointerOrg, pointerPos))
        {
            idx = i;
        }
    }

    if (idx > -1)
    {
        osg::Sphere *sphere = new osg::Sphere();
        mSD = new osg::ShapeDrawable(sphere);
        mHighlightGeode = new osg::Geode();
        mHighlightGeode->addDrawable(mSD);
        sphere->setRadius(0.25);
        mSD->setColor(osg::Vec4(1, 1, 1, 0.5));

        StateSet *stateset = mSD->getOrCreateStateSet();
        stateset->setMode(GL_BLEND, StateAttribute::OVERRIDE | StateAttribute::ON);
        stateset->setMode(GL_CULL_FACE, StateAttribute::OVERRIDE | StateAttribute::ON);
        stateset->setRenderingHint(StateSet::TRANSPARENT_BIN);

        ((osg::PositionAttitudeTransform*)(mShapeSwitchEntryArray[idx]->mSwitch->getChild(0)))->addChild(mHighlightGeode);
    }
    mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, mSphereExteriorGeode);
}
 
