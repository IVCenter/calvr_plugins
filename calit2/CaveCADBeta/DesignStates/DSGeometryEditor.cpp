/***************************************************************
* File Name: DSGeometryEditor.cpp
*
* Description: 
*
* Written by ZHANG Lelin on Jan 18, 2010
*
***************************************************************/
#include "DSGeometryEditor.h"

using namespace std;
using namespace osg;


// Constructor
DSGeometryEditor::DSGeometryEditor(): mEdittingState(IDLE), mAudioConfigHandler(NULL)
{
    /* create instance of intersector */
    mDSIntersector = new DSIntersector();
    mDOIntersector = new DOIntersector();
    mDSIntersector->loadRootTargetNode(NULL, NULL);
    mDOIntersector->loadRootTargetNode(NULL, NULL);

    mSnapLevelController = new SnapLevelController();

    setAllChildrenOff();
    mDevPressedFlag = false;
}


// Destructor
DSGeometryEditor::~DSGeometryEditor()
{
}


/***************************************************************
* Function: setObjectEnabled()
*
* Description:
*
***************************************************************/
void DSGeometryEditor::setObjectEnabled(bool flag)
{
    mObjEnabledFlag = flag;
    mDSParticleSystemPtr->setEmitterEnabled(false);

    if (flag) 
        setAllChildrenOn();

    if (flag)
    {
        /* switched from upper state 'DSGeometryCreator' and perform pointer test right after being active */
        mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, NULL);
        mDOIntersector->loadRootTargetNode(gDesignObjectRootGroup, NULL);

        mEdittingState = READY_TO_EDIT;
        mDOGeometryEditor->setToolkitEnabled(true);
        mDOGeometryEditor->setActiveIconToolkit(NULL);

        /* using the locked flag to guarantee that no state switch is allowed when geometries are selected,
           the flag is released when the CAVEGeodeShape vector of 'DOGeometryCollector' is empty.  */
        setLocked(true);
    } 
    else 
    {
        /* release all intersector targets */
        mDSIntersector->loadRootTargetNode(NULL, NULL);
        mDOIntersector->loadRootTargetNode(NULL, NULL);

        mEdittingState = IDLE;
        mDOGeometryEditor->setToolkitEnabled(false);
        mDOGeometryEditor->setActiveIconToolkit(NULL);
    }
}


/***************************************************************
* Function: switchToPrevSubState()
***************************************************************/
void DSGeometryEditor::switchToPrevSubState()
{
    if (mEdittingState == IDLE || mEdittingState == READY_TO_EDIT)
    {
        mDOGeometryEditor->setPrevToolkit();
    }
    else if (mEdittingState == START_EDITTING)
    {
        mSnapLevelController->switchToUpperLevel();
        mDOGeometryEditor->setScalePerUnit(&mSnapLevelController);
    }
}


/***************************************************************
* Function: switchToNextSubState()
***************************************************************/
void DSGeometryEditor::switchToNextSubState()
{
    if (mEdittingState == IDLE || mEdittingState == READY_TO_EDIT)
    {
        mDOGeometryEditor->setNextToolkit();
    }
    else if (mEdittingState == START_EDITTING)
    {
        mSnapLevelController->switchToLowerLevel();
        mDOGeometryEditor->setScalePerUnit(&mSnapLevelController);
    }
}


/***************************************************************
* Function: inputDevMoveEvent()
***************************************************************/
void DSGeometryEditor::inputDevMoveEvent(const osg::Vec3 &pointerOrg, const osg::Vec3 &pointerPos)
{	
    if (mDevPressedFlag && mEdittingState == START_EDITTING)
    {
        mDOGeometryEditor->setSnapUpdate(pointerPos);
    }
}


/***************************************************************
* Function: inputDevPressEvent()
***************************************************************/
bool DSGeometryEditor::inputDevPressEvent(const osg::Vec3 &pointerOrg, const osg::Vec3 &pointerPos)
{
    mDevPressedFlag = true;

    /* 'mEdittingState' only switches between 'READY_TO_EDIT' and 'START_EDITTING', 'IDLE' state
        is a transient state when 'DSGeometryEditor' is not activated.
    */
    if (mEdittingState == READY_TO_EDIT)
    {
        if (mDOIntersector->test(pointerOrg, pointerPos))
        {
            /* 1) If hit node is 'CAVEGeodeIconToolkit' change state to 'START_EDITTING' */
            CAVEGeodeIconToolkit *hitIconToolkit = dynamic_cast <CAVEGeodeIconToolkit*>(mDOIntersector->getHitNode());
            if (hitIconToolkit)
            {
                mEdittingState = START_EDITTING;
                mDOGeometryEditor->setActiveIconToolkit(hitIconToolkit);
                mDOGeometryEditor->setSnapStarted(pointerPos);
                mDOGeometryEditor->setPointerDir(pointerPos - pointerOrg);

                return true;
            }

            /* 2) If hit node is 'CAVEGeodeIconSurface', modify geometry selections. Before toggle function call,
               if geometry list is empty, set all surface highlights as 'unselected'; after toggle function call,
               if geometry geometry list is empty, reset all surface hightlights back to 'normal'. */

            CAVEGeodeIconSurface *hitIconSurface = dynamic_cast <CAVEGeodeIconSurface*>(mDOIntersector->getHitNode());
            if (hitIconSurface)
            {
                if (mDOGeometryCollector->isGeometryVectorEmpty())
                    mDOGeometryCollector->setAllIconSurfaceHighlightsUnselected();

                mDOGeometryCollector->toggleCAVEGeodeIconSurface(hitIconSurface);

                if (mDOGeometryCollector->isGeometryVectorEmpty())
                    mDOGeometryCollector->setAllIconSurfaceHighlightsNormal();

                return false;
            }

            /* if hit node is 'CAVEGeodeShape', modify geometry selections, there is a chance that all
               CAVEGeodeShape are cleared from collector, if so, switch back to 'DSGeometryCreator' */
            CAVEGeodeShape *hitShape = dynamic_cast <CAVEGeodeShape*>(mDOIntersector->getHitNode());
            if (hitShape)
            {
                mDOGeometryCollector->toggleCAVEGeodeShape(hitShape);
                if (mDOGeometryCollector->isGeodeShapeVectorEmpty())
                {
                    /* need to release the locked flag before 'switchToUpperDesignState()', 
                       otherwise the function 'setObjectEnabled' will not be called */
                    setLocked(false);
                    switchToUpperDesignState(0);
                }
                return false;
            }
        }
        else
        {
            /* 3) If hit nothing: clear 'mGeometryVector' if it is not clear, or else, clear  'mGeodeShapeVector'
               and return to upper state */
            if (!mDOGeometryCollector->isGeometryVectorEmpty())
            {
                mDOGeometryCollector->clearGeometryVector();
            }
            else
            {
                mDOGeometryCollector->clearGeodeShapeVector();

                /* need to release the locked flag before 'switchToUpperDesignState()', 
                   otherwise the function 'setObjectEnabled' will not be called */
                setLocked(false);
                switchToUpperDesignState(0);
                return false;
            }
        }
    }

    return false;
}


/***************************************************************
* Function: inputDevReleaseEvent()
***************************************************************/
bool DSGeometryEditor::inputDevReleaseEvent()
{
    mDevPressedFlag = false;
    if (mEdittingState == START_EDITTING)
    {
        mDOGeometryEditor->setActiveIconToolkit(NULL);
        mDOGeometryEditor->setSnapFinished();
        mEdittingState = READY_TO_EDIT;

        /* update audio parameters */
        mAudioConfigHandler->updateShapes();

        return true;
    }

    return false;
}


/***************************************************************
* Function: update()
***************************************************************/
void DSGeometryEditor::update()
{
}


/***************************************************************
* Function: setDesignObjectHandlerPtr()
***************************************************************/
void DSGeometryEditor::setDesignObjectHandlerPtr(DesignObjectHandler *designObjectHandler)
{
    mDOGeometryCollector = designObjectHandler->getDOGeometryCollectorPtr();
    mDOGeometryEditor = designObjectHandler->getDOGeometryEditorPtr();
}


void DSGeometryEditor::setHighlight(bool isHighlighted, const osg::Vec3 &pointerOrg, const osg::Vec3 &pointerPos) 
{

}

