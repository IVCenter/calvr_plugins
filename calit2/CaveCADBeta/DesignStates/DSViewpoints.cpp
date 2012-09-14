/***************************************************************
* File Name: DSVirtualSphere.cpp
*
* Description: 
*
* Written by ZHANG Lelin on Oct 5, 2010
*
***************************************************************/
#include "DSViewpoints.h"

using namespace std;
using namespace osg;

//Constructor
DSViewpoints::DSViewpoints()
{
    
    CAVEAnimationModeler::ANIMCreateViewpoints(&fwdVec, &bwdVec);
    for (int i = 0; i < fwdVec.size(); ++i)
    {
        this->addChild(fwdVec[i]);
        this->addChild(bwdVec[i]);
    }

    setAllChildrenOff();

    mDSIntersector = new DSIntersector();
    mDOIntersector = new DOIntersector();
    mDSIntersector->loadRootTargetNode(NULL, NULL);
    mDOIntersector->loadRootTargetNode(NULL, NULL);

    mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, fwdVec[0]->getChild(0));

    mIsOpen = false;
}


/***************************************************************
* Function: setObjectEnabled()
*
* Description:
*
***************************************************************/
void DSViewpoints::setObjectEnabled(bool flag)
{
    AnimationPathCallback* animCallback = NULL;
    setAllChildrenOff();
    if (flag) 
    {
        if (!mIsOpen) // open menu
        {
            setSingleChildOn(0);
            for (int i = 0; i < fwdVec.size(); ++i)
            {
                setChildValue(fwdVec[i], true);
                animCallback = dynamic_cast <AnimationPathCallback*> (fwdVec[i]->getUpdateCallback());
                mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, fwdVec[i]->getChild(0));

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
            for (int i = 1; i < bwdVec.size(); ++i)
            {
                setChildValue(bwdVec[i], true);
                animCallback = dynamic_cast <AnimationPathCallback*> (bwdVec[i]->getUpdateCallback());
                mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, bwdVec[i]);

                if (animCallback)
                {
                    animCallback->reset();
                }
            }
            mIsOpen = false;
        }
    }
    else // close menu
    {
        setSingleChildOn(0);
        for (int i = 1; i < bwdVec.size(); ++i)
        {
            setChildValue(bwdVec[i], true);
            animCallback = dynamic_cast <AnimationPathCallback*> (bwdVec[i]->getUpdateCallback());
            mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, bwdVec[i]);

            if (animCallback)
            {
                animCallback->reset();
            }
        }
        mIsOpen = false;
    }
    mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, fwdVec[0]->getChild(0));
}


/***************************************************************
* Function: switchToPrevState()
***************************************************************/
void DSViewpoints::switchToPrevSubState()
{
   /* AnimationPathCallback* animCallback = NULL;
    setAllChildrenOff();
    if (mIsOpen)
    {
        setSingleChildOn(0);
        for (int i = 1; i < bwdVec.size(); ++i)
        {
            setChildValue(bwdVec[i], true);
            animCallback = dynamic_cast <AnimationPathCallback*> (bwdVec[i]->getUpdateCallback());
            mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, bwdVec[i]);

            if (animCallback)
                animCallback->reset();
        }
        mIsOpen = false;
    }
    else
    {
        setSingleChildOn(0);
        for (int i = 1; i < fwdVec.size(); ++i)
        {
            setChildValue(fwdVec[i], true);
            animCallback = dynamic_cast <AnimationPathCallback*> (fwdVec[i]->getUpdateCallback());
            mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, fwdVec[i]);

            if (animCallback)
                animCallback->reset();
        }
        mIsOpen = true;
    }
    */
}


/***************************************************************
* Function: switchToNextState()
***************************************************************/
void DSViewpoints::switchToNextSubState()
{
    switchToPrevSubState();
/*    AnimationPathCallback* animCallback = NULL;

    if (mIsOpen)
    {
        animCallback = dynamic_cast <AnimationPathCallback*> (mPATransBwd->getUpdateCallback());
        mIsOpen = false;
    }
    else
    {
        animCallback = dynamic_cast <AnimationPathCallback*> (mPATransFwd->getUpdateCallback());
        mIsOpen = true;
    }

    if (animCallback) 
        animCallback->reset();
        */
}


/***************************************************************
* Function: inputDevMoveEvent()
***************************************************************/
void DSViewpoints::inputDevMoveEvent(const osg::Vec3 &pointerOrg, const osg::Vec3 &pointerPos)
{	
    if (mDevPressedFlag)
    {
    }
    if (!mDevPressedFlag)
    {
    }
}


/***************************************************************
* Function: inputDevPressEvent()
***************************************************************/
bool DSViewpoints::inputDevPressEvent(const osg::Vec3 &pointerOrg, const osg::Vec3 &pointerPos)
{
    mDevPressedFlag = true;

    for (int i = 0; i < fwdVec.size(); ++i)
    {
        mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, fwdVec[i]->getChild(0));
        //setChildValue(fwdVec[i], true);

        if (mDSIntersector->test(pointerOrg, pointerPos))
        {
            // open/close menu
            if (i == 0)
            {
                switchToPrevSubState();
            }
            // add a new viewpoint
            else if (i == (fwdVec.size() - 1))
            {
                osg::Matrix mat = cvr::PluginHelper::getObjectMatrix();
                osg::Vec3 eye, center, up;
                mat.getLookAt(eye, center, up);

                Location loc;
                loc.eye = eye;
                loc.center = center;
                loc.up = up;
                locations.push_back(loc);

                CAVEAnimationModeler::ANIMAddViewpoint(&fwdVec, &bwdVec);

                this->removeChildren(0, fwdVec.size() * 2);
                for (int i = 0; i < fwdVec.size(); ++i)
                {
                    this->addChild(fwdVec[i]);
                    this->addChild(bwdVec[i]);
                }
                setAllChildrenOn();
            }
            // load a viewpoint
            else
            {
                osg::Matrix mat;
                Location loc = locations[i - 1];
                mat.makeLookAt(loc.eye, loc.center, loc.up);
                cvr::PluginHelper::setObjectMatrix(mat);
            }
        }
    }
    mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, fwdVec[0]->getChild(0));
    return false;
}


/***************************************************************
* Function: inputDevReleaseEvent()
***************************************************************/
bool DSViewpoints::inputDevReleaseEvent()
{
    mDevPressedFlag = false;

    return false;
}

