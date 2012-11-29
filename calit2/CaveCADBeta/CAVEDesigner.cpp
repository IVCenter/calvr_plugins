/***************************************************************
* File Name: CAVEDesigner.cpp
*
* Description:  
*
* Written by ZHANG Lelin on Sep 10, 2010
*
***************************************************************/
#include "CAVEDesigner.h"
#include <cvrKernel/PluginHelper.h>

using namespace std;
using namespace osg;

//Constructor
CAVEDesigner::CAVEDesigner(Group* rootGroup): mActiveFlag(false), mKeypressFlag(false)
{
    mDesignStateHandler = new DesignStateHandler(rootGroup);
    mDesignObjectHandler = new DesignObjectHandler(rootGroup);
    mAudioConfigHandler = new AudioConfigHandler(mDesignObjectHandler->getCAVEShapeSwitchPtr());

    // exchange shared handle pointers
    mDesignStateHandler->setDesignObjectHandlerPtr(mDesignObjectHandler);
    mDesignStateHandler->setScenicHandlerPtr(mDesignObjectHandler->getScenicHandlerPtr());
    mDesignStateHandler->setAudioConfigHandlerPtr(mAudioConfigHandler);
}

//Destructor
CAVEDesigner::~CAVEDesigner()
{

}


/***************************************************************
* Function: setActive()
*
* Description: Enable/Disable CAVEDesigner toolkit
*
***************************************************************/
void CAVEDesigner::setActive(bool flag)
{
    mActiveFlag = flag;
    mDesignStateHandler->setActive(flag);
    mDesignObjectHandler->setActive(flag);
}


/***************************************************************
* Function: inputDevMoveEvent()
*
* Description: Handle input device move event
*
***************************************************************/
void CAVEDesigner::inputDevMoveEvent(const Vec3 pointerOrg, const Vec3 pointerPos)
{
    if (!mActiveFlag)   
        return;

    mDesignStateHandler->inputDevMoveEvent(pointerOrg, pointerPos);
    mDesignObjectHandler->inputDevMoveEvent();
}


/***************************************************************
* Function: update()
***************************************************************/
void CAVEDesigner::update(const osg::Vec3 &viewDir, const osg::Vec3 &viewPos)
{
    mDesignStateHandler->update(viewDir, viewPos);
    mAudioConfigHandler->updatePoses(viewDir, viewPos);
}


/***************************************************************
* Function: inputDevPressEvent()
*
* Description: Handle input device press event
*
***************************************************************/
bool CAVEDesigner::inputDevPressEvent(const Vec3 &pointerOrg, const Vec3 &pointerPos, int button)
{
    if (!mActiveFlag) 
        return false;

    bool flagDS = mDesignStateHandler->inputDevPressEvent(pointerOrg, pointerPos);
    bool flagDO = mDesignObjectHandler->inputDevPressEvent(pointerOrg, pointerPos);
    return (flagDS | flagDO);
}


/***************************************************************
* Function: inputDevReleaseEvent()
*
* Description: Handle input device release event
*
***************************************************************/
bool CAVEDesigner::inputDevReleaseEvent()
{
    if (!mActiveFlag) return false;

    bool flagDS = mDesignStateHandler->inputDevReleaseEvent();
    bool flagDO = mDesignObjectHandler->inputDevReleaseEvent();
    return (flagDS | flagDO);
}


/***************************************************************
* Function: inputDevButtonEvent()
*
* Description: Handle extra button events from keyboard
*
***************************************************************/
void CAVEDesigner::inputDevButtonEvent(const int keySym)
{
    if (!mActiveFlag) return;

    DesignStateHandler::InputDevButtonType btnTyp;

    // key index look-up
    switch (keySym)
    {
        case 32: 	btnTyp = DesignStateHandler::TOGGLE;break;
        case 65361:	btnTyp = DesignStateHandler::RIGHT;	break;
        case 65362:	btnTyp = DesignStateHandler::UP;	break;
        case 65363: btnTyp = DesignStateHandler::LEFT;	break;
        case 65364: btnTyp = DesignStateHandler::DOWN;	break;
        default: return;
    }
    mDesignStateHandler->inputDevButtonEvent(btnTyp);
}


/***************************************************************
* Function: inputDevButtonEvent()
*
* Description: Handle extra button events from spin wheel
*
***************************************************************/
void CAVEDesigner::inputDevButtonEvent(const float spinX, const float spinY, const int btnStat)
{
    if (!mActiveFlag) return;

    DesignStateHandler::InputDevButtonType btnTyp = DesignStateHandler::TOGGLE;
}

void CAVEDesigner::setSkydomeVisible(bool flag)
{
    mDesignObjectHandler->setActive(flag);
}

