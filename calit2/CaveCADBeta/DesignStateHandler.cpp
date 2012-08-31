/***************************************************************
* File Name: DesignStateHandler.cpp
*
* Description:  
*
* Written by ZHANG Lelin on Sep 10, 2010
*
***************************************************************/
#include "DesignStateHandler.h"

using namespace std;
using namespace osg;

//Constructor
DesignStateHandler::DesignStateHandler(Group* rootGroup): mActiveFlag(false)
{
    mDesignStateRoot = new Group();
    rootGroup->addChild(mDesignStateRoot);

    mDesignStateRenderer = new DesignStateRenderer(mDesignStateRoot);
}

//Destructor
DesignStateHandler::~DesignStateHandler()
{
}


/***************************************************************
* Function: setActive()
*
* Description: Load activating/deactivating animation paths
*
***************************************************************/
void DesignStateHandler::setActive(bool flag)
{
    mActiveFlag = flag;

    /* initialize menu system */


}


/***************************************************************
* Function: setVisible()
***************************************************************/
void DesignStateHandler::setVisible(bool flag)
{
    mDesignStateRenderer->setVisible(flag);
}


/***************************************************************
* Function: inputDevButtonEvent()
*
* Description: Switch between design states, re-render icons
*
***************************************************************/
void DesignStateHandler::inputDevButtonEvent(const InputDevButtonType& typ)
{
    if (typ == TOGGLE)
    {
        mDesignStateRenderer->toggleDSVisibility();
    }
    else if (typ == LEFT)
    {
        mDesignStateRenderer->switchToPrevState();
    }
    else if (typ == RIGHT)
    {
        mDesignStateRenderer->switchToNextState();
    }
    else if (typ == UP)
    {
        mDesignStateRenderer->switchToPrevSubState();
    }
    else if (typ == DOWN)
    {
        mDesignStateRenderer->switchToNextSubState();
    }
}


/***************************************************************
* Function: inputDevMoveEvent()
*
* Description: Move virtual sphere along pointer's position
*
***************************************************************/
void DesignStateHandler::inputDevMoveEvent(const Vec3 pointerOrg, const Vec3 pointerPos)
{
    mDesignStateRenderer->inputDevMoveEvent(pointerOrg, pointerPos);
}


/***************************************************************
* Function: update()
***************************************************************/
void DesignStateHandler::update(const Vec3 &viewDir, const Vec3 &viewPos)
{
    mDesignStateRenderer->update(viewDir, viewPos);
}


/***************************************************************
* Function: inputDevPressEvent()
***************************************************************/
bool DesignStateHandler::inputDevPressEvent(const Vec3 &pointerOrg, const Vec3 &pointerPos)
{
    return (mDesignStateRenderer->inputDevPressEvent(pointerOrg, pointerPos));
}


/***************************************************************
* Function: inputDevReleaseEvent()
***************************************************************/
bool DesignStateHandler::inputDevReleaseEvent()
{
    return (mDesignStateRenderer->inputDevReleaseEvent());
}


/***************************************************************
* Function: setDesignObjectHandlerPtr()
***************************************************************/
void DesignStateHandler::setDesignObjectHandlerPtr(DesignObjectHandler *designObjectHandler)
{
    mDesignStateRenderer->setDesignObjectHandlerPtr(designObjectHandler);
}


/***************************************************************
* Function: setScenicHandlerPtr()
***************************************************************/
void DesignStateHandler::setScenicHandlerPtr(VirtualScenicHandler *virtualScenicHandler)
{
    mDesignStateRenderer->setScenicHandlerPtr(virtualScenicHandler);
}


/***************************************************************
* Function: setAudioConfigHandlerPtr()
***************************************************************/
void DesignStateHandler::setAudioConfigHandlerPtr(AudioConfigHandler *audioConfigHandler)
{
    mDesignStateRenderer->setAudioConfigHandlerPtr(audioConfigHandler);
}




















