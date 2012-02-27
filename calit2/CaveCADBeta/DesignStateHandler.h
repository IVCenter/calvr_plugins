/***************************************************************
* File Name: DesignStateHandler.h
*
* Class Name: DesignStateHandler
*
***************************************************************/

#ifndef _DESIGN_STATE_HANDLER_H_
#define _DESIGN_STATE_HANDLER_H_


// C++
#include <iostream>

// Open scene graph
#include <osg/Group>

// Local includes
#include "DesignStateRenderer.h"
#include "AudioConfigHandler.h"


/***************************************************************
* Class: DesignStateHandler
***************************************************************/
class DesignStateHandler
{
  public:
    DesignStateHandler(osg::Group* rootGroup);
    ~DesignStateHandler();

    /* definition of button types */
    enum InputDevButtonType
    {
	TOGGLE,
	LEFT,
	UP,
	RIGHT,
	DOWN
    };

    void setActive(bool flag);
    void setVisible(bool flag);
    void inputDevButtonEvent(const InputDevButtonType& typ);
    void inputDevMoveEvent(const osg::Vec3 pointerOrg, const osg::Vec3 pointerPos);
    bool inputDevPressEvent(const Vec3 &pointerOrg, const Vec3 &pointerPos);
    bool inputDevReleaseEvent();
    void update(const osg::Vec3 &viewDir, const osg::Vec3 &viewPos);

    /* accept handler pointers passed from 'DesignObjectHandler' */
    void setDesignObjectHandlerPtr(DesignObjectHandler *designObjectHandler);
    void setScenicHandlerPtr(VirtualScenicHandler *virtualScenicHandler);
    void setAudioConfigHandlerPtr(AudioConfigHandler *audioConfigHandler);

  protected:
    bool mActiveFlag;
    osg::Group *mDesignStateRoot;

    DesignStateRenderer *mDesignStateRenderer;

    // Testing Function
    void TestAnimationPaths();
};


#endif
