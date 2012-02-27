/***************************************************************
* File Name: CAVEDesigner.h
*
* Class Name: CAVEDesigner
*
***************************************************************/

#ifndef _CAVE_DESIGNER_H_
#define _CAVE_DESIGNER_H_

// Open scene graph
#include <osg/Group>

// local includes
#include "DesignStateHandler.h"
#include "DesignObjectHandler.h"
#include "AudioConfigHandler.h"


/***************************************************************
* Class: CAVEDesigner
***************************************************************/
class CAVEDesigner
{
  public:
    CAVEDesigner(osg::Group* rootGroup);
    ~CAVEDesigner();

    void setActive(bool flag);

    void inputDevMoveEvent(const osg::Vec3 pointerOrg, const osg::Vec3 pointerPos);
    bool inputDevPressEvent(const Vec3 &pointerOrg, const Vec3 &pointerPos);
    bool inputDevReleaseEvent();
    void update(const osg::Vec3 &viewDir, const osg::Vec3 &viewPos);
    void inputDevButtonEvent(const int keySym);
    void inputDevButtonEvent(const float spinX, const float spinY, const int btnStat);

    DesignStateHandler *getStateHandler() { return mDesignStateHandler; }
    DesignObjectHandler *getObjectHandler() { return mDesignObjectHandler; }
    AudioConfigHandler *getAudioConfigHandler() { return mAudioConfigHandler; }

  protected:

    bool mActiveFlag, mKeypressFlag;

    DesignStateHandler *mDesignStateHandler;
    DesignObjectHandler *mDesignObjectHandler;
    AudioConfigHandler *mAudioConfigHandler;
};


#endif
