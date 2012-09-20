/***************************************************************
* File Name: DSGeometryEditor.h
*
* Description: Derived class from DesignStateBase
*
***************************************************************/

#ifndef _DS_GEOMETRY_EDITOR_H_
#define _DS_GEOMETRY_EDITOR_H_


// Local include
#include "DesignStateBase.h"
#include "../AudioConfigHandler.h"
#include "../SnapLevelController.h"
#include "../AnimationModeler/ANIMGeometryEditor.h"


/***************************************************************
* Class: DSGeometryEditor
***************************************************************/
class DSGeometryEditor: public DesignStateBase
{
  public:
    DSGeometryEditor();
    ~DSGeometryEditor();

    /* virtual functions inherited from base class */
    virtual void setObjectEnabled(bool flag);
    virtual void switchToPrevSubState();
    virtual void switchToNextSubState();

    void inputDevMoveEvent(const osg::Vec3 &pointerOrg, const osg::Vec3 &pointerPos);
    bool inputDevPressEvent(const osg::Vec3 &pointerOrg, const osg::Vec3 &pointerPos);
    bool inputDevReleaseEvent();
    void update();
    void setHighlight(bool isHighlighted, const osg::Vec3 &pointerOrg, const osg::Vec3 &pointerPos);

    /* definition of editting states */
    enum EdittingState
    {
	IDLE,		// 'IDLE' state is always transient when 'DSGeometryEditor' is not activated
	READY_TO_EDIT,
	START_EDITTING
    };

    /* link 'DOGeometryCollector' and 'DOGeometryEditor' to 'DSGeometryEditor',
       class 'DSGeometryEditor' listens to input devices and translate inputs
       to geometry editting actions.
    */
    void setDesignObjectHandlerPtr(DesignObjectHandler *designObjectHandler);

    void setAudioConfigHandlerPtr(AudioConfigHandler *audioConfigHandler) { mAudioConfigHandler = audioConfigHandler; }

  protected:

    EdittingState mEdittingState;
    SnapLevelController *mSnapLevelController;
    AudioConfigHandler *mAudioConfigHandler;

    /* interfaces to design object (DO) controllers */
    DOGeometryCollector *mDOGeometryCollector;
    DOGeometryEditor *mDOGeometryEditor;
};


#endif

