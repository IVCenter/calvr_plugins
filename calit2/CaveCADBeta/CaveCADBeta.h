/***************************************************************
* File Name: CaveCADBeta.h
*
* Class Name: CaveCADBeta
* Major functions: init(), preFrame(), menuEvent(), buttonEvent()
*
***************************************************************/
#ifndef _CAVECAD_BETA_H_
#define _CAVECAD_BETA_H_

// C++
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Open scene graph
#include <osg/BoundingBox>
#include <osg/Drawable>
#include <osg/Geode>
#include <osg/Group>
#include <osg/ShapeDrawable>
#include <osg/Switch>
#include <osgDB/ReadFile>

// CalVR plugin support
#include <config/ConfigManager.h>
#include <kernel/ComController.h>
#include <kernel/CVRPlugin.h>
#include <kernel/SceneManager.h>
#include <kernel/PluginHelper.h>

// CalVR menu system
#include <menu/SubMenu.h>
#include <menu/MenuButton.h>
#include <menu/MenuRangeValue.h>
#include <menu/MenuCheckbox.h>
#include <menu/MenuText.h>

// Local Classes
#include "CAVEDesigner.h"


/** Class: CaveCADBeta
*/
class CaveCADBeta : public cvr::MenuCallback, public cvr::CVRPlugin
{
  public:
    CaveCADBeta();
    ~CaveCADBeta();

    /* OpenCOVER plugin functions */
	virtual bool init();
	virtual void preFrame();
	virtual void menuCallback(cvr::MenuItem * item);
	virtual bool processEvent(cvr::InteractionEvent *event);
	void message(int type, char * data);
	int getPriority() { return 51; }

    /* pointer functions */
    void spinWheelEvent(const float spinX, const float spinY, const int pointerStat);
    void pointerMoveEvent(const osg::Vec3 pointerOrg, const osg::Vec3 pointerPos);
    void pointerPressEvent(const osg::Vec3 pointerOrg, const osg::Vec3 pointerPos);
    void pointerReleaseEvent();

  private:

    /* use flag to ensure that each click event is only handled once */
    bool pointerPressFlag;
    int frameCnt;
    string mDataDir;
  
  protected:

    /* Main row menu items */
	cvr::SubMenu *mainMenu;
    cvr::MenuCheckbox *enablePluginCheckbox, *setToolkitVisibleCheckbox;

    /* CaveCAD local objects */
    CAVEDesigner *mCAVEDesigner;
};

#endif

