/***************************************************************
* File Name: Maze2.h
*
* Class Name: Maze2
* Functions: init(), preFrame(), menuEvent()
*
***************************************************************/
#ifndef _MAZE2_H_
#define _MAZE2_H_

// C++
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Open scene graph
#include <osg/Matrixd>
#include <osg/Matrix>
#include <osg/Geode>
#include <osg/ShapeDrawable>
#include <osgDB/ReadFile>

// CalVR menu system
#include <cvrMenu/SubMenu.h>
#include <cvrMenu/MenuButton.h>
#include <cvrMenu/MenuRangeValue.h>
#include <cvrMenu/MenuCheckbox.h>

// CalVR plugin support
#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/ComController.h>
#include <cvrKernel/CVRPlugin.h>
#include <cvrKernel/InteractionEvent.h>
#include <cvrKernel/SceneManager.h>
#include <cvrKernel/PluginHelper.h>

// local
#include "MazeModelHandler.h"
#include "AudioConfigHandler.h"
#include "NavigationHandler.h"

// EOGClient plug-in
#include "EOGClient/ECGClient.h"


/***************************************************************
* Class Name: Maze2
*
* Imported covise plugin 'Maze2' to CalVR
*
***************************************************************/
class Maze2: public cvr::MenuCallback, public cvr::CVRPlugin
{
  public:
    Maze2();
    virtual ~Maze2();

    virtual bool init();
	virtual void preFrame();
	virtual void menuCallback(cvr::MenuItem *item);
	virtual bool processEvent(cvr::InteractionEvent *event);

	void message(int type, char * data) {}
	int getPriority() { return 51; }

  protected:

    void initUI();

    /* Main row menu items */
    cvr::SubMenu *mainMenu, *loadMazeFileMenu, *setStartingPositionMenu, *beginTaskMenu, *optionsMenu;

    /* Button menu items */
    cvr::MenuButton *loadSoundButton, *easyModelButton, *mediumModelButton, *hardModelButton, *deleteModelButton;
    cvr::MenuRangeValue *hardModelSlider;
    cvr::MenuButton *startPosENButton, *startPosESButton, *startPosWNButton, *startPosWSButton,
                     *startPosNEButton, *startPosNWButton, *startPosSEButton, *startPosSWButton, *startPosRandomButton;
    cvr::MenuButton *task01Button, *task02Button, *terminateTaskButton;
    cvr::MenuCheckbox *showLocalHintCheckbox, *enableTexturedWallCheckbox, *enableNaviCheckbox;

    /* maze control objects */
    MazeModelHandler *mMazeModelHandler;
    AudioConfigHandler *mAudioConfigHandler;
    NavigationHandler *mNaviHandler;

    /* EOGClient handle that imported from 'CaveCAD' */
    osg::Group *mEOGRootGroup;
    ECGClient *mECGClient;
    double mClkBuf[2];		// buffer that used to exchange time stamps between master and slave nodes
};

#endif





