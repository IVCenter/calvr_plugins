/***************************************************************
* File Name: PathRecorder.h
*
* Class Name: PathRecorder
* Major functions: init(), preFrame(), menuEvent()
*
***************************************************************/
#ifndef _PATH_RECORDER_H_
#define _PATH_RECORDER_H_

// C++
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Open scene graph
#include <osg/Matrixd>

// CalVR
#include <cvrKernel/CVRPlugin.h>
#include <cvrMenu/SubMenu.h>
#include <cvrMenu/MenuCheckbox.h>
#include <cvrMenu/MenuButton.h>
#include <cvrMenu/MenuText.h>
#include <cvrMenu/MenuRangeValue.h>

// Local includes
#include "PathRecordManager.h"


using namespace cvr;

/** Class: PathRecorder
*/
class PathRecorder: public CVRPlugin, public MenuCallback
{
  public:
    PathRecorder();
    ~PathRecorder();

    /* OpenCOVER plugin functions */
    bool init();
    void preFrame();
    void menuCallback(MenuItem *);
    void message(int type, char * & data, bool);

    void initUI();

  protected:
    string mDataDir;
    PathRecordManager *mPathRecordManager;

    /* Menu item prototypes */

    SubMenu * mainMenu, *selectPathFilesMenu;
    MenuCheckbox * recordPathCheckbox, * playbackPathCheckbox;
    MenuCheckbox ** recordEntryCheckboxList;
    MenuText * timeInfoLabel;
    MenuRangeValue * playbackSpeedSlider;
    MenuButton *startButton, *pauseButton, *stopButton;

    /*coSubMenuItem *mainMenuItem, *selectPathFilesMenuItem;
    coRowMenu *mainMenu, *selectPathFilesMenu;
    coCheckboxMenuItem *recordPathCheckbox, *playbackPathCheckbox;
    coCheckboxMenuItem **recordEntryCheckboxList;
    coLabelMenuItem *timeInfoLabel;
    coSliderToolboxItem *playbackSpeedSlider;
    coButtonMenuItem *startButton, *pauseButton, *stopButton;*/
};

#endif










