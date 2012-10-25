/***************************************************************
* File Name: Maze2.cpp
*
* Description: Second version of maze plugin for EOG testing
*
* Written by ZHANG Lelin on May 9, 2011
* Converted into CalVR plugin on Nov 8, 2011
*
***************************************************************/
#include "Maze2.h"

using namespace osg;
using namespace std;
using namespace cvr;


CVRPLUGIN(Maze2)


Maze2::Maze2()
{
}


Maze2::~Maze2()
{
}


/***************************************************************
*  Function: init()
***************************************************************/
bool Maze2::init()
{
    /* get data directory */
    string dataDir = ConfigManager::getEntry("Plugin.Maze2.DataDir");
    dataDir = dataDir + "/";

    mMazeModelHandler = new MazeModelHandler(dataDir);
    mAudioConfigHandler = new AudioConfigHandler();
    mNaviHandler = new NavigationHandler();

    /* setup EOG client connections */
    mEOGRootGroup = new Group;
    SceneManager::instance()->getObjectsRoot()->addChild(mEOGRootGroup);
    mECGClient = new ECGClient(mEOGRootGroup);

    /* setup audio connections for head node */
    if(ComController::instance()->isMaster())
    {
		mAudioConfigHandler->setMasterFlag(true);
		mAudioConfigHandler->connectServer();

		mECGClient->setMasterFlag(true);
    }

    initUI();	// 'initUI' must be called after initialization of 'mECGClient'

    return true;
}


/***************************************************************
*  Function: initUI()
***************************************************************/
void Maze2::initUI()
{
    /* init OpenCOVER UI: main menu item */
	mainMenu = new SubMenu("Maze2", "Maze2");
	MenuSystem::instance()->addMenuItem(mainMenu);

    /* init EOG client sub menus */
    mECGClient->initCVRMenu(mainMenu);

    /* load maze files checkbox */
    loadMazeFileMenu = new SubMenu("Load Maze Files", "Load Maze Files");
    mainMenu->addItem(loadMazeFileMenu);

    loadSoundButton = new MenuButton("Load Sound");
    easyModelButton = new MenuButton("Easy");
    mediumModelButton = new MenuButton("Medium");
    hardModelButton = new MenuButton("Hard");
    hardModelSlider = new MenuRangeValue("How hard could a 'hard' maze be?", 20, 60, 1);
    deleteModelButton = new MenuButton("Delete All");

    loadSoundButton->setCallback(this);
    easyModelButton->setCallback(this);
    mediumModelButton->setCallback(this);
    hardModelButton->setCallback(this);
    deleteModelButton->setCallback(this);
    hardModelSlider->setCallback(this);

    hardModelSlider->setValue(20);
    loadMazeFileMenu->addItem(loadSoundButton);
    loadMazeFileMenu->addItem(easyModelButton);
    loadMazeFileMenu->addItem(mediumModelButton);
    loadMazeFileMenu->addItem(hardModelButton);
    loadMazeFileMenu->addItem(hardModelSlider);
    loadMazeFileMenu->addItem(deleteModelButton);

    /* set starting position checkbox */
    setStartingPositionMenu = new SubMenu("Set Starting Positions", "Set Starting Positions");
    mainMenu->addItem(setStartingPositionMenu);

    startPosENButton = new MenuButton("East-North");
    startPosESButton = new MenuButton("East-South");
    startPosWNButton = new MenuButton("West-North");
    startPosWSButton = new MenuButton("West-South");
    startPosNEButton = new MenuButton("North-East");
    startPosNWButton = new MenuButton("North-West");
    startPosSEButton = new MenuButton("South-East");
    startPosSWButton = new MenuButton("South-West");
    startPosRandomButton = new MenuButton("Random");

    startPosENButton->setCallback(this);
    startPosESButton->setCallback(this);
    startPosWNButton->setCallback(this);
    startPosWSButton->setCallback(this);
    startPosNEButton->setCallback(this);
    startPosNWButton->setCallback(this);
    startPosSEButton->setCallback(this);
    startPosSWButton->setCallback(this);
    startPosRandomButton->setCallback(this);

    setStartingPositionMenu->addItem(startPosENButton);
    setStartingPositionMenu->addItem(startPosESButton);
    setStartingPositionMenu->addItem(startPosWNButton);
    setStartingPositionMenu->addItem(startPosWSButton);
    setStartingPositionMenu->addItem(startPosNEButton);
    setStartingPositionMenu->addItem(startPosNWButton);
    setStartingPositionMenu->addItem(startPosSEButton);
    setStartingPositionMenu->addItem(startPosSWButton);
    setStartingPositionMenu->addItem(startPosRandomButton);

    /* begin task checkbox */
    beginTaskMenu = new SubMenu("Begin Tasks", "Begin Tasks"); 
    mainMenu->addItem(beginTaskMenu);

    task01Button = new MenuButton("Task: Find Objective Door");
    task02Button = new MenuButton("Task: Find Way Out");
    terminateTaskButton = new MenuButton("Terminate Tasks");
    task01Button->setCallback(this);
    task02Button->setCallback(this);
    terminateTaskButton->setCallback(this);
    beginTaskMenu->addItem(task01Button);
    beginTaskMenu->addItem(task02Button);
    beginTaskMenu->addItem(terminateTaskButton);

    /* options checkbox */
    optionsMenu = new SubMenu("Options", "Options");
    mainMenu->addItem(optionsMenu);

    showLocalHintCheckbox = new MenuCheckbox("Show Local Hints", true);
    showLocalHintCheckbox->setCallback(this);
    optionsMenu->addItem(showLocalHintCheckbox);

    enableTexturedWallCheckbox = new MenuCheckbox("Show Wall Textures", true);
    enableTexturedWallCheckbox->setCallback(this);
    optionsMenu->addItem(enableTexturedWallCheckbox);

    /* navigations checkbox */
    enableNaviCheckbox = new MenuCheckbox("Enable Navigation Buttons", false);
    enableNaviCheckbox->setCallback(this);
    mainMenu->addItem(enableNaviCheckbox);
}


/***************************************************************
*  Function: preFrame()
***************************************************************/
void Maze2::preFrame()
{
    /* get pointer position in world space */
    Matrixf invBaseMat = PluginHelper::getWorldToObjectTransform();
    Matrixf viewMat = PluginHelper::getHeadMat(0);
    Matrixf xformMat = PluginHelper::getObjectMatrix();
 
    /* get viewer's position in world space */
    Vec3 viewOrg = viewMat.getTrans() * invBaseMat; 
    Vec3 viewPos = Vec3(0.0, 1.0, 0.0) * viewMat * invBaseMat; 
    Vec3 viewDir = viewPos - viewOrg;
    viewDir.normalize();

    mAudioConfigHandler->updatePoses(viewDir, viewPos);
    mNaviHandler->updateNaviStates(Navigation::instance()->getScale(), viewDir, viewPos);
    // mNaviHandler->updateButtonStates();
    mNaviHandler->updateXformMat();

    /* ECGClient: Master-Slave operations: Read current time for update */
    double frameDuration;
    if(ComController::instance()->isMaster())
    {
		frameDuration = PluginHelper::getLastFrameDuration();
		((double*)mClkBuf)[0] = frameDuration;
		ComController::instance()->sendSlaves((char*) &mClkBuf, sizeof(mClkBuf));
    } 
    else 
    {
		ComController::instance()->readMaster((char*) &mClkBuf, sizeof(mClkBuf));
		frameDuration = ((double*)mClkBuf)[0];
    }
    mECGClient->update(viewMat, invBaseMat, xformMat, frameDuration);

/*
cerr << "viewPos = " << viewPos.x() << " " << viewPos.y() << " " << viewPos.z() << endl;
cerr << "scale = " << PluginHelper::getObjectScale() << endl;
cerr << "invBaseMat: "  << endl;
cerr << invBaseMat(0, 0) << " " << invBaseMat(0, 1) << " " << invBaseMat(0, 2) << " " << invBaseMat(0, 3) << " " << endl;
cerr << invBaseMat(1, 0) << " " << invBaseMat(1, 1) << " " << invBaseMat(1, 2) << " " << invBaseMat(1, 3) << " " << endl;
cerr << invBaseMat(2, 0) << " " << invBaseMat(2, 1) << " " << invBaseMat(2, 2) << " " << invBaseMat(2, 3) << " " << endl;
cerr << invBaseMat(3, 0) << " " << invBaseMat(3, 1) << " " << invBaseMat(3, 2) << " " << invBaseMat(3, 3) << " " << endl;
cerr << "xformMat: "  << endl;
cerr << xformMat(0, 0) << " " << xformMat(0, 1) << " " << xformMat(0, 2) << " " << xformMat(0, 3) << " " << endl;
cerr << xformMat(1, 0) << " " << xformMat(1, 1) << " " << xformMat(1, 2) << " " << xformMat(1, 3) << " " << endl;
cerr << xformMat(2, 0) << " " << xformMat(2, 1) << " " << xformMat(2, 2) << " " << xformMat(2, 3) << " " << endl;
cerr << xformMat(3, 0) << " " << xformMat(3, 1) << " " << xformMat(3, 2) << " " << xformMat(3, 3) << " " << endl;
cerr << "viewMat: "  << endl;
cerr << viewMat(0, 0) << " " << viewMat(0, 1) << " " << viewMat(0, 2) << " " << viewMat(0, 3) << " " << endl;
cerr << viewMat(1, 0) << " " << viewMat(1, 1) << " " << viewMat(1, 2) << " " << viewMat(1, 3) << " " << endl;
cerr << viewMat(2, 0) << " " << viewMat(2, 1) << " " << viewMat(2, 2) << " " << viewMat(2, 3) << " " << endl;
cerr << viewMat(3, 0) << " " << viewMat(3, 1) << " " << viewMat(3, 2) << " " << viewMat(3, 3) << " " << endl;
cerr << endl;
*/

}


/***************************************************************
*  Function: menuCallback()
***************************************************************/
void Maze2::menuCallback(MenuItem *item)
{
    /* load/remove maze model files */
    if (item == loadSoundButton)
    {
  		mAudioConfigHandler->loadSoundSource();
    }
    else if (item == easyModelButton)
    {
		mMazeModelHandler->loadModel(MazeModelHandler::EASY);
    }
    else if (item == mediumModelButton)
    {
		mMazeModelHandler->loadModel(MazeModelHandler::MEDIUM);
    }
    else if (item == hardModelButton)
    {
		mMazeModelHandler->loadModel(MazeModelHandler::HARD);
    }
    else if (item == hardModelSlider)
    {
		float val = hardModelSlider->getValue();
		mMazeModelHandler->setHardModelSize((int)val);
    }
    else if (item == deleteModelButton)
    {
		mMazeModelHandler->removeModel();
    }

    /* set viewer's initial positions */
    if (item == startPosENButton)
    {
		mMazeModelHandler->setStartPos(MazeFileImporter::EAST_NORTH);
    }
    else if (item == startPosESButton)
    {
		mMazeModelHandler->setStartPos(MazeFileImporter::EAST_SOUTH);
    }
    else if (item == startPosWNButton)
    {
		mMazeModelHandler->setStartPos(MazeFileImporter::WEST_NORTH);
    }
    else if (item == startPosWSButton)
    {
		mMazeModelHandler->setStartPos(MazeFileImporter::WEST_SOUTH);
    }
    else if (item == startPosNEButton)
    {
		mMazeModelHandler->setStartPos(MazeFileImporter::NORTH_EAST);
    }
    else if (item == startPosNWButton)
    {
		mMazeModelHandler->setStartPos(MazeFileImporter::NORTH_WEST);
    }
    else if (item == startPosSEButton)
    {
		mMazeModelHandler->setStartPos(MazeFileImporter::SOUTH_EAST);
    }
    else if (item == startPosSWButton)
    {
		mMazeModelHandler->setStartPos(MazeFileImporter::SOUTH_WEST);
    }
    else if (item == startPosRandomButton)
    {
		mMazeModelHandler->setStartPos(MazeFileImporter::RANDOM);
    }

    /* set wayfinding tasks */
    if (item == task01Button)
    {
		mMazeModelHandler->beginTask();
    }
    else if (item == task02Button)
    {
		mMazeModelHandler->beginTask();
    }
    else if (item == terminateTaskButton)
    {
		mMazeModelHandler->terminateTask();
    }

    /* set display options */
    if (item == showLocalHintCheckbox)
    {
		mMazeModelHandler->setLocalHintEnabled(showLocalHintCheckbox->getValue());
    }
    else if (item == enableTexturedWallCheckbox)
    {
		mMazeModelHandler->setTexturedWallEnabled(enableTexturedWallCheckbox->getValue());
    }

    /* enable / disable button based navigation */
    if (item == enableNaviCheckbox)
    {
		mNaviHandler->setEnabled(enableNaviCheckbox->getValue());
    }
}


/***************************************************************
*  Function: processEvent()
***************************************************************/
bool Maze2::processEvent(InteractionEvent *event)
{
	if (event->getEventType() == KEYBOARD_INTER_EVENT)
	{
		KeyboardInteractionEvent *keyEvent = event->asKeyboardEvent();
		int key = keyEvent->getKey();
		if (keyEvent->getInteraction() == KEY_DOWN) 
            mNaviHandler->updateKeys(key, true);
		else 
            mNaviHandler->updateKeys(key, false);
	}

	return false;
}

