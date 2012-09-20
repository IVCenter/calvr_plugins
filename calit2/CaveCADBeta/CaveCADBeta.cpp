/***************************************************************
* File Name: CaveCADBeta.cpp
*
* Description: Main plugin source code file for intuitive CAD 
* tools in starcave. 
*
* Written by ZHANG Lelin on Sep 8, 2010
*
***************************************************************/
#include "CaveCADBeta.h"

CVRPLUGIN(CaveCADBeta)

using namespace std;
using namespace osg;
using namespace cvr;

//Constructor
CaveCADBeta::CaveCADBeta(): mCAVEDesigner(NULL)
{
}

//Destructor
CaveCADBeta::~CaveCADBeta()
{
}


/***************************************************************
*  Function: init()
***************************************************************/
bool CaveCADBeta::init()
{
    mIsEnabled = false;
    mValCutoff = 0.5;
    mValDownTime = PluginHelper::getProgramDuration();
    mValPressed = false;
    pointerPressFlag = false;
    frameCnt = 0;

    /* get data directory from config file */
    mDataDir = ConfigManager::getEntry("Plugin.CaveCADBeta.DataDir");
    mDataDir = mDataDir + "/"; 

    mMenuDistance = ConfigManager::getFloat("Plugin.CaveCADBeta.MenuDistance", 2000.0);

    /* init CalVR UI */
    mainMenu = new SubMenu("CaveCADBeta", "CaveCADBeta");
	MenuSystem::instance()->addMenuItem(mainMenu);

    /* Main row menu items */
    enablePluginCheckbox = new MenuCheckbox("Enable CaveCADBeta", false);
    enablePluginCheckbox->setCallback(this);
    mainMenu->addItem(enablePluginCheckbox);

    setToolkitVisibleCheckbox = new MenuCheckbox("Set toolkit visible", false);
    setToolkitVisibleCheckbox->setCallback(this);
    mainMenu->addItem(setToolkitVisibleCheckbox);

    /* CaveCADBeta local objects */
    mCAVEDesigner = new CAVEDesigner(SceneManager::instance()->getObjectsRoot());
    if(ComController::instance()->isMaster())
    {
		mCAVEDesigner->getAudioConfigHandler()->setMasterFlag(true);
		mCAVEDesigner->getAudioConfigHandler()->connectServer();
    }

    return true;
}


/***************************************************************
*  Function: preFrame()
***************************************************************/
void CaveCADBeta::preFrame()
{
    if (mIsEnabled)
    {
        Matrixf invBaseMat = PluginHelper::getWorldToObjectTransform();
        Matrixf viewMat;// = PluginHelper::getHeadMat(0);
        viewMat.makeTranslate(0, 200, 0);
        
        float x, y, z;
        x = ConfigManager::getFloat("x", "Plugin.CaveCADBeta.MenuPosition", 3000.0);
        y = ConfigManager::getFloat("y", "Plugin.CaveCADBeta.MenuPosition", 8000.0);
        z = ConfigManager::getFloat("z", "Plugin.CaveCADBeta.MenuPosition", 0.0);

        osg::Vec3 pos(x, y, z);

        Vec3 viewOrg = viewMat.getTrans() * invBaseMat; 
        Vec3 viewPos = pos * viewMat * invBaseMat; 
        Vec3 viewDir = viewPos - viewOrg;
        viewDir.normalize(); 

        osg::Vec3 pointerOrg, pointerPos;
        osg::Matrixd w2o = PluginHelper::getWorldToObjectTransform();

        pointerOrg = osg::Vec3(0, 0, 0) * TrackingManager::instance()->getHandMat(0) * w2o;
        pointerPos = osg::Vec3(0, 1000, 0) * TrackingManager::instance()->getHandMat(0) * w2o;

        mCAVEDesigner->update(viewDir, viewPos);
        mCAVEDesigner->inputDevMoveEvent(pointerOrg, pointerPos);

        // valuator press cutoff
        if (mValPressed && PluginHelper::getProgramDuration() - mValDownTime > mValCutoff)
        {
            mValPressed = false;
        }
    }

    // get pointer position in world space 
    /*Matrixf invBaseMat = PluginHelper::getWorldToObjectTransform();
    Matrixf baseMat = PluginHelper::getObjectToWorldTransform();
    Matrixf viewMat = PluginHelper::getHeadMat(0);
	Matrixf pointerMat = TrackingManager::instance()->getHandMat(0);

    Vec3 pointerPos = Vec3(0.0, 1000.0, 0.0) * pointerMat * invBaseMat;
    Vec3 pointerOrg = Vec3(0.0, 0.0, 0.0) * pointerMat * invBaseMat;

    // get viewer's position in world space 
    Vec3 viewOrg = viewMat.getTrans() * invBaseMat; 
    Vec3 viewPos = Vec3(0.0, 1000.0, 0.0) * viewMat * invBaseMat; 
    Vec3 viewDir = viewPos - viewOrg;
    viewDir.normalize(); 

    //handle pointer/update events 
    // coPointerButton* pointerBtn = cover->getPointerButton();
	//unsigned int btn = TrackingManager::instance()->getRawButtonMask();
    // if (pointerBtn->wasPressed())
	if (btn)
    {
		if (!pointerPressFlag)
		{
	    	pointerPressFlag = true;
	    	pointerPressEvent(pointerOrg, pointerPos);
		}	
    }
	// else if (pointerBtn->wasReleased()) 
    else if (!btn)
    {
		pointerReleaseEvent();
		pointerPressFlag = false;
    }
    else
    {
		pointerMoveEvent(pointerOrg, pointerPos);
    }
    mCAVEDesigner->update(viewDir, viewPos);

    // spin wheel and top pointer buttons 
    float spinX = PluginHelper::getValuator(0, 0);
    float spinY = PluginHelper::getValuator(0, 1);
    int pointerStat = TrackingManager::instance()->getRawButtonMask();

    spinWheelEvent(spinX, spinY, pointerStat);

    //Debugging codes for model calibration
    float scale = PluginHelper::getObjectScale();
    Matrix xMat = PluginHelper::getObjectMatrix();

    cerr << endl << "Scale = " << scale << endl;
    cerr << xMat(0, 0) << " " << xMat(0, 1) << " " << xMat(0, 2) << " " << xMat(0, 3) << endl;
    cerr << xMat(1, 0) << " " << xMat(1, 1) << " " << xMat(1, 2) << " " << xMat(1, 3) << endl;
    cerr << xMat(2, 0) << " " << xMat(2, 1) << " " << xMat(2, 2) << " " << xMat(2, 3) << endl;
    cerr << xMat(3, 0) << " " << xMat(3, 1) << " " << xMat(3, 2) << " " << xMat(3, 3) << endl;

    cerr << " Frame Count = " << frameCnt++ << endl;
    */
}


/***************************************************************
*  Function: menuCallback()
***************************************************************/
void CaveCADBeta::menuCallback(MenuItem *item)
{
    if (item == enablePluginCheckbox)
    {
      	if (enablePluginCheckbox->getValue())
      	{
            //mainMenu->setVisible(false);
	    	if (mCAVEDesigner) 
                mCAVEDesigner->setActive(true);

	    	/* set initial scale and viewport */
			Matrixd intObeMat = Matrixd(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, -500, 1);
			PluginHelper::setObjectScale(1000.f);
	    	PluginHelper::setObjectMatrix(intObeMat);
            mIsEnabled = true;
      	} 
        else 
        {
            //mainMenu->setVisible(true);
            mIsEnabled = false;
	    	if (mCAVEDesigner) 
                mCAVEDesigner->setActive(false);
		}
    }

    if (item == setToolkitVisibleCheckbox)
    {
		bool flag = setToolkitVisibleCheckbox->getValue();
		if (mCAVEDesigner) 
            mCAVEDesigner->getStateHandler()->setVisible(flag);
    }
}


/***************************************************************
*  Function: processEvent()
***************************************************************/
bool CaveCADBeta::processEvent(cvr::InteractionEvent *event)
{
    if (!mIsEnabled)
        return false;

    KeyboardInteractionEvent * kie = event->asKeyboardEvent();
    if (kie)
    {
        if (kie->getInteraction() == KEY_DOWN)
        {
            mCAVEDesigner->inputDevButtonEvent(kie->getKey());
            return true;
        }
    }

    TrackedButtonInteractionEvent * tie = event->asTrackedButtonEvent();
    if (tie)
    {
        osg::Vec3 pointerOrg, pointerPos;
        osg::Matrixd w2o = PluginHelper::getWorldToObjectTransform();

        pointerOrg = osg::Vec3(0, 0, 0) * TrackingManager::instance()->getHandMat(0) * w2o;
        pointerPos = osg::Vec3(0, 1000, 0) * TrackingManager::instance()->getHandMat(0) * w2o;

        if (tie->getHand() == 0 && tie->getButton() == 0)
        {

            if (tie->getInteraction() == BUTTON_DOWN)
            {
                bool res = mCAVEDesigner->inputDevPressEvent(pointerOrg, pointerPos);
                return res;
            }

            else if (tie->getInteraction() == BUTTON_UP)
            {
                bool res = mCAVEDesigner->inputDevReleaseEvent();
                return res;
            }
        }
        if (tie->getHand() == 0 && tie->getButton() == 1)
        {
            if (tie->getInteraction() == BUTTON_DOWN)
            {
                bool res = mCAVEDesigner->inputDevPressEvent(pointerOrg, pointerPos, 1);
                return res;
            }
        }

        return false;
    }
    
    ValuatorInteractionEvent * vie = event->asValuatorEvent();
    if(vie)
    {
        int id = vie->getValuator();
        int mainVal = 1, subVal = 0;
        int left = 65361, up = 65362, right = 65363, down = 65364;

        if (id == mainVal)
        {
            float val = vie->getValue();
            // LEFT
            if (val == 1)
            {
                if (!mValPressed)
                {
                    mCAVEDesigner->inputDevButtonEvent(left);
                    mValPressed = true;
                    mValDownTime = PluginHelper::getProgramDuration();
                }
            }
            // RIGHT       
            else if (val == -1)
            {
                if (!mValPressed)
                {
                    mCAVEDesigner->inputDevButtonEvent(right);
                    mValPressed = true;
                    mValDownTime = PluginHelper::getProgramDuration();
                }
            }
            else
            {
                mValPressed = false;
            }
        }
        if (id == subVal)
        {
            float val = vie->getValue();
            // UP
            if (val == 1)
            {
                if (!mValPressed)
                {
                    mValPressed = true;
                    mCAVEDesigner->inputDevButtonEvent(up);
                    mValDownTime = PluginHelper::getProgramDuration();
                }
            }
            // DOWN
            else if(val == -1)
            {
                if (!mValPressed)
                {
                    mValPressed = true;
                    mCAVEDesigner->inputDevButtonEvent(down);
                    mValDownTime = PluginHelper::getProgramDuration();
                }
            }
            else
            {
                mValPressed = false;
            }
        }
    }
    else
    {
        mValPressed = false;
    }

	return false;
}


/***************************************************************
*  Function: spinWheelEvent()
***************************************************************/
void CaveCADBeta::spinWheelEvent(const float spinX, const float spinY, const int pointerStat)
{
//    mCAVEDesigner->inputDevButtonEvent(spinX, spinY, pointerStat);
}


/***************************************************************
*  Function: pointerMoveEvent()
***************************************************************/
void CaveCADBeta::pointerMoveEvent(const Vec3 pointerOrg, const Vec3 pointerPos)
{
   // mCAVEDesigner->inputDevMoveEvent(pointerOrg, pointerPos);
}


/***************************************************************
*  Function: pointerPressEvent()
***************************************************************/
void CaveCADBeta::pointerPressEvent(const Vec3 pointerOrg, const Vec3 pointerPos)
{
    if (0)//mCAVEDesigner->inputDevPressEvent(pointerOrg, pointerPos))
    {
/*	Disable all other navigations when pointer button is pressed
	cover->disableNavigation("WALK");
	cover->disableNavigation("FLY");
	cover->disableNavigation("DRIVE");
*/
    }
}


/***************************************************************
*  Function: pointerReleaseEvent()
***************************************************************/
void CaveCADBeta::pointerReleaseEvent()
{
    if (0)//mCAVEDesigner->inputDevReleaseEvent())
    {
/*  Enable 'walk' or 'drive' when pointer buttons is released
	cover->enableNavigation("WALK");
*/
    }
}

