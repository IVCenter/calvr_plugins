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
    mValCutoff = 1.0;
    mValDownTime = PluginHelper::getProgramDuration();
    mValPressed = false;
    pointerPressFlag = false;
    frameCnt = 0;

    // get data directory from config file
    mDataDir = ConfigManager::getEntry("Plugin.CaveCADBeta.DataDir");
    mDataDir = mDataDir + "/"; 

    // init CalVR UI
    mainMenu = new SubMenu("CaveCADBeta", "CaveCADBeta");
	MenuSystem::instance()->addMenuItem(mainMenu);

    // Main row menu items
    enablePluginCheckbox = new MenuCheckbox("Enable CaveCADBeta", false);
    enablePluginCheckbox->setCallback(this);
    mainMenu->addItem(enablePluginCheckbox);

    setToolkitVisibleCheckbox = new MenuCheckbox("Set toolkit visible", false);
    setToolkitVisibleCheckbox->setCallback(this);
    //mainMenu->addItem(setToolkitVisibleCheckbox);

    mSkydomeCheckbox = new MenuCheckbox("Skydome", true);
    mSkydomeCheckbox->setCallback(this);
    mainMenu->addItem(mSkydomeCheckbox);

    mShadowCheckbox = new MenuCheckbox("Shadows", false);
    mShadowCheckbox->setCallback(this);
    mainMenu->addItem(mShadowCheckbox);


    // CaveCADBeta local objects
    osg::Group *root = new osg::Group();

    // set initial scale and viewport
    // Note: Originally this was rescaling the global objects root, so this is a substitute
    // until I can extract and change all the hardcoded sizes and distances
    mScaleMat = new osg::MatrixTransform();
    osg::Matrixd mat;
    mat.makeScale(osg::Vec3(1000, 1000, 1000));
    Matrixd intObeMat = Matrixd(1, 0, 0, 0, 
                                0, 1, 0, 0, 
                                0, 0, 1, 0, 
                                0, 0, 1, 1);//-.500, 1);
//    mat.postMult(intObeMat);
    mat.preMult(intObeMat);

    mScaleMat->setMatrix(mat);
    mScaleMat->addChild(root);

    
//    SceneManager::instance()->getObjectsRoot()->addChild(scaleMat);//root);
    mCAVEDesigner = new CAVEDesigner(root);
    mShadowedScene = NULL; 
/*
    osgShadow::ShadowedScene *shadowedScene = new osgShadow::ShadowedScene();
//    shadowedScene->setReceivesShadowTraversalMask(0x2);
//    shadowedScene->setCastsShadowTraversalMask(0x3);
//    scaleMat->setNodeMask(0xFFFFFF | (0x2 | 0x3) | osg::StateAttribute::OVERRIDE);
    
    osgShadow::ShadowVolume *sv = new osgShadow::ShadowVolume();
    osgShadow::ShadowTexture *st = new osgShadow::ShadowTexture();
    osgShadow::ShadowMap *sm = new osgShadow::ShadowMap();
    shadowedScene->setShadowTechnique(sm);
    sm->setTextureSize(osg::Vec2s(1024,1024));
    
    shadowedScene->addChild(scaleMat);*/
//    SceneManager::instance()->getObjectsRoot()->addChild(mScaleMat);

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
        osg::Matrixf w2o = PluginHelper::getWorldToObjectTransform();
        osg::Matrixd o2cad = mScaleMat->getInverseMatrix();
        Matrixf viewMat;
                
        float x, y, z;
        x = ConfigManager::getFloat("x", "Plugin.CaveCADBeta.MenuPosition", 3000.0);
        y = ConfigManager::getFloat("y", "Plugin.CaveCADBeta.MenuPosition", 8000.0);
        z = ConfigManager::getFloat("z", "Plugin.CaveCADBeta.MenuPosition", 0.0);

        viewMat.makeTranslate(0, 100, 0);

        osg::Vec3 pos(x, y, z);
        Vec3 viewOrg = viewMat.getTrans() * w2o * o2cad;
        Vec3 viewPos = pos * w2o * o2cad;

        Vec3 viewDir = viewPos - viewOrg;
        viewDir.normalize(); 

        osg::Vec3 pointerOrg, pointerPos;

        pointerOrg = osg::Vec3(0, 0, 0) * TrackingManager::instance()->getHandMat(0) * w2o * o2cad;
        pointerPos = osg::Vec3(0, 100, 0) * TrackingManager::instance()->getHandMat(0) * w2o * o2cad;

        mCAVEDesigner->update(viewDir, viewPos);
        mCAVEDesigner->inputDevMoveEvent(pointerOrg, pointerPos);

        // valuator press cutoff
        if (mValPressed && PluginHelper::getProgramDuration() - mValDownTime > mValCutoff)
        {
            mValPressed = false;
        }
    }
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
            SceneManager::instance()->getObjectsRoot()->addChild(mScaleMat);
            if (mCAVEDesigner)
                mCAVEDesigner->getStateHandler()->setVisible(true);

            //mainMenu->setVisible(false);
	    	if (mCAVEDesigner) 
                mCAVEDesigner->setActive(true);

	    	// set initial scale and viewport
			//Matrixd intObeMat = Matrixd(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, -500, 1);
			//PluginHelper::setObjectScale(1000.f);
	    	//PluginHelper::setObjectMatrix(intObeMat);
            mIsEnabled = true;
      	} 
        else 
        {
            SceneManager::instance()->getObjectsRoot()->removeChild(mScaleMat);

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

    if (item == mSkydomeCheckbox)
    {
        mCAVEDesigner->setSkydomeVisible(mSkydomeCheckbox->getValue());
 //       std::cout << "skydome checkbox" << std::endl;
    }
    
    if (item == mShadowCheckbox)
    {
        if (mShadowCheckbox->getValue() && enablePluginCheckbox->getValue())
        {
            //osgShadow::ShadowedScene *shadowedScene = new osgShadow::ShadowedScene();
            if (!mShadowedScene)
            {
                mShadowedScene = new osgShadow::ShadowedScene();
                mShadowedScene->addChild(mScaleMat);
            }
        //    shadowedScene->setReceivesShadowTraversalMask(0x2);
        //    shadowedScene->setCastsShadowTraversalMask(0x3);
        //    scaleMat->setNodeMask(0xFFFFFF | (0x2 | 0x3) | osg::StateAttribute::OVERRIDE);
            
            osgShadow::ShadowMap *sm = new osgShadow::ShadowMap();
            mShadowedScene->setShadowTechnique(sm);
            sm->setTextureSize(osg::Vec2s(1024,1024));

            SceneManager::instance()->getObjectsRoot()->removeChild(mScaleMat);
            SceneManager::instance()->getObjectsRoot()->addChild(mShadowedScene);
        }
        else
        {
            if (!mShadowedScene)
            {
                mShadowedScene = new osgShadow::ShadowedScene();
                mShadowedScene->addChild(mScaleMat);
            }

            SceneManager::instance()->getObjectsRoot()->removeChild(mShadowedScene);
            SceneManager::instance()->getObjectsRoot()->addChild(mScaleMat);
        }
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
        
        osg::Matrixd o2cad = mScaleMat->getInverseMatrix();
        pointerOrg = osg::Vec3(0, 0, 0) * TrackingManager::instance()->getHandMat(0) * w2o * o2cad;
        pointerPos = osg::Vec3(0, 100, 0) * TrackingManager::instance()->getHandMat(0) * w2o * o2cad;

        if (!enablePluginCheckbox->getValue())
            return false;
            
        if (tie->getInteraction() == BUTTON_DOUBLE_CLICK)
        {
            //std::cout << "double click" << std::endl;
        }

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
        int valID = 0;
        int left = 65361, up = 65362, right = 65363, down = 65364;
        //mValPressed = false;

        if (id == valID)
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
        }
        else if (id == 1)
        {
            float val = vie->getValue();
            // RIGHT 
            if (val == 1)
            {
                if (!mValPressed)
                {
                    mValPressed = true;
                    mCAVEDesigner->inputDevButtonEvent(right);
                    mValDownTime = PluginHelper::getProgramDuration();
                }
            }
            // LEFT
            else if(val == -1)
            {
                if (!mValPressed)
                {
                    mValPressed = true;
                    mCAVEDesigner->inputDevButtonEvent(left);
                    mValDownTime = PluginHelper::getProgramDuration();
                }
            }
        }

    }
	return false;
}

