/***************************************************************
* File Name: DSTexturePallette.cpp
*
* Description: 
*
* Written by ZHANG Lelin on Jan 12, 2011
*
***************************************************************/
#include "DSTexturePallette.h"

using namespace std;
using namespace osg;


// Constructor
DSTexturePallette::DSTexturePallette(): mTexIndex(0), mTexturingState(IDLE), mAudioConfigHandler(NULL)
{
    // load objects for design state root switch
    CAVEAnimationModeler::ANIMLoadTexturePalletteRoot(&mPATransFwd, &mPATransBwd, &mButtonFwd, &mButtonBwd);

    // load objects for 'IDLE' state
    CAVEAnimationModeler::ANIMLoadTexturePalletteIdle(&mIdleStateSwitch, &mTextureStatesIdleEntry);

    // load objects for 'SELECT_TEXTURE' and 'APPLY_TEXTURE' state
//    CAVEAnimationModeler::ANIMLoadTexturePalletteSelect(&mSelectStateSwitch, &mAlphaTurnerSwitch, 
//							mNumTexs, &mTextureStatesSelectEntryArray);

    mSelectStateSwitch = new osg::Switch();
    mAlphaTurnerSwitch = new osg::Switch();

    // read config for texture entries
    bool isFile = true;
    int j = 0, numTex;
    std::string filename = "", dir, path;
    std::vector<std::string> filenames;

    dir = cvr::ConfigManager::getEntry("dir", "Plugin.CaveCADBeta.Textures", "/home/cehughes");
    dir = dir + "/";
    path = "Plugin.CaveCADBeta.Textures.0";
    filename = cvr::ConfigManager::getEntry(path, "", &isFile);

    while (isFile)
    {
        filenames.push_back(dir + filename);

        j++;
        char buf[50];
        sprintf(buf, "Plugin.CaveCADBeta.Textures.%d", j);
        std::string path = std::string(buf);
        filename = cvr::ConfigManager::getEntry(path, "", &isFile);
    }

    // add texture entries
    CAVEAnimationModeler::ANIMLoadTextureEntries(&mSelectStateSwitch, &mAlphaTurnerSwitch,
        &filenames, &mTextureEntries);


    // read config for color entries
    j = 0; 
    isFile = true;
    int numCol, r, g, b, a;
    osg::Vec4 color;
    std::vector<osg::Vec3> colors; 
    path = "Plugin.CaveCADBeta.Colors.0";

    r = cvr::ConfigManager::getInt("r", path, 1);
    g = cvr::ConfigManager::getInt("g", path, 1);
    b = cvr::ConfigManager::getInt("b", path, 1);
    a = cvr::ConfigManager::getInt("a", path, 1);

    while (isFile)
    {
        colors.push_back(osg::Vec3(r, g, b));

        j++;
        char buf[50];
        sprintf(buf, "Plugin.CaveCADBeta.Colors.%d", j);
        std::string path = std::string(buf);

        r = cvr::ConfigManager::getInt("r", path, 1, &isFile);
        g = cvr::ConfigManager::getInt("g", path, 1, &isFile);
        b = cvr::ConfigManager::getInt("b", path, 1, &isFile);
        a = cvr::ConfigManager::getInt("a", path, 1, &isFile);
    }

    // add color entries
    CAVEAnimationModeler::ANIMLoadColorEntries(&mSelectStateSwitch, &mAlphaTurnerSwitch,
        &colors, &mColorEntries);


    this->addChild(mPATransFwd);
    this->addChild(mPATransBwd);
    this->addChild(mButtonFwd);
    this->addChild(mButtonBwd);

    mPATransFwd->addChild(mIdleStateSwitch);		mPATransBwd->addChild(mIdleStateSwitch);
    mPATransFwd->addChild(mSelectStateSwitch);		mPATransBwd->addChild(mSelectStateSwitch);
    mPATransFwd->addChild(mAlphaTurnerSwitch);		mPATransBwd->addChild(mAlphaTurnerSwitch);

    // use both instances of intersector
    mDSIntersector = new DSIntersector();
    mDOIntersector = new DOIntersector();
    mDSIntersector->loadRootTargetNode(NULL, NULL);
    mDOIntersector->loadRootTargetNode(NULL, NULL);

    setAllChildrenOff();
    mIdleStateSwitch->setAllChildrenOn(); // default state = IDLE
    mSelectStateSwitch->setAllChildrenOff();
    mAlphaTurnerSwitch->setAllChildrenOff();

    mDevPressedFlag = false;
    mIsOpen = false;

    mColorSelector = new ColorSelector(osg::Vec4(0.5,0.5,0.5,1));
    mColorSelector->setVisible(false);
    int x, y, z;
    path = "Plugin.CaveCADBeta.ColorSelectorPosition";
    x = cvr::ConfigManager::getInt("x", path, -300);
    y = cvr::ConfigManager::getInt("y", path, 0);
    z = cvr::ConfigManager::getInt("z", path, 0);

    mColorSelector->setPosition(osg::Vec3(x,y,z));
}


// Destructor
DSTexturePallette::~DSTexturePallette()
{

}


/***************************************************************
* Function: setObjectEnabled()
*
* Description:
*
***************************************************************/
void DSTexturePallette::setObjectEnabled(bool flag)
{
    mObjEnabledFlag = flag;

    if (flag) setAllChildrenOn();
    if (!mPATransFwd || !mPATransBwd) return;
    if (!mButtonFwd || !mButtonBwd) return;

    AnimationPathCallback* animCallback = NULL;

    if (flag && !mIsOpen) // open the menu
    {
        mTexturingState = SELECT_TEXTURE;
        mIsOpen = true;
    
        // open main menu icon
        this->setSingleChildOn(0);
        animCallback = dynamic_cast <AnimationPathCallback*> (mPATransFwd->getUpdateCallback());
        if (animCallback)
            animCallback->reset();
        
        // open all texture buttons
        mSelectStateSwitch->setAllChildrenOn();
        /*
        for (int i = 0; i < mNumTexs; i++)
        {
            mTextureStatesSelectEntryArray[i]->mStateAnimationArray[0]->reset();
            mTextureStatesSelectEntryArray[i]->mEntrySwitch->setSingleChildOn(0);
        }
        */
        for (int i = 0; i < mColorEntries.size(); ++i)
        {
            mColorEntries[i]->mStateAnimationArray[0]->reset();
            mColorEntries[i]->mEntrySwitch->setSingleChildOn(0);
        }

        for (int i = 0; i < mTextureEntries.size(); ++i)
        {
            mTextureEntries[i]->mStateAnimationArray[0]->reset();
            mTextureEntries[i]->mEntrySwitch->setSingleChildOn(0);
        }
        
        // open save button
        animCallback = dynamic_cast <AnimationPathCallback*> (mButtonFwd->getUpdateCallback());
        this->setChildValue(mButtonFwd, true);
        if (animCallback)
            animCallback->reset();

        resetIntersectionRootTarget();
        //mColorSelector->setVisible(true);
    }
    else // close menu
    {
        mTexturingState = IDLE;
        mIsOpen = false;

        // reset all texture buttons
        /*for (int i = 0; i < mNumTexs; i++)
        {
            mTextureStatesSelectEntryArray[i]->mStateAnimationArray[1]->reset();
            mTextureStatesSelectEntryArray[i]->mEntrySwitch->setSingleChildOn(1);
        }*/

        for (int i = 0; i < mColorEntries.size(); ++i)
        {
            mColorEntries[i]->mStateAnimationArray[1]->reset();
            mColorEntries[i]->mEntrySwitch->setSingleChildOn(1);
        }

        for (int i = 0; i < mTextureEntries.size(); ++i)
        {
            mTextureEntries[i]->mStateAnimationArray[1]->reset();
            mTextureEntries[i]->mEntrySwitch->setSingleChildOn(1);
        }

        // reset save button
        animCallback = dynamic_cast <AnimationPathCallback*> (mButtonBwd->getUpdateCallback());
        this->setChildValue(mButtonFwd, false);
        this->setChildValue(mButtonBwd, true);
        if (animCallback)
            animCallback->reset();

        resetIntersectionRootTarget();
    }
}


/***************************************************************
* Function: switchToPrevSubState()
***************************************************************/
void DSTexturePallette::switchToPrevSubState()
{
    // prev state look up 
    switch (mTexturingState)
    {
        case IDLE:
        {
            mTexturingState = APPLY_TEXTURE;
            texturingStateTransitionHandle(IDLE, APPLY_TEXTURE);
            break;
        }
        case SELECT_TEXTURE:
        {
            mTexturingState = IDLE; 
            texturingStateTransitionHandle(SELECT_TEXTURE, IDLE);
            break;
        }
        case APPLY_TEXTURE:
        {
            mTexturingState = SELECT_TEXTURE; 
            texturingStateTransitionHandle(APPLY_TEXTURE, SELECT_TEXTURE);
            break;
        }
        default: break;
    }
    
}


/***************************************************************
* Function: switchToNextSubState()
***************************************************************/
void DSTexturePallette::switchToNextSubState()
{
    // next state look up 
    switch (mTexturingState)
    {
        case IDLE:
        {
            mTexturingState = SELECT_TEXTURE;
            texturingStateTransitionHandle(IDLE, SELECT_TEXTURE);
            break;
        }
        case SELECT_TEXTURE:
        {
            mTexturingState = APPLY_TEXTURE;
            texturingStateTransitionHandle(SELECT_TEXTURE, APPLY_TEXTURE);
            break;
        }
        case APPLY_TEXTURE:
        {
            mTexturingState = IDLE;
            texturingStateTransitionHandle(APPLY_TEXTURE, IDLE);
            break;
        }
        default: break;
    }
}


/***************************************************************
* Function: inputDevMoveEvent()
***************************************************************/
void DSTexturePallette::inputDevMoveEvent(const osg::Vec3 &pointerOrg, const osg::Vec3 &pointerPos)
{
    if (mDevPressedFlag)
    {
        if (mColorSelector->isVisible())
        {
            mColorSelector->buttonEvent(cvr::BUTTON_DRAG, cvr::TrackingManager::instance()->getHandMat(0));
        }
    }
    if (!mDevPressedFlag)
    {

    }
}


/***************************************************************
* Function: inputDevPressEvent()
*
* Proceed to next substate when current state is either 'IDLE'
* or 'SELECT_TEXTURE'.
*
***************************************************************/
bool DSTexturePallette::inputDevPressEvent(const osg::Vec3 &pointerOrg, const osg::Vec3 &pointerPos)
{
    mDevPressedFlag = true;
    if (mIsOpen)
    {
        if (mColorSelector->buttonEvent(cvr::BUTTON_DOWN, cvr::TrackingManager::instance()->getHandMat(0)))
            return true;
        
        // Save Color button
        mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, mButtonFwd->getChild(0));
        if (mDSIntersector->test(pointerOrg, pointerPos))
        {
            osg::Vec4 color4 = mColorSelector->getColor();
            osg::Vec3 color = osg::Vec3();
            color[0] = color4[0];
            color[1] = color4[1];
            color[2] = color4[2];

            CAVEAnimationModeler::ANIMAddColorEntry(&mSelectStateSwitch, &mAlphaTurnerSwitch,
                color, &mColorEntries);

            resetIntersectionRootTarget();
            return true;
        }

        // Color selector open/close button
        mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, mButtonFwd->getChild(1));
        if (mDSIntersector->test(pointerOrg, pointerPos))
        {
            mColorSelector->setVisible(!mColorSelector->isVisible());
            return true;
        }

        // test all submenus for intersection
        /*for (int i = 0; i < mNumTexs; ++i)
        {
            mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, mTextureStatesSelectEntryArray[i]->mEntryGeode);
            if (mDSIntersector->test(pointerOrg, pointerPos))
            {
                mTexIndex = i;
                mTexturingState = APPLY_TEXTURE;
                return true;
            }
        }*/
        
        //mTexIndex = -1;
        //mColorIndex = -1;
        for (int i = 0; i < mColorEntries.size(); ++i)
        {
            mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, mColorEntries[i]->mEntryGeode);
            if (mDSIntersector->test(pointerOrg, pointerPos))
            {
                mTexIndex = -1;
                mColorIndex = i;
                mTexturingState = APPLY_TEXTURE;
                return true;
            }
        }
        for (int i = 0; i < mTextureEntries.size(); ++i)
        {
            mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, mTextureEntries[i]->mEntryGeode);
            if (mDSIntersector->test(pointerOrg, pointerPos))
            {
                mTexIndex = i;
                mColorIndex = -1;
                mTexturingState = APPLY_TEXTURE;
                return true;
            }
        }
        resetIntersectionRootTarget();

        // test world geometry for intersection
        if (mTexturingState == APPLY_TEXTURE)
        {
            mDOIntersector->loadRootTargetNode(gDesignObjectRootGroup, NULL);
            if (mDOIntersector->test(pointerOrg, pointerPos))
            {
                // adjust texture transparency or apply texture to geode
                Node *hitNode = mDOIntersector->getHitNode();
                CAVEGeode *geode = dynamic_cast <CAVEGeode*> (hitNode);
                if (geode)
                {
                    if (mColorIndex > -1)
                    {
                        Vec3 diffuse = mColorEntries[mColorIndex]->getDiffuse();
                        Vec3 specular = mColorEntries[mColorIndex]->getSpecular();
                        geode->applyColor(diffuse, specular, 1.0f);
                    } 
                    else if (mTexIndex > -1)
                    {
                        string filename = mTextureEntries[mTexIndex]->getTexFilename();
                        string audioinfo = mTextureEntries[mTexIndex]->getAudioInfo();
                        geode->applyTexture(filename);
                        geode->applyAudioInfo(audioinfo);
                    }

                    // update audio parameters 
                    mAudioConfigHandler->updateShapes();
                    mTexturingState = IDLE;
                    return true;
                }
            }
        }
        return false;
    }
    return false;
}


/***************************************************************
* Function: inputDevReleaseEvent()
***************************************************************/
bool DSTexturePallette::inputDevReleaseEvent()
{
    mDevPressedFlag = false;
    if (mColorSelector->isVisible())
        mColorSelector->buttonEvent(cvr::BUTTON_UP, cvr::TrackingManager::instance()->getHandMat(0));

    return false;
}


/***************************************************************
* Function: update()
***************************************************************/
void DSTexturePallette::update()
{
}


/***************************************************************
* Function: texturingStateTransitionHandle()
***************************************************************/
void DSTexturePallette::texturingStateTransitionHandle(const TexturingState& prevState, const TexturingState& nextState)
{
    // mIdleStateSwitch: always on except transition between 'SELECT_TEXTURE' and 'APPLY_TEXTURE' 
    mIdleStateSwitch->setAllChildrenOn();	
    mSelectStateSwitch->setAllChildrenOn();
    mAlphaTurnerSwitch->setAllChildrenOff();

    int idxSelected = -1;	// index of animation that to be reset for selected texture entry
    int idxUnselected = -1;	// index of animation that to be reset for all un-selected texture entry

    // transitions between 'IDLE' and 'SELECT_TEXTURE' 
    if (prevState == IDLE && nextState == SELECT_TEXTURE)
    {
        idxSelected = 0;	idxUnselected = 0;
        mTextureStatesIdleEntry->mEntrySwitch->setSingleChildOn(1);
        mTextureStatesIdleEntry->mBwdAnim->reset();
    }
    else if (prevState == SELECT_TEXTURE && nextState == IDLE)
    {
        idxSelected = 1;	idxUnselected = 1;
        mTextureStatesIdleEntry->mEntrySwitch->setSingleChildOn(0);
        mTextureStatesIdleEntry->mFwdAnim->reset();
    }

    // transitions between 'SELECT_TEXTURE' and 'APPLY_TEXTURE' 
    else if (prevState == SELECT_TEXTURE && nextState == APPLY_TEXTURE)
    {
        idxSelected = 4;	idxUnselected = 2;
        mIdleStateSwitch->setAllChildrenOff();
    }
    else if (prevState == APPLY_TEXTURE && nextState == SELECT_TEXTURE)
    {
        idxSelected = 5;	idxUnselected = 3;
        mIdleStateSwitch->setAllChildrenOff();

    }

    // transitions between 'IDLE' and 'APPLY_TEXTURE' 
    else if (prevState == IDLE && nextState == APPLY_TEXTURE)
    {
        idxSelected = 7;	idxUnselected = -1;
        mTextureStatesIdleEntry->mEntrySwitch->setSingleChildOn(1);
        mTextureStatesIdleEntry->mBwdAnim->reset();
    }
    else if (prevState == APPLY_TEXTURE && nextState == IDLE)
    {
        idxSelected = 6;	idxUnselected = -1;
        mTextureStatesIdleEntry->mEntrySwitch->setSingleChildOn(0);
        mTextureStatesIdleEntry->mFwdAnim->reset();
    }

    // reset animation associated with 'mTextureStatesSelectEntryArray' 
    for (int i = 0; i < mNumTexs; i++)
    {
        if (i == mTexIndex && idxSelected >= 0)
        {
            mTextureStatesSelectEntryArray[mTexIndex]->mEntrySwitch->setSingleChildOn(idxSelected);
            mTextureStatesSelectEntryArray[mTexIndex]->mStateAnimationArray[idxSelected]->reset();
        }
        else if (idxUnselected >= 0)
        {
            mTextureStatesSelectEntryArray[i]->mEntrySwitch->setSingleChildOn(idxUnselected);
            mTextureStatesSelectEntryArray[i]->mStateAnimationArray[idxUnselected]->reset();
        }
    }
    resetIntersectionRootTarget();
}


/***************************************************************
* Function: resetIntersectionRootTarget()
***************************************************************/
void DSTexturePallette::resetIntersectionRootTarget()
{
    if (mTexturingState == IDLE)
    {
        Node *targetNode = mTextureStatesIdleEntry->mEntryGeode;
        mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, targetNode);
    }
    else if (mTexturingState == SELECT_TEXTURE)
    {
        Node *targetNode = mTextureStatesIdleEntry->mEntryGeode;
        mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, targetNode);
    }
    else if (mTexturingState == APPLY_TEXTURE)
    {
        Node *targetNode = mTextureStatesIdleEntry->mEntryGeode;
        mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, targetNode);
    }
}


void DSTexturePallette::setHighlight(bool isHighlighted, const osg::Vec3 &pointerOrg, const osg::Vec3 &pointerPos) 
{
    if (isHighlighted)
    {
        // test for intersection
        int idx = -1;
/*        for (int i = 0; i < mNumTexs; ++i)
        {
            mTextureStatesSelectEntryArray[i]->mEntryGeode->removeDrawable(mSD);
            mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, mTextureStatesSelectEntryArray[i]->mEntryGeode);
            if (mDSIntersector->test(pointerOrg, pointerPos))
            {
                idx = i;
            }
        }*/

        for (int i = 0; i < mColorEntries.size(); ++i)
        {
            mColorEntries[i]->mEntryGeode->removeDrawable(mSD);
            mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, mColorEntries[i]->mEntryGeode);
            if (mDSIntersector->test(pointerOrg, pointerPos))
            {
                idx = i;
            }
        }
        
        if (idx > 0)
        {
            osg::Sphere *sphere = new osg::Sphere();
            mSD = new osg::ShapeDrawable(sphere);
            sphere->setRadius(0.15);
            mSD->setColor(osg::Vec4(1, 1, 1, 0.5));
            
            // add highlight geode
            mColorEntries[idx]->mEntryGeode->addDrawable(mSD);

            resetIntersectionRootTarget(); 
            return;
        }

        for (int i = 0; i < mTextureEntries.size(); ++i)
        {
            mTextureEntries[i]->mEntryGeode->removeDrawable(mSD);
            mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, mTextureEntries[i]->mEntryGeode);
            if (mDSIntersector->test(pointerOrg, pointerPos))
            {
                idx = i;
            }
        }
        
        if (idx > 0)
        {
            osg::Sphere *sphere = new osg::Sphere();
            mSD = new osg::ShapeDrawable(sphere);
            sphere->setRadius(0.15);
            mSD->setColor(osg::Vec4(1, 1, 1, 0.5));
            
            // add highlight geode
            mTextureEntries[idx]->mEntryGeode->addDrawable(mSD);

            resetIntersectionRootTarget(); 
            return;
        }

        /*
        resetIntersectionRootTarget();
        
        // no intersection
        if (idx == -1)
            return;

        osg::Sphere *sphere = new osg::Sphere();
        mSD = new osg::ShapeDrawable(sphere);
        sphere->setRadius(0.15);
        mSD->setColor(osg::Vec4(1, 1, 1, 0.5));
        
        // add highlight geode
        mTextureStatesSelectEntryArray[idx]->mEntryGeode->addDrawable(mSD);
        */
    }
    else
    {
/*        for (int i = 0; i < mNumTexs; ++i)
        {
            mTextureStatesSelectEntryArray[i]->mEntryGeode->removeDrawable(mSD);
        }*/

        for (int i = 0; i < mColorEntries.size(); ++i)
        {
            mColorEntries[i]->mEntryGeode->removeDrawable(mSD);
        }

        for (int i = 0; i < mTextureEntries.size(); ++i)
        {
            mTextureEntries[i]->mEntryGeode->removeDrawable(mSD);
        }
    }
    TexturingState oldState = mTexturingState;
    mTexturingState = IDLE;
    resetIntersectionRootTarget();
    mTexturingState = oldState;
}

