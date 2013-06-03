/***************************************************************
* File Name: DSObjectPlacer.cpp
*
* Description: 
*
* Written by Cathy Hughes 25 Sept 2012
*
***************************************************************/
#include "DSObjectPlacer.h"

using namespace std;
using namespace osg;


// Constructor
DSObjectPlacer::DSObjectPlacer()
{
    CAVEAnimationModeler::ANIMCreateObjectPlacer(&mFwdVec, &mBwdVec);
    for (int i = 0; i < mFwdVec.size(); ++i)
    {
        this->addChild(mFwdVec[i]);
        this->addChild(mBwdVec[i]);
    }

    // create both instances of intersector
    mDSIntersector = new DSIntersector();
    mDOIntersector = new DOIntersector();
    mDSIntersector->loadRootTargetNode(NULL, NULL);
    mDOIntersector->loadRootTargetNode(NULL, NULL);

    setAllChildrenOff();
    mDevPressedFlag = false;
    mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, mFwdVec[0]->getChild(0));

    prevGeode = NULL;
    mDrawingState = IDLE;
}


// Destructor
DSObjectPlacer::~DSObjectPlacer()
{

}


/***************************************************************
* Function: setObjectEnabled()
*
* Description:
*
***************************************************************/
void DSObjectPlacer::setObjectEnabled(bool flag)
{
    mDrawingState = IDLE;

    mObjEnabledFlag = flag;
    AnimationPathCallback* animCallback = NULL;
    setAllChildrenOff();
    if (flag && !mIsOpen) // open menu
    {
        setSingleChildOn(0);
        for (int i = 0; i < mFwdVec.size(); ++i)
        {
            setChildValue(mFwdVec[i], true);
            animCallback = dynamic_cast <AnimationPathCallback*> (mFwdVec[i]->getUpdateCallback());
            mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, mFwdVec[i]->getChild(0));

            if (animCallback)
            {
                animCallback->reset();
            }
        }
        mIsOpen = true;
    }
    else // close menu
    {
        setSingleChildOn(0);
        for (int i = 1; i < mBwdVec.size(); ++i)
        {
            setChildValue(mBwdVec[i], true);
            animCallback = dynamic_cast <AnimationPathCallback*> (mBwdVec[i]->getUpdateCallback());
            mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, mBwdVec[i]);

            if (animCallback)
            {
                animCallback->reset();
            }
        }
        mIsOpen = false;
    }

    mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, mFwdVec[0]->getChild(0));
}


/***************************************************************
* Function: switchToPrevSubState()
***************************************************************/
void DSObjectPlacer::switchToPrevSubState()
{
    if (mDrawingState == IDLE)
    {

    }
    else
    {

    }
}


/***************************************************************
* Function: switchToNextSubState()
***************************************************************/
void DSObjectPlacer::switchToNextSubState()
{
    if (mDrawingState == IDLE)
    {

    }
    else
    {

    }
}


/***************************************************************
* Function: inputDevMoveEvent()
***************************************************************/
void DSObjectPlacer::inputDevMoveEvent(const osg::Vec3 &pointerOrg, const osg::Vec3 &pointerPos)
{	
    if (mDevPressedFlag)
    {
        if (mDrawingState == PLACE_OBJECT && _activeObject)
        {
            osg::Vec3 pos = osg::Vec3(0,5000,0) * cvr::PluginHelper::getHandMat() 
                * cvr::PluginHelper::getWorldToObjectTransform();
            _activeObject->setPosition(pos);//pointerPos); 
        }
    }
    if (!mDevPressedFlag)
    {
        if (mDrawingState == PLACE_OBJECT && _activeObject)
        {
            osg::Vec3 pos = osg::Vec3(0,3000,0) * cvr::PluginHelper::getHandMat() 
                * cvr::PluginHelper::getWorldToObjectTransform();
            _activeObject->setPosition(pos);//pointerPos); 
        }
    }
}


/***************************************************************
* Function: inputDevPressEvent()
***************************************************************/
bool DSObjectPlacer::inputDevPressEvent(const osg::Vec3 &pointerOrg, const osg::Vec3 &pointerPos)
{
//std::cout << "ObjectPlacer press" << std::endl;
    mDevPressedFlag = true;

    for (int i = 0; i < mFwdVec.size(); ++i)
    {
        mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, mFwdVec[i]->getChild(0));
        //setChildValue(fwdVec[i], true);

        if (mDSIntersector->test(pointerOrg, pointerPos))
        {
            if (i > 0) // not the top menu item
            {
                _activeObject = new osg::PositionAttitudeTransform();

                string dir = cvr::ConfigManager::getEntry("dir", "Plugin.CaveCADBeta.Objects", "/home/cehughes");
                dir = dir + "/";

                char buf[50];
                sprintf(buf, "Plugin.CaveCADBeta.Objects.%d", i-1);
                string path = std::string(buf);
                
                bool isFile;
                string filename = cvr::ConfigManager::getEntry(path, "", &isFile);
                filename = dir + filename; 
                float scale = cvr::ConfigManager::getFloat("scale", path, 1, &isFile);
                float zoffset = cvr::ConfigManager::getFloat("zoffset", path, 0, &isFile);

                if (isFile)
                {
                    osgDB::ReaderWriter::Options *options = new osgDB::ReaderWriter::Options();
                    options->setObjectCacheHint(osgDB::ReaderWriter::Options::CACHE_NONE);

                    osg::Node *node = osgDB::readNodeFile(filename, options);
                    //_activeObject->addChild(node);
                    _activeObject->setScale(osg::Vec3(scale, scale, scale));
                    osg::PositionAttitudeTransform *pat = new osg::PositionAttitudeTransform();
                    pat->setPosition(osg::Vec3(0,0,scale*zoffset));
                    pat->addChild(node);
                    _activeObject->addChild(pat);
                    cvr::PluginHelper::getObjectsRoot()->addChild(_activeObject); 
                    mDrawingState = PLACE_OBJECT;
                    return true;
                }
                //std::cout << "intersecting " << i << std::endl;    
            }
        }
    }

    mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, mFwdVec[0]->getChild(0));


    if (mDrawingState == PLACE_OBJECT)
    {
        if (_activeObject)
        {
            _activeObject = NULL;
            mDrawingState = IDLE;
        }
        return true;
    }
    return false;
}


/***************************************************************
* Function: inputDevReleaseEvent()
***************************************************************/
bool DSObjectPlacer::inputDevReleaseEvent()
{
    mDevPressedFlag = false;

    if (mDrawingState == PLACE_OBJECT)
    {
        //mDrawingState = IDLE;
        return true;
    }
    return false;
}


/***************************************************************
* Function: update()
***************************************************************/
void DSObjectPlacer::update()
{
}


/***************************************************************
* Function: setDesignObjectHandlerPtr()
***************************************************************/
void DSObjectPlacer::setDesignObjectHandlerPtr(DesignObjectHandler *designObjectHandler)
{
    mDOGeometryCollector = designObjectHandler->getDOGeometryCollectorPtr();
    mDOGeometryCreator = designObjectHandler->getDOGeometryCreatorPtr();
}


/***************************************************************
* Function: DrawingStateTransitionHandle()
*
* Take actions on transition between drawing states
*
***************************************************************/
void DSObjectPlacer::DrawingStateTransitionHandle(const DrawingState& prevState, const DrawingState& nextState)
{

}


void DSObjectPlacer::setHighlight(bool isHighlighted, const osg::Vec3 &pointerOrg, const osg::Vec3 &pointerPos) 
{
    int idx = -1;
    mIsHighlighted = isHighlighted;

    mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, mSphereExteriorGeode);
    mDSIntersector->loadRootTargetNode(gDesignStateRootGroup, mFwdVec[0]->getChild(0));
}
 
