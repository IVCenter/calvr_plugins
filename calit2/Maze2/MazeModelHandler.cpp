/***************************************************************
* File Name: MazeModelHandler.cpp
*
* Class entry that controls over maze scene graph: load maze
* files, setting initial positionas and display options.
*
* Writen by Lelin Zhang on May 10, 2011.
*
***************************************************************/

#include "MazeModelHandler.h"

using namespace std;
using namespace osg;
using namespace cvr;


/***************************************************************
*  Constructor
***************************************************************/
MazeModelHandler::MazeModelHandler(const std::string &datadir): mHardModelSize(20)
{
    mDataDir = datadir;

    mRootSwitch = new Switch();
    SceneManager::instance()->getObjectsRoot()->addChild(mRootSwitch);

    mGlobalHintSwitch = new Switch();		// currently empty
    mLocalHintSwitch = new Switch();		// switch of all wall paints
    mTextureWallSwitch = new Switch();		// switch that stores textured wall components
    mNonTextureWallSwitch = new Switch();	// switch that stores non-textured wall components
    mStillModelSwitch = new Switch();		// switch of other maze components
    mRootSwitch->addChild(mGlobalHintSwitch);
    mRootSwitch->addChild(mLocalHintSwitch);
    mRootSwitch->addChild(mTextureWallSwitch);
    mRootSwitch->addChild(mNonTextureWallSwitch);
    mRootSwitch->addChild(mStillModelSwitch);

    /* set initial configuration of switches */
    mGlobalHintSwitch->setAllChildrenOff();
    mLocalHintSwitch->setAllChildrenOn();
    mTextureWallSwitch->setAllChildrenOn();
    mNonTextureWallSwitch->setAllChildrenOff();
    mStillModelSwitch->setAllChildrenOn();

    mMazeFileImporter = new MazeFileImporter(mGlobalHintSwitch, mLocalHintSwitch, 
		mTextureWallSwitch, mNonTextureWallSwitch, mStillModelSwitch, mDataDir);
    mMazeGenerator = NULL;
}


/***************************************************************
*  Function: loadModel()
***************************************************************/
void MazeModelHandler::loadModel(const ModelType &typ)
{
    /* remove existing model file to avoid geometry duplications */
    mMazeFileImporter->removeModel();

    string modelFileName;
    if (typ == EASY) modelFileName = mDataDir + "EasyModel.MAZ";
    else if (typ == MEDIUM) modelFileName = mDataDir + "MediumModel.MAZ";
    else if (typ == HARD)
    {
		modelFileName = mDataDir + "HardModel.MAZ";

		string fileHeader = mDataDir;
		fileHeader += string("HardModel");
		mMazeGenerator = new MazeGenerator(fileHeader.c_str());
		mMazeGenerator->resize(mHardModelSize);
		mMazeGenerator->buildStructure();
		mMazeGenerator->exportToDSNFile();
		// mMazeGenerator->exportToMAZFile();
		delete mMazeGenerator;

		// return;
    }
    mMazeFileImporter->loadVRMLComponents();
    mMazeFileImporter->loadModel(modelFileName);

    /* set initial scale and viewport */
	Matrix initXMat(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, -1.25, 1);
	Navigation::instance()->setScale(1.f);
	PluginHelper::setObjectMatrix(initXMat);
	PluginHelper::setObjectScale(1000.0);
}


/***************************************************************
*  Function: removeModel()
***************************************************************/
void MazeModelHandler::removeModel()
{
    mMazeFileImporter->removeModel();
}


/***************************************************************
*  Function: setStartPos()
***************************************************************/
void MazeModelHandler::setStartPos(const MazeFileImporter::StartPosition &pos)
{
	Matrix initXMat = mMazeFileImporter->getStartPosMatrix(pos, 1.f);
    Navigation::instance()->setScale(1.f);
    PluginHelper::setObjectMatrix(initXMat);
}


/***************************************************************
*  Function: beginTask()
***************************************************************/
void MazeModelHandler::beginTask()
{
}


/***************************************************************
*  Function: terminateTask()
***************************************************************/
void MazeModelHandler::terminateTask()
{
}


/***************************************************************
*  Function: setLocalHintEnabled()
***************************************************************/
void MazeModelHandler::setLocalHintEnabled(bool flag)
{
    if (flag) mLocalHintSwitch->setAllChildrenOn();
    else mLocalHintSwitch->setAllChildrenOff();
}


/***************************************************************
*  Function: setTexturedWallEnabled()
***************************************************************/
void MazeModelHandler::setTexturedWallEnabled(bool flag)
{
    if (flag)
    {
		mTextureWallSwitch->setAllChildrenOn();
		mNonTextureWallSwitch->setAllChildrenOff();
    }
    else
    {
		mTextureWallSwitch->setAllChildrenOff();
		mNonTextureWallSwitch->setAllChildrenOn();
    }
}


























