/***************************************************************
* File Name: MazeModelHandler.h
*
***************************************************************/
#ifndef _MAZE_MODEL_HANDLER_H_
#define _MAZE_MODEL_HANDLER_H_


// C++
#include <iostream>
#include <string.h>

// Open scene graph
#include <osg/Matrixd>
#include <osg/Group>
#include <osg/Switch>

// CalVR plugin support
#include <cvrKernel/Navigation.h>
#include <cvrKernel/SceneManager.h>
#include <cvrKernel/PluginHelper.h>

// local
#include "MazeGenerator.h"
#include "MazeFileImporter.h"


/***************************************************************
* Class Name: MazeModelHandler
***************************************************************/
class MazeModelHandler
{
  public:
    MazeModelHandler(const std::string &datadir);

    /* enum for difficulty levels */
    enum ModelType
    {
		EASY,
		MEDIUM,
		HARD
    };

    /* control functions called from OpenVRUI */
    void loadModel(const ModelType &typ);
    void removeModel();
    void setStartPos(const MazeFileImporter::StartPosition &pos);
    void beginTask();
    void terminateTask();
    void setHardModelSize(const int &size) { mHardModelSize = size; }
    void setLocalHintEnabled(bool flag);
    void setTexturedWallEnabled(bool flag);

  protected:

    std::string mDataDir;
    osg::Switch *mRootSwitch, *mGlobalHintSwitch, *mLocalHintSwitch, 
	*mTextureWallSwitch, *mNonTextureWallSwitch, *mStillModelSwitch;

    int mHardModelSize;
    MazeGenerator *mMazeGenerator;
    MazeFileImporter *mMazeFileImporter;
};

#endif


