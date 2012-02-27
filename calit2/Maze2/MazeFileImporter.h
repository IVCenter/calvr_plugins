/***************************************************************
* File Name: MazeFileImporter.h
*
***************************************************************/
#ifndef _MAZE_FILE_IMPORTER_H_
#define _MAZE_FILE_IMPORTER_H_

// C++
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string.h>
#include <list>
#include <math.h>

// Open scene graph
#include <osg/Matrixd>
#include <osg/MatrixTransform>
#include <osg/Switch>
#include <osg/Vec3>
#include <osgDB/ReadFile>


typedef std::list<osg::Node*> EntryNodeList;


/***************************************************************
* Class Name: MazeFileImporter
***************************************************************/
class MazeFileImporter
{
  public:
    MazeFileImporter(osg::Switch *globalHintSwitch, osg::Switch *localHintSwitch, osg::Switch *texturedWallSwitch,
		osg::Switch *nonTexturedWallSwitch,  osg::Switch *stillModelSwitch, const std::string &datadir);

    /* enum for starting positions */
    enum StartPosition
    {
	EAST_NORTH,
	EAST_SOUTH,
	WEST_NORTH,
	WEST_SOUTH,
	NORTH_EAST,
	NORTH_WEST,
	SOUTH_EAST,
	SOUTH_WEST,
	RANDOM
    };

    void loadVRMLComponents();
    void loadModel(const std::string &filename);
    void removeModel();
    osg::Matrixd getStartPosMatrix(const StartPosition &pos, const float &scale);

  protected:

    /* enum for general directions */
    enum EntryOrientation
    {
	EAST,
	NORTH,
	WEST,
	SOUTH
    };

    /* component position info: cell index and orientation */
    class EntryPosInfo
    {
      public:
	EntryPosInfo(): x(0), y(0), eo(EAST) {}
	void setOrientation(const std::string &str)
	{
	    if (!strcmp(str.c_str(), "E")) eo = EAST;
	    else if (!strcmp(str.c_str(), "N")) eo = NORTH;
	    else if (!strcmp(str.c_str(), "W")) eo = WEST;
	    else if (!strcmp(str.c_str(), "S")) eo = SOUTH;
	}

	float x, y;
	EntryOrientation eo;
    };

    /* starting position information is stored in 'EntryPosInfoVector' */
    typedef std::vector<EntryPosInfo*> EntryPosInfoVector;
    EntryPosInfoVector mStartPosVector;

    std::string mDataDir;

    /* switch objects passed from 'MazeModelHanlder' */
    osg::Switch *mGlobalHintSwitch, *mLocalHintSwitch, *mTextureWallSwitch, *mNonTextureWallSwitch, *mStillModelSwitch;

    /* preloaded WRL components */
    osg::Node   *mCeilingNode, *mFloorNode, *mCloseWallNode, *mOpenWallNode, *mCloseWallWhiteNode, *mOpenWallWhiteNode,
		*mWallPaintFrameNode, *mLeftRedDoorNode, *mLeftGreenDoorNode, *mLeftYellowDoorNode, *mLeftBlueDoorNode,
		*mRightRedDoorNode, *mRightGreenDoorNode, *mRightYellowDoorNode, *mRightBlueDoorNode,
		*mWallPaint01Node, *mWallPaint02Node, *mWallPaint03Node, *mWallPaint04Node,
		*mColumnNode, *mColumnWhiteNode, *mIndentDoorFrameNode, *mIndentDoorFrameWhiteNode;

    bool lookupEntryNode(const std::string &entryname, EntryPosInfo *infoPtr, EntryNodeList &nodelist, EntryNodeList &hintlist,
			 EntryNodeList &texturedNodeList, EntryNodeList &nonTexturedNodeList, std::ifstream *inFilePtr);

    /* global settings parameters */
    static const float gGridSize;
};


#endif



