/***************************************************************
* File Name: MazeFileImporter.cpp
*
* Class that defines and construct maze components
*
* Writen by Lelin Zhang on May 10, 2011.
*
***************************************************************/

#include "MazeFileImporter.h"

using namespace std;
using namespace osg;


/* maze grid size is fixed, as it is associated with a set of WRL files */
const float MazeFileImporter::gGridSize(4.8f);


/***************************************************************
*  Constructor
***************************************************************/
MazeFileImporter::MazeFileImporter(Switch *globalHintSwitch, Switch *localHintSwitch, Switch *texturedWallSwitch,
		Switch *nonTexturedWallSwitch, Switch *stillModelSwitch, const string &datadir): mDataDir(datadir)
{
    mGlobalHintSwitch = globalHintSwitch;
    mLocalHintSwitch = localHintSwitch;
    mStillModelSwitch = stillModelSwitch;
    mTextureWallSwitch = texturedWallSwitch;
    mNonTextureWallSwitch = nonTexturedWallSwitch;

    mStartPosVector.clear();
}


/***************************************************************
*  Function: loadVRMLComponents
*
*  This function need to be called before 'loadModel' to avoid
*  empty data pointers caused by 'removeModel'
*
***************************************************************/
void MazeFileImporter::loadVRMLComponents()
{
    /* preload WRL components */
    mCeilingNode = osgDB::readNodeFile(mDataDir + "Ceiling.WRL");
    mFloorNode = osgDB::readNodeFile(mDataDir + "Floor.WRL");
    mCloseWallNode = osgDB::readNodeFile(mDataDir + "CloseWall.WRL");
    mOpenWallNode = osgDB::readNodeFile(mDataDir + "OpenWall.WRL");
    mCloseWallWhiteNode = osgDB::readNodeFile(mDataDir + "CloseWallWhite.WRL");
    mOpenWallWhiteNode = osgDB::readNodeFile(mDataDir + "OpenWallWhite.WRL");
    mWallPaintFrameNode = osgDB::readNodeFile(mDataDir + "WallPaintFrame.WRL");

    mLeftRedDoorNode = osgDB::readNodeFile(mDataDir + "DoorLeftLeafRED.WRL");
    mRightRedDoorNode = osgDB::readNodeFile(mDataDir + "DoorRightLeafRED.WRL");
    mLeftGreenDoorNode = osgDB::readNodeFile(mDataDir + "DoorLeftLeafGREEN.WRL");
    mRightGreenDoorNode = osgDB::readNodeFile(mDataDir + "DoorRightLeafGREEN.WRL");
    mLeftYellowDoorNode = osgDB::readNodeFile(mDataDir + "DoorLeftLeafYELLOW.WRL");
    mRightYellowDoorNode = osgDB::readNodeFile(mDataDir + "DoorRightLeafYELLOW.WRL");
    mLeftBlueDoorNode = osgDB::readNodeFile(mDataDir + "DoorLeftLeafBLUE.WRL");
    mRightBlueDoorNode = osgDB::readNodeFile(mDataDir + "DoorRightLeafBLUE.WRL");

    mWallPaint01Node = osgDB::readNodeFile(mDataDir + "WallPaint01.WRL");
    mWallPaint02Node = osgDB::readNodeFile(mDataDir + "WallPaint02.WRL");
    mWallPaint03Node = osgDB::readNodeFile(mDataDir + "WallPaint03.WRL");
    mWallPaint04Node = osgDB::readNodeFile(mDataDir + "WallPaint04.WRL");

    mColumnNode = osgDB::readNodeFile(mDataDir + "Column.WRL");
    mColumnWhiteNode = osgDB::readNodeFile(mDataDir + "ColumnWhite.WRL");
    mIndentDoorFrameNode = osgDB::readNodeFile(mDataDir + "IndentDoorFrame.WRL");
    mIndentDoorFrameWhiteNode = osgDB::readNodeFile(mDataDir + "IndentDoorFrameWhite.WRL");
}


/***************************************************************
*  Function: loadModel
***************************************************************/
void MazeFileImporter::loadModel(const std::string &filename)
{
    ifstream inFile;
    inFile.open(filename.c_str());
    if (!inFile) {
        cout << "Unable to open maze model file " << filename << endl;
        return;
    }

    /* keep iterating through model file searching for all valid entries */
    string entryNameStr;
    while (inFile >> entryNameStr)
    {
	EntryPosInfo *infoPtr = new EntryPosInfo();
	EntryNodeList nodelist, hintlist, texturedNodeList, nonTexturedNodeList;
	if (lookupEntryNode(entryNameStr, infoPtr, nodelist, hintlist, texturedNodeList, nonTexturedNodeList, &inFile));
	{
	    /* translate EntryPosInfo into matrix transforms */
	    Matrixd localOffsetMat, rotMat, gridOffsetMat;
	    localOffsetMat.makeTranslate(Vec3(gGridSize * 0.5f, 0, 0));
	    rotMat.makeRotate((int)(infoPtr->eo) * M_PI * 0.5f, Vec3(0, 0, 1));
	    gridOffsetMat.makeTranslate(Vec3(infoPtr->x, infoPtr->y, 0) * gGridSize);
	    Matrixd transmat = localOffsetMat * rotMat * gridOffsetMat;

	    /* load component VRML files into switches of different categories */
	    if (nodelist.size() > 0)
	    {
		MatrixTransform *matTrans = new MatrixTransform;
		matTrans->setMatrix(transmat);
		for (EntryNodeList::const_iterator itrNode = nodelist.begin(); itrNode != nodelist.end(); itrNode++)
		    matTrans->addChild(*itrNode);
		mStillModelSwitch->addChild(matTrans);
	    }
	    if (hintlist.size() > 0)
	    {
		MatrixTransform *matTrans = new MatrixTransform;
		matTrans->setMatrix(transmat);
		for (EntryNodeList::const_iterator itrNode = hintlist.begin(); itrNode != hintlist.end(); itrNode++)
		    matTrans->addChild(*itrNode);
		mLocalHintSwitch->addChild(matTrans);
	    }
	    if (texturedNodeList.size() > 0)
	    {
		MatrixTransform *matTrans = new MatrixTransform;
		matTrans->setMatrix(transmat);
		for (EntryNodeList::const_iterator itrNode = texturedNodeList.begin();
			itrNode != texturedNodeList.end(); itrNode++)
		    matTrans->addChild(*itrNode);
		mTextureWallSwitch->addChild(matTrans);
	    }
	    if (nonTexturedNodeList.size() > 0)
	    {
		MatrixTransform *matTrans = new MatrixTransform;
		matTrans->setMatrix(transmat);
		for (EntryNodeList::const_iterator itrNode = nonTexturedNodeList.begin();
			itrNode != nonTexturedNodeList.end(); itrNode++)
		    matTrans->addChild(*itrNode);
		mNonTextureWallSwitch->addChild(matTrans);
	    }
	}
	delete infoPtr;

	nodelist.clear();
    }
    inFile.close();
}


/***************************************************************
*  Function: removeModel
***************************************************************/
void MazeFileImporter::removeModel()
{
    /* remove all loaded components from root switches */
    int numGlobalHints = mGlobalHintSwitch->getNumChildren();
    if (numGlobalHints > 0) mGlobalHintSwitch->removeChildren(0, numGlobalHints);

    int numLocalHints = mLocalHintSwitch->getNumChildren();
    if (numLocalHints > 0) mLocalHintSwitch->removeChildren(0, numLocalHints);

    int numStillComponents = mStillModelSwitch->getNumChildren();
    if (numStillComponents > 0) mStillModelSwitch->removeChildren(0, numStillComponents);

    int numTextureComponents = mTextureWallSwitch->getNumChildren();
    if (numTextureComponents > 0) mTextureWallSwitch->removeChildren(0, numTextureComponents);

    int numNonTextureComponents = mNonTextureWallSwitch->getNumChildren();
    if (numNonTextureComponents > 0) mNonTextureWallSwitch->removeChildren(0, numNonTextureComponents);

    mStartPosVector.clear();
}


/***************************************************************
*  Function: getStartPosMatrix
***************************************************************/
Matrixd MazeFileImporter::getStartPosMatrix(const StartPosition &pos, const float &scale)
{
    int idx = (int) pos;
    Matrixd mat, rotMat, offMat;

    if (pos == RANDOM) idx = (int)((float) (random()) / RAND_MAX * 8);
    if (idx >= mStartPosVector.size()) return mat;

    if (mStartPosVector[idx]->eo == EAST) rotMat.makeRotate(M_PI * 0.5f, Vec3(0, 0, 1));
    else if (mStartPosVector[idx]->eo == NORTH) rotMat.makeRotate(0, Vec3(0, 0, 1));
    else if (mStartPosVector[idx]->eo == WEST) rotMat.makeRotate(-M_PI * 0.5f, Vec3(0, 0, 1));
    else if (mStartPosVector[idx]->eo == SOUTH) rotMat.makeRotate(M_PI, Vec3(0, 0, 1));
    offMat.makeTranslate(Vec3((-mStartPosVector[idx]->x) * gGridSize, (-mStartPosVector[idx]->y) * gGridSize, -1.25) * scale);
    return offMat * rotMat;
}


/***************************************************************
*  Function: lookupEntryNode
***************************************************************/
bool MazeFileImporter::lookupEntryNode(const string &entryname, EntryPosInfo *infoPtr, EntryNodeList &nodelist,
	EntryNodeList &hintlist, EntryNodeList &texturedNodeList, EntryNodeList &nonTexturedNodeList, ifstream *inFilePtr)
{
    string orient;

    /* for normal object, push associated nodes in to 'nodelist' and return 'true' */
    if (!strcmp(entryname.c_str(), "StartPos"))
    {
	EntryPosInfo *info = new EntryPosInfo;
	(*inFilePtr) >> info->x;
	(*inFilePtr) >> info->y;
	(*inFilePtr) >> orient;
	info->setOrientation(orient);
	mStartPosVector.push_back(info);
	return false;
    }
    else if (!strcmp(entryname.c_str(), "Ceiling"))
    {
	nodelist.push_back(mCeilingNode);
	(*inFilePtr) >> infoPtr->x;
	(*inFilePtr) >> infoPtr->y;
    }
    else if (!strcmp(entryname.c_str(), "Floor"))
    {
	nodelist.push_back(mFloorNode);
	(*inFilePtr) >> infoPtr->x;
	(*inFilePtr) >> infoPtr->y;
    } 
    else if (!strcmp(entryname.c_str(), "Close_Wall"))
    {
	texturedNodeList.push_back(mCloseWallNode);
	nonTexturedNodeList.push_back(mCloseWallWhiteNode);

	(*inFilePtr) >> infoPtr->x;
	(*inFilePtr) >> infoPtr->y;
	(*inFilePtr) >> orient;
	infoPtr->setOrientation(orient);
    } 
    else if (!strcmp(entryname.c_str(), "Open_Wall"))
    {
	texturedNodeList.push_back(mOpenWallNode);
	nonTexturedNodeList.push_back(mOpenWallWhiteNode);

	(*inFilePtr) >> infoPtr->x;
	(*inFilePtr) >> infoPtr->y;
	(*inFilePtr) >> orient;
	infoPtr->setOrientation(orient);
    }
    else if (!strcmp(entryname.c_str(), "Paint"))
    {
	(*inFilePtr) >> infoPtr->x;
	(*inFilePtr) >> infoPtr->y;
	(*inFilePtr) >> orient;
	infoPtr->setOrientation(orient);

	int idx;
	(*inFilePtr) >> idx;
	if (idx == 1) hintlist.push_back(mWallPaint01Node);
	else if (idx == 2) hintlist.push_back(mWallPaint02Node);
	else if (idx == 3) hintlist.push_back(mWallPaint03Node);
	else if (idx == 4) hintlist.push_back(mWallPaint04Node);

	hintlist.push_back(mWallPaintFrameNode);	
    }
    else if (!strcmp(entryname.c_str(), "Door"))
    {
	(*inFilePtr) >> infoPtr->x;
	(*inFilePtr) >> infoPtr->y;
	(*inFilePtr) >> orient;
	infoPtr->setOrientation(orient);

	string color;
	(*inFilePtr) >> color;
	if (!strcmp(color.c_str(), "Red"))
	{
	    nodelist.push_back(mLeftRedDoorNode);
	    nodelist.push_back(mRightRedDoorNode);
	}
	else if (!strcmp(color.c_str(), "Green"))
	{
	    nodelist.push_back(mLeftGreenDoorNode);
	    nodelist.push_back(mRightGreenDoorNode);
	}
	else if (!strcmp(color.c_str(), "Yellow"))
	{
	    nodelist.push_back(mLeftYellowDoorNode);
	    nodelist.push_back(mRightYellowDoorNode);
	}
	else if (!strcmp(color.c_str(), "Blue"))
	{
	    nodelist.push_back(mLeftBlueDoorNode);
	    nodelist.push_back(mRightBlueDoorNode);
	}
	else	// push blue doors into the list by default
	{
	    nodelist.push_back(mLeftBlueDoorNode);
	    nodelist.push_back(mRightBlueDoorNode);
	}

	texturedNodeList.push_back(mIndentDoorFrameNode);
	nonTexturedNodeList.push_back(mIndentDoorFrameWhiteNode);
    }

    /* for general information, do not modify 'nodelist', return false */
    else if (!strcmp(entryname.c_str(), "FloorPlan") || !strcmp(entryname.c_str(), "CeilingPlan"))
    {
	int xmin, ymin, xmax, ymax;
	(*inFilePtr) >> xmin;
	(*inFilePtr) >> ymin;
	(*inFilePtr) >> xmax;
	(*inFilePtr) >> ymax;

	/* duplicate floor nodes and add them directly into 'mStillModelSwitch' */
	for (int i = xmin; i <= xmax; i++)
	{
	    for (int j = ymin; j <= ymax; j++)
	    {
		MatrixTransform *matTrans = new MatrixTransform;
		Matrixd offsetMat;
		offsetMat.makeTranslate(Vec3(gGridSize * 0.5f, 0, 0) + Vec3(i, j, 0) * gGridSize);
		matTrans->setMatrix(offsetMat);

		if (!strcmp(entryname.c_str(), "FloorPlan")) matTrans->addChild(mFloorNode);
		if (!strcmp(entryname.c_str(), "CeilingPlan")) matTrans->addChild(mCeilingNode);
		mStillModelSwitch->addChild(matTrans);
	    }
	}

	/* create columes with respect to sizes of the floorplan */
	if (!strcmp(entryname.c_str(), "FloorPlan"))
	{
	    for (int i = xmin; i < xmax; i++)
	    {
		for (int j = ymin; j < ymax; j++)
		{
		    MatrixTransform *matTrans = new MatrixTransform;
		    MatrixTransform *_matTrans = new MatrixTransform;
		    Matrixd offsetMat;
		    offsetMat.makeTranslate(Vec3(gGridSize * 0.5f, gGridSize * 0.5f, 0) + Vec3(i, j, 0) * gGridSize);
		    matTrans->setMatrix(offsetMat);
		    _matTrans->setMatrix(offsetMat);

		    matTrans->addChild(mColumnNode);
		    _matTrans->addChild(mColumnWhiteNode);
		    mTextureWallSwitch->addChild(matTrans);
		    mNonTextureWallSwitch->addChild(_matTrans);
		}
	    }
	}

	return false;
    }
    else return false;

    return true;
}
















