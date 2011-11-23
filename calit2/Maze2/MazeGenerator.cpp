// MazeGenerator.cpp:
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "MazeGenerator.h"


using namespace std;

/** MazeGenerator: Constructor
*/
MazeGenerator::MazeGenerator(const char* outputFileName)
{
	strcpy(fileName, outputFileName);

	/* Default maze parameters */
	nGridRow = 40;
	nGridCol = 40;
	nRoom = 300;
	nGrid = nGridRow * nGridCol;

	nOpenWall = 0;		nCloseWall = 0;
	nOpenDoor = 0;		nCloseDoor = 0;
	nWindow = 0;
}


/** MazeGenerator: Destructor
*/
MazeGenerator::~MazeGenerator()
{
	for (int i = 0; i < nRoom; i++) delete mRoomInfo[i];
	delete []mRoomInfo;

	for (int i = 0; i < nGridCol; i++)
	{
		for (int j = 0; j < nGridRow; j++)
		{
			delete mGridInfo[i][j];
		}
	}
	for (int i = 0; i < nGridCol; i++) 
	{
		delete mGridInfo[i];
	}
	delete []mGridInfo;
}


/** MazeGenerator: resize
*/
void MazeGenerator::resize(const int &size)
{
	nGridRow = size;
	nGridCol = size;
	nRoom = size * size / 8;
	nGrid = nGridRow * nGridCol;
}


/** MazeGenerator: buildStructure()
    build maze structure using initial parameters
*/
void MazeGenerator::buildStructure()
{
	/* memory pre-allocation: use 2D array for mGridInfo */
	mGridInfo = new MazeGrid**[nGridCol];
	for (int i = 0; i < nGridCol; i++) 
	{
		mGridInfo[i] = new MazeGrid*[nGridRow];
	}
	for (int i = 0; i < nGridCol; i++)
	{
		for (int j = 0; j < nGridRow; j++)
		{
			mGridInfo[i][j] = new MazeGrid(i, j);
		}
	}

	mRoomInfo = new MazeRoom*[nRoom];
	for (int i = 0; i < nRoom; i++) 
	{
		mRoomInfo[i] = new MazeRoom(i, this);
	}

	setRoomIndex();
	for (int i = 0; i < nRoom; i++)  mRoomInfo[i]->setNeighborInfo();
	setOpenPaths();
	nCloseWall = nCloseWall - nOpenWall + (nGridRow + nGridCol) * 4;
}


/** MazeGenerator: setRoomIndex()
    assign grids with room index numbers
*/
void MazeGenerator::setRoomIndex()
{
	time_t Time;
	srand((int) time(&Time));

	/* setup seed points */
	int seedCnt = 0, randX, randY;
	while (seedCnt < nRoom)
	{
		randX = (int) (((float) rand() - 1.f) / RAND_MAX * nGridCol);
		randY = (int) (((float) rand() - 1.f) / RAND_MAX * nGridRow);
		randX = randX >= nGridCol ? (nGridCol-1) : randX;
		randY = randY >= nGridRow ? (nGridRow-1) : randY;

		if ( ! mGridInfo[randX][randY]->mFlag )
		{
			mRoomInfo[seedCnt]->addGrid(mGridInfo[randX][randY]);
			seedCnt++;
		}
	}

	/* grids saturation */
	int nSteps = nGrid - nRoom;		// number of grids that to be saturated
	int nSatur = 0, itrID = 0;

	while (nSatur < nSteps)
	{
		if (++itrID >= nRoom) itrID = 0;		// loop through all rooms using room index iterator
		if ( !mRoomInfo[itrID]->saturFlag )
		{
			MazeGrid* saturGrid = mRoomInfo[itrID]->getGrowableNeighborGrid();
			if (saturGrid) {
				int mx = saturGrid->mx;
				int my = saturGrid->my;
				mRoomInfo[itrID]->addGrid(mGridInfo[mx][my]);
				nSatur++;
			} else {
				mRoomInfo[itrID]->saturFlag = true;
			}
		} 
	}
}


/** MazeGenerator: setOpenPaths()
    set connections between adjacent rooms
*/
void MazeGenerator::setOpenPaths()
{
	for (int i = 0; i < nRoom; i++)
	{
		/* Iterate through all MazeNgbRoomInfo of each room */
		std::list<MazeNgbRoomInfo*>::const_iterator itrNgbRoomInfo;
		for (itrNgbRoomInfo = mRoomInfo[i]->mNgbRoomInfoList.begin(); 
			itrNgbRoomInfo != mRoomInfo[i]->mNgbRoomInfoList.end(); itrNgbRoomInfo++)
		{
			MazeNgbRoomInfo* curNgbRoomInfo = (*itrNgbRoomInfo);
			int ngbID = curNgbRoomInfo->mNgbRoomPtr->mID;
			if (i < ngbID) 
			{
				setOpenPathPosition(i, ngbID);
				nOpenWall += 2;
				nOpenDoor++;
				connectRooms(i, ngbID);
			}
		}
	}
}


/** MazeGenerator: setOpenPathPosition()
    set an open path position between adjacent rooms roomID1 and roomID2
*/
void MazeGenerator::setOpenPathPosition(int roomID1, int roomID2)
{
	MazeNgbRoomInfo* ngbRoomInfo1 = mRoomInfo[roomID1]->getNgbRoomInfo(roomID2);
	MazeNgbRoomInfo* ngbRoomInfo2 = mRoomInfo[roomID2]->getNgbRoomInfo(roomID1);
	if (!ngbRoomInfo1 || !ngbRoomInfo2)
	{
		printf("Failed to set open path between room %d and room %d. \n", roomID1, roomID2);
		return;
	}

	/* select an open position from neighbor info list 1*/
	MazeNgbGridInfo* ngbGridInfo1 = *(ngbRoomInfo1->mNgbGridInfoList.begin());
	ngbRoomInfo1->mOpenNgbGridInfo = ngbGridInfo1;

	MazeGrid* ngbGrid1 = ngbGridInfo1->mNgbGridPtr;
	MazeGrid* ngbGrid2 = getGridNeighbor(ngbGrid1, ngbGridInfo1->mDir);

	if (!ngbGrid1 || !ngbGrid2)	return;

	MazeNgbGridInfo* ngbGridInfo2 = ngbRoomInfo2->getNgbGridInfo(ngbGrid2, toOpposite(ngbGridInfo1->mDir));
	ngbRoomInfo2->mOpenNgbGridInfo = ngbGridInfo2;
}


/** MazeGenerator: connectRooms()
    connect rooms pair 'roomID1' and 'roomID2'
*/
void MazeGenerator::connectRooms(int roomID1, int roomID2)
{
	MazeNgbGridInfo* ngbGridInfo1 = mRoomInfo[roomID1]->getNgbRoomInfo(roomID2)->mOpenNgbGridInfo;
	MazeNgbGridInfo* ngbGridInfo2 = mRoomInfo[roomID2]->getNgbRoomInfo(roomID1)->mOpenNgbGridInfo;

	ngbGridInfo1->isOpen = true;
	ngbGridInfo2->isOpen = true;
}


/** MazeGenerator: getGridNeighbor()
    get grid neighbor  by specifying the direction, return NULL if reaching boundary
*/
MazeGrid* MazeGenerator::getGridNeighbor(const MazeGrid* grid, const MAZDirectionType dir)
{
	int x = grid->mx, y = grid->my;

	if ( dir == MAZ_NORTH) { y++; }
	else if ( dir == MAZ_EAST) { x++; }
	else if ( dir == MAZ_SOUTH) { y--; }
	else if ( dir == MAZ_WEST) { x--; }
	else if ( dir == MAZ_CENTER) { }
	else return NULL;

	if (x >= nGridCol || x < 0 ||  y >= nGridRow || y < 0)  return NULL;
	return mGridInfo[x][y];
}


/** MazeRoom: Constructor
    bind reference pointer to top level MazeGenerator
*/
MazeRoom::MazeRoom(int ID, MazeGenerator*  refPtrMazeGenerator): nGrid(0), saturFlag(false)
{
	mID = ID;
	mMazeGenerator = refPtrMazeGenerator;
}


/** MazeRoom: addGrid()
    add a single grid to the grid list
*/
void MazeRoom::addGrid(MazeGrid* grid)
{
	nGrid++;
	grid->mFlag = true;
	grid->roomID = mID;
	mGridList.push_back(grid);
}



/** MazeRoom: setNeighborInfo()
    check surrounding grids and push them to neighbor list
*/
void MazeRoom::setNeighborInfo()
{
	int ngbRoomID;
	MazeGrid* ngbGrid;
	std::list<MazeGrid*>::const_iterator itrGrid;

	/* Step 1: check all grid neighbors of a room: push room IDs to  'mNgbRoomIDList' */
	for (itrGrid = mGridList.begin(); itrGrid != mGridList.end(); itrGrid++)
	{
		MazeGrid* curGrid = (*itrGrid);
		for (int dir = MAZ_NORTH; dir <= MAZ_WEST; dir++)
		{		
			ngbGrid = mMazeGenerator->getGridNeighbor(curGrid, (MAZDirectionType)dir);
			if (ngbGrid)
			{
				ngbRoomID = ngbGrid->roomID;		
				if (ngbRoomID != mID) 
				{
					mNgbRoomIDList.push(ngbRoomID);
				}
				curGrid->ngbGridID[dir] = ngbRoomID;
			} else {
				curGrid->ngbGridID[dir] = -1;		// set neighbor for boundary grid
			}
		}
	}

	/* Step 2: Build neighbor room info list, assign MazeRoom pointer to each  MazeNgbRoomInfo */
	int ngbID, itrID = -1;
	while (!mNgbRoomIDList.empty())
	{	
		ngbID = mNgbRoomIDList.top();
		if (itrID != ngbID) 
		{
			itrID = ngbID;
			MazeNgbRoomInfo *ngbRoomInfo = new MazeNgbRoomInfo(mMazeGenerator->mRoomInfo[itrID]);
			mNgbRoomInfoList.push_front(ngbRoomInfo);
		}	
		mNgbRoomIDList.pop();
	}

	/* Step 3: Re-check grid neighbors and fiil in 'mNgbGridInfoList' of each MazeNgbRoomInfo */
	for (itrGrid = mGridList.begin(); itrGrid != mGridList.end(); itrGrid++)
	{
		MazeGrid* curGrid = (*itrGrid);
		for (int dir = MAZ_NORTH; dir <= MAZ_WEST; dir++)
		{		
			ngbGrid = mMazeGenerator->getGridNeighbor(curGrid, (MAZDirectionType)dir);
			if (ngbGrid)
			{
				ngbRoomID = ngbGrid->roomID;		
				if (ngbRoomID != mID) 
				{
					MazeNgbRoomInfo* ngbRoomInfo =  getNgbRoomInfo(ngbRoomID);
					MazeNgbGridInfo* ngbGridInfo = new MazeNgbGridInfo(curGrid, (MAZDirectionType)dir);
					ngbRoomInfo->addNgbGridInfo(ngbGridInfo);
					(mMazeGenerator->nCloseWall)++;
				}
			} 
		}
	}
}


/** MazeRoom: getGrowableNeighborGrid()
    get one growable grid from its neighborhood
*/
MazeGrid* MazeRoom::getGrowableNeighborGrid()
{
	/* find one neighbor that has not been saturated */
	MazeGrid* ngbGrid;
	std::list<MazeGrid*>::const_iterator itrGrid;
	for (itrGrid = mGridList.begin(); itrGrid != mGridList.end(); itrGrid++)
	{
		MazeGrid* curGrid = (*itrGrid);
		for (int dir = MAZ_NORTH; dir <= MAZ_WEST; dir++)
		{		
			ngbGrid = mMazeGenerator->getGridNeighbor(curGrid, (MAZDirectionType)dir);
			if (ngbGrid)
			{
				if (!ngbGrid->mFlag) return ngbGrid;
			}
		}
	}
	return NULL;
}

/** MazeRoom: getNgbRoomInfo()
    fetch a MazeNgbRoomInfo pointer from 'mNgbRoomInfoList' given neighbor room ID
*/
MazeNgbRoomInfo* MazeRoom::getNgbRoomInfo(const int roomID)
{
	std::list<MazeNgbRoomInfo*>::const_iterator itrNgbRoomInfo;
	for (itrNgbRoomInfo = mNgbRoomInfoList.begin(); itrNgbRoomInfo != mNgbRoomInfoList.end(); itrNgbRoomInfo++)
	{
		MazeNgbRoomInfo* curNbgRoomInfo = (*itrNgbRoomInfo);

		int curID = curNbgRoomInfo->mNgbRoomPtr->mID;
		if (curID == roomID) return curNbgRoomInfo;
	}
	return NULL;
}


/** MazeNgbRoomInfo: Constructor
*/
MazeNgbRoomInfo::MazeNgbRoomInfo(MazeRoom* ngbRoomPtr): mNgbRoomPtr(NULL), mOpenNgbGridInfo(NULL)
{ 
	mNgbRoomPtr = ngbRoomPtr; 
}


/** MazeNgbRoomInfo: addNgbGridInfo
*/
void MazeNgbRoomInfo::addNgbGridInfo(MazeNgbGridInfo* ngbGridInfo)
{
	mNgbGridInfoList.push_back(ngbGridInfo);
}


/** MazeNgbRoomInfo: getNgbGridInfo
*   get the neighbor grid info given reference grid pointer and direction
*/
MazeNgbGridInfo* MazeNgbRoomInfo::getNgbGridInfo(const MazeGrid* grid, const MAZDirectionType dir)
{
	
	std::list<MazeNgbGridInfo*>::const_iterator itrNgbGridInfo;
	for (itrNgbGridInfo = mNgbGridInfoList.begin(); itrNgbGridInfo != mNgbGridInfoList.end(); itrNgbGridInfo++)
	{
		MazeNgbGridInfo* ngbGridInfo = (*itrNgbGridInfo);
		MazeGrid* gridRef = ngbGridInfo->mNgbGridPtr;
		if (gridRef->mx == grid->mx && gridRef->my == grid->my && ngbGridInfo->mDir == dir)
			return ngbGridInfo;
	}			
	return NULL;
}


/** MazeNgbGridInfo: Constructor
*/
MazeNgbGridInfo::MazeNgbGridInfo(MazeGrid* ngbGridPtr, MAZDirectionType dir): isOpen(false)
{
	mNgbGridPtr = ngbGridPtr;
	mDir = dir;	
}


/** MazeGrid: Constructor
*/
MazeGrid::MazeGrid(int col, int row):mx(col), my(row), roomID(-1), mFlag(false) { }


/** MazeObject: Constructor
*/
MazeObject::MazeObject(MAZDirectionType dir, MAZObjectType obj): mDir(dir), mObj(obj) { }


