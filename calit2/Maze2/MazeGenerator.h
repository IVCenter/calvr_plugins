// MazeGenerator.h
#ifndef _MAZE_GENERATOR_H
#define _MAZE_GENERATOR_H

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <list>
#include <queue>

/**************** Class Prototypes ****************/
class MazeGenerator;
class MazeRoom;
class MazeNgbRoomInfo;
class MazeGrid;
class MazeNgbGridInfo;
class MazeObject;

/** Enumeration MAZDirectionType
*/
enum MAZDirectionType
{
	MAZ_NORTH = 0,
	MAZ_EAST = 1,
	MAZ_SOUTH = 2,
	MAZ_WEST = 3,
	MAZ_CENTER = 4,
	MAZ_NO_DIRECTION = 5
};

/** Enumeration MAZObjectType
    Types of objects that can be placed in maze
*/
enum MAZObjectType
{
	MAZ_CLOSE_WALL,
	MAZ_OPEN_WALL,
	MAZ_LOCKED_DOOR,
	MAZ_UNLOCKED_DOOR,
	MAZ_WINDOW,
	MAZ_FLOOR,
	MAZ_CEILING,
};

/** class MazeObject:
*/
class MazeObject
{
public:
	MazeObject(MAZDirectionType dir, MAZObjectType obj);
	MAZDirectionType mDir;
	MAZObjectType mObj;
};

/** class MazeGrid:
*/
class MazeGrid
{
public:
	MazeGrid(int col, int row);
	int mx, my, roomID;
	int ngbGridID[4];		// roomIDs of the four grid neighbors, recorded from NORTH to EAST
	bool mFlag;				// note if the grid has been marked with a room index
	std::list<MazeObject*> mObjList;		// put arbitrary objects in grid: extended  for decoration use
};

/** class MazeNgbGridInfo:
*/
class MazeNgbGridInfo
{
public:
	MazeNgbGridInfo(MazeGrid* ngbGridPtr, MAZDirectionType dir);
	MazeGrid* mNgbGridPtr;
	MAZDirectionType mDir;
	bool isOpen;	
};


/** class MazeRoom:
*/
class MazeRoom
{
public:
	MazeRoom(int ID, MazeGenerator*  refPtrMazeGenerator);

	int nGrid, mID;
	bool saturFlag;		// true if the room can no longer grow
	void addGrid(MazeGrid* grid);
	void setNeighborInfo();
	MazeGrid* getGrowableNeighborGrid();
	MazeNgbRoomInfo* getNgbRoomInfo(const int roomID);

	std::list<MazeGrid*> mGridList;
	std::list<MazeNgbRoomInfo*> mNgbRoomInfoList;	// room neighbors are listed with increasing roomIDs

protected:
	MazeGenerator* mMazeGenerator;
	std::priority_queue<int> mNgbRoomIDList;
};

/** class MazeNgbRoomInfo:
*/
class MazeNgbRoomInfo
{
public:
	MazeNgbRoomInfo(MazeRoom* ngbRoomPtr);
	void addNgbGridInfo(MazeNgbGridInfo* ngbGridInfo);
	MazeNgbGridInfo* getNgbGridInfo(const MazeGrid* grid, const MAZDirectionType dir);

	MazeRoom* mNgbRoomPtr;
	MazeNgbGridInfo* mOpenNgbGridInfo;		// grid info that open to adjacent room
	std::list<MazeNgbGridInfo*> mNgbGridInfoList;
};


/** class MazeGenerator:
*/
class MazeGenerator
{
	friend class MazeRoom;
public:
	MazeGenerator(const char* outputFileName);
	~MazeGenerator();
	void resize(const int &size);
	void buildStructure();
	void exportToDSNFile();
	void exportToMAZFile();


	static char* toString(MAZDirectionType dir);
	static MAZDirectionType toOpposite(MAZDirectionType dir);

protected:
	FILE *fptrDSN, *fptrMAZ;
	char fileName[256];
	int nGridRow, nGridCol, nRoom, nGrid;
	int nOpenWall, nCloseWall, nOpenDoor, nCloseDoor, nWindow;

	MazeRoom** mRoomInfo;
	MazeGrid*** mGridInfo;

	void setRoomIndex();
	void setOpenPaths();
	void setOpenPathPosition(int roomID1, int roomID2);
	void connectRooms(int roomID1, int roomID2);
	MazeGrid* getGridNeighbor(const MazeGrid* grid, const MAZDirectionType dir);
};

#endif
