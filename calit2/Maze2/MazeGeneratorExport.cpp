// MazeGeneratorExport.cpp:
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "MazeGenerator.h"


/** MazeGenerator: exportToDSNFile()
    export existing structure to .DSN file
*/
void MazeGenerator::exportToDSNFile()
{
	char fileNameDSN[128];
	sprintf(fileNameDSN, "%s.MAZ", fileName);
	fptrDSN=fopen(fileNameDSN, "wb");

	/*------------------------ header information ------------------------
	fprintf(fptrDSN, "#HEADER_START\n");
	fprintf(fptrDSN, "NUM_ROW %d\n", nGridRow);
	fprintf(fptrDSN, "NUM_COL %d\n", nGridCol);
	fprintf(fptrDSN, "NUM_ROOM %d\n", nRoom);
	fprintf(fptrDSN, "NUM_FLOOR %d\n", nGrid);
	fprintf(fptrDSN, "NUM_CEILING %d\n", nGrid);
	fprintf(fptrDSN, "NUM_CLOSE_WALL %d\n", nCloseWall);
	fprintf(fptrDSN, "NUM_OPEN_WALL %d\n", nOpenWall);
	fprintf(fptrDSN, "NUM_LOCKED_DOOR %d\n", nCloseDoor);
	fprintf(fptrDSN, "NUM_UNLOCKED_DOOR %d\n", nOpenDoor);
	fprintf(fptrDSN, "NUM_WINDOW %d\n", nWindow);
	fprintf(fptrDSN, "#HEADER_END\n\n");*/

const int nRowOff = nGridRow / 2;
const int nColOff = nGridCol / 2;
fprintf(fptrDSN, "FloorPlan %d %d %d %d\n", -nColOff, -nRowOff, nGridCol-nColOff, nGridRow-nRowOff);
fprintf(fptrDSN, "StartPos %d %d N\n", -nColOff+1, -nRowOff);
fprintf(fptrDSN, "StartPos %d %d N\n", nColOff-1, -nRowOff);
fprintf(fptrDSN, "StartPos %d %d E\n", -nColOff, -nRowOff+1);
fprintf(fptrDSN, "StartPos %d %d E\n", -nColOff, nRowOff-1);
fprintf(fptrDSN, "StartPos %d %d S\n", -nColOff+1, nRowOff);
fprintf(fptrDSN, "StartPos %d %d S\n", nColOff-1, nRowOff);
fprintf(fptrDSN, "StartPos %d %d W\n", nColOff, nRowOff-1);
fprintf(fptrDSN, "StartPos %d %d W\n", nColOff, -nRowOff+1);

	/*------------------------ room information ------------------------
	fprintf(fptrDSN, "Room_Info\n{\n");
	for (int i = 0; i < nGridCol; i++)
		for (int j = 0; j < nGridRow; j++)
			fprintf(fptrDSN, "\t%d %d %d\n", i, j, mGridInfo[i][j]->roomID);
	// fprintf(fptrDSN, "}\n\n");*/

	/*------------------------ room neighbor information ------------------------
	fprintf(fptrDSN, "Room_Neighbor_Info\n{\n");
	for (int i = 0; i < nRoom; i++)
	{
		fprintf(fptrDSN, "\t%d", mRoomInfo[i]->mID);	// roomID
		fprintf(fptrDSN, "\t%d\t", (int)(mRoomInfo[i]->mNgbRoomInfoList.size()));	// number of neighbors
		std::list<MazeNgbRoomInfo*>::const_iterator itrNgbRoomInfo;
		for (itrNgbRoomInfo = mRoomInfo[i]->mNgbRoomInfoList.begin(); 
			itrNgbRoomInfo != mRoomInfo[i]->mNgbRoomInfoList.end(); itrNgbRoomInfo++)
		{
			std::list<MazeNgbGridInfo*>::const_iterator itrNgbGridInfo;
			MazeNgbRoomInfo* curNgbRoomInfo = (*itrNgbRoomInfo);
			fprintf(fptrDSN, "%d ", curNgbRoomInfo->mNgbRoomPtr->mID);	// neighbor roomID;
		}
		fprintf(fptrDSN, "\n");
	}
	// fprintf(fptrDSN, "}\n\n");*/

	/*------------------------ close_wall ------------------------
	fprintf(fptrDSN, "Close_Wall\n{\n");*/

	/* close_wall between adjacent rooms */
	for (int i = 0; i < nRoom; i++)	
	{
		std::list<MazeNgbRoomInfo*>::const_iterator itrNgbRoomInfo;
		for (itrNgbRoomInfo = mRoomInfo[i]->mNgbRoomInfoList.begin(); 
			itrNgbRoomInfo != mRoomInfo[i]->mNgbRoomInfoList.end(); itrNgbRoomInfo++)
		{
			std::list<MazeNgbGridInfo*>::const_iterator itrNgbGridInfo;
			MazeNgbRoomInfo* curNgbRoomInfo = (*itrNgbRoomInfo);
			for (itrNgbGridInfo = curNgbRoomInfo->mNgbGridInfoList.begin(); 
				itrNgbGridInfo != curNgbRoomInfo->mNgbGridInfoList.end(); itrNgbGridInfo++)
			{
				MazeNgbGridInfo* curNgbGridInfo = (*itrNgbGridInfo);
				if ( !curNgbGridInfo->isOpen )
				{
					int mx = curNgbGridInfo->mNgbGridPtr->mx;
					int my = curNgbGridInfo->mNgbGridPtr->my;
					int dir = curNgbGridInfo->mDir;
					// fprintf(fptrDSN, "\t%d %d %s\n", mx, my, toString((MAZDirectionType)dir));
fprintf(fptrDSN, "Close_Wall %d %d %s\n", mx-nColOff, my-nRowOff, toString((MAZDirectionType)dir));
				}
			}
		}
	}

	/* close_wall on boundaries */
	for (int i = 0; i < nGridCol; i++)
	{
/*
		fprintf(fptrDSN, "\t%d %d %s\n", i, -1, toString(MAZ_NORTH));
		fprintf(fptrDSN, "\t%d %d %s\n", i, 0, toString(MAZ_SOUTH));
		fprintf(fptrDSN, "\t%d %d %s\n", i, nGridRow-1, toString(MAZ_NORTH));
		fprintf(fptrDSN, "\t%d %d %s\n", i, nGridRow, toString(MAZ_SOUTH));
*/
		fprintf(fptrDSN, "Close_Wall %d %d %s\n", i-nColOff, -1-nRowOff, toString(MAZ_NORTH));
		fprintf(fptrDSN, "Close_Wall %d %d %s\n", i-nColOff, 0-nRowOff, toString(MAZ_SOUTH));
		fprintf(fptrDSN, "Close_Wall %d %d %s\n", i-nColOff, nGridRow-1-nRowOff, toString(MAZ_NORTH));
		fprintf(fptrDSN, "Close_Wall %d %d %s\n", i-nColOff, nGridRow-nRowOff, toString(MAZ_SOUTH));
	}
	for (int j = 0; j < nGridRow; j++)
	{
/*
		fprintf(fptrDSN, "\t%d %d %s\n", -1, j, toString(MAZ_EAST));
		fprintf(fptrDSN, "\t%d %d %s\n", 0, j, toString(MAZ_WEST));
		fprintf(fptrDSN, "\t%d %d %s\n", nGridCol-1, j, toString(MAZ_EAST));
		fprintf(fptrDSN, "\t%d %d %s\n", nGridCol, j, toString(MAZ_WEST));
*/
		fprintf(fptrDSN, "Close_Wall %d %d %s\n", -1-nColOff, j-nRowOff, toString(MAZ_EAST));
		fprintf(fptrDSN, "Close_Wall %d %d %s\n", 0-nColOff, j-nRowOff, toString(MAZ_WEST));
		fprintf(fptrDSN, "Close_Wall %d %d %s\n", nGridCol-1-nColOff, j-nRowOff, toString(MAZ_EAST));
		fprintf(fptrDSN, "Close_Wall %d %d %s\n", nGridCol-nColOff, j-nRowOff, toString(MAZ_WEST));
	}

	// fprintf(fptrDSN, "}\n\n");

	/*------------------------ open_wall ------------------------
	fprintf(fptrDSN, "Open_Wall\n{\n");*/
	for (int i = 0; i < nRoom; i++)
	{
		std::list<MazeNgbRoomInfo*>::const_iterator itrNgbRoomInfo;
		for (itrNgbRoomInfo = mRoomInfo[i]->mNgbRoomInfoList.begin(); 
			itrNgbRoomInfo != mRoomInfo[i]->mNgbRoomInfoList.end(); itrNgbRoomInfo++)
		{
			MazeNgbRoomInfo* curNgbRoomInfo = (*itrNgbRoomInfo);
			int ngbID = curNgbRoomInfo->mNgbRoomPtr->mID;
			if (i < ngbID) 
			{
				int mx1, my1, dir1, mx2, my2, dir2;
				MazeNgbGridInfo* ngbGridInfo1 = mRoomInfo[i]->getNgbRoomInfo(ngbID)->mOpenNgbGridInfo;
				MazeNgbGridInfo* ngbGridInfo2 = mRoomInfo[ngbID]->getNgbRoomInfo(i)->mOpenNgbGridInfo;
				mx1 = ngbGridInfo1->mNgbGridPtr->mx;			mx2 = ngbGridInfo2->mNgbGridPtr->mx;
				my1 = ngbGridInfo1->mNgbGridPtr->my;			my2 = ngbGridInfo2->mNgbGridPtr->my;
				dir1 = ngbGridInfo1->mDir;								dir2 = ngbGridInfo2->mDir;
				// fprintf(fptrDSN, "\t%d %d %s\n", mx1, my1, toString((MAZDirectionType)dir1));
				// fprintf(fptrDSN, "\t%d %d %s\n", mx2, my2, toString((MAZDirectionType)dir2));
				fprintf(fptrDSN, "Open_Wall %d %d %s\n", mx1-nColOff, my1-nRowOff, toString((MAZDirectionType)dir1));
				fprintf(fptrDSN, "Open_Wall %d %d %s\n", mx2-nColOff, my2-nRowOff, toString((MAZDirectionType)dir2));
			}
		}
	}
	// fprintf(fptrDSN, "}\n\n");

	/*------------------------ door ------------------------
	fprintf(fptrDSN, "Unlocked_Door\n{\n");*/
	for (int i = 0; i < nRoom; i++)
	{
		std::list<MazeNgbRoomInfo*>::const_iterator itrNgbRoomInfo;
		for (itrNgbRoomInfo = mRoomInfo[i]->mNgbRoomInfoList.begin(); 
			itrNgbRoomInfo != mRoomInfo[i]->mNgbRoomInfoList.end(); itrNgbRoomInfo++)
		{
			MazeNgbRoomInfo* curNgbRoomInfo = (*itrNgbRoomInfo);
			int ngbID = curNgbRoomInfo->mNgbRoomPtr->mID;
			if (i < ngbID) 
			{
				MazeNgbGridInfo* ngbGridInfo = mRoomInfo[i]->getNgbRoomInfo(ngbID)->mOpenNgbGridInfo;
				int mx = ngbGridInfo->mNgbGridPtr->mx;
				int my = ngbGridInfo->mNgbGridPtr->my;
				int dir = ngbGridInfo->mDir;
				// fprintf(fptrDSN, "\t%d %d %s\n", mx, my, toString((MAZDirectionType)dir));
fprintf(fptrDSN, "Door %d %d %s Green\n", mx-nColOff, my-nRowOff, toString((MAZDirectionType)dir));
			}
		}
	}
	// fprintf(fptrDSN, "}\n\n");

	/*------------------------ floor ------------------------
	fprintf(fptrDSN, "Floor\n{\n");
	for (int i = 0; i < nGridCol; i++)
		for (int j = 0; j < nGridRow; j++)
			fprintf(fptrDSN, "\t%d %d %c\n", i, j, 'C');
	// fprintf(fptrDSN, "}\n\n");*/

	/*------------------------ ceiling ------------------------
	fprintf(fptrDSN, "Ceiling\n{\n");
	for (int i = 0; i < nGridCol; i++)
		for (int j = 0; j < nGridRow; j++)
			fprintf(fptrDSN, "\t%d %d %c\n", i, j, 'C');
	// fprintf(fptrDSN, "}\n\n");*/

	fclose(fptrDSN);
}


/** MazeGenerator: exportToMAZFile()
    export existing structure to .MAZ file
*/
void MazeGenerator::exportToMAZFile()
{
	FILE* fptrMAZ;
	char fileNameMAZ[128];
	sprintf(fileNameMAZ, "%s.MAZ", fileName);
	fptrMAZ=fopen(fileNameMAZ, "wb");

	fprintf(fptrMAZ, "Room Index Map: \n");
	for (int j = nGridRow - 1; j >= 0; j--)
	{
		for (int i = 0; i < nGridCol; i++)
		{
			fprintf(fptrMAZ, "%4d\t", mGridInfo[i][j]->roomID);
		}
		fprintf(fptrMAZ, "\n");
	}
	fprintf(fptrMAZ, "\n");
	
	/* Iterate through all rooms */
	fprintf(fptrMAZ, "\nRoom Neighbor Info: \n");
	for (int i = 0; i < nRoom; i++)	
	{
		fprintf(fptrMAZ, "\n Room No. %d: \n", i);
		
		/* Iterate through all MazeNgbRoomInfo of each room */
		std::list<MazeNgbRoomInfo*>::const_iterator itrNgbRoomInfo;
		for (itrNgbRoomInfo = mRoomInfo[i]->mNgbRoomInfoList.begin(); 
			itrNgbRoomInfo != mRoomInfo[i]->mNgbRoomInfoList.end(); itrNgbRoomInfo++)
		{
			MazeNgbRoomInfo* curNgbRoomInfo = (*itrNgbRoomInfo);
			int ngbID = curNgbRoomInfo->mNgbRoomPtr->mID;
			fprintf(fptrMAZ, "    Neighbor ID = %d: ", ngbID);

			/* Iterate through all MazeNgbGridInfo of each MazeNgbRoomInfo */
			std::list<MazeNgbGridInfo*>::const_iterator itrNgbGridInfo;
			for (itrNgbGridInfo = curNgbRoomInfo->mNgbGridInfoList.begin(); 
				itrNgbGridInfo != curNgbRoomInfo->mNgbGridInfoList.end(); itrNgbGridInfo++)
			{
				MazeNgbGridInfo* curNgbGridInfo = (*itrNgbGridInfo);
				int mx = curNgbGridInfo->mNgbGridPtr->mx;
				int my = curNgbGridInfo->mNgbGridPtr->my;
				int dir = curNgbGridInfo->mDir;
				fprintf(fptrMAZ, "%4d%4d%4s, ", mx, my, toString((MAZDirectionType)dir));
			}
			fprintf(fptrMAZ, "\n");
		}
		fprintf(fptrMAZ, "\n");
	}

	/* Open path positions */
	fprintf(fptrMAZ, "\nOpen path positions: \n\n");
	for (int i = 0; i < nRoom; i++)
	{
		std::list<MazeNgbRoomInfo*>::const_iterator itrNgbRoomInfo;
		for (itrNgbRoomInfo = mRoomInfo[i]->mNgbRoomInfoList.begin(); 
			itrNgbRoomInfo != mRoomInfo[i]->mNgbRoomInfoList.end(); itrNgbRoomInfo++)
		{
			MazeNgbRoomInfo* curNgbRoomInfo = (*itrNgbRoomInfo);
			int ngbID = curNgbRoomInfo->mNgbRoomPtr->mID;
			if (i < ngbID) 
			{
				int roomID1 = i, roomID2 = ngbID;
				int mx1, my1, dir1, mx2, my2, dir2;

				MazeNgbGridInfo* ngbGridInfo1 = mRoomInfo[roomID1]->getNgbRoomInfo(roomID2)->mOpenNgbGridInfo;
				MazeNgbGridInfo* ngbGridInfo2 = mRoomInfo[roomID2]->getNgbRoomInfo(roomID1)->mOpenNgbGridInfo;
				mx1 = ngbGridInfo1->mNgbGridPtr->mx;		mx2 = ngbGridInfo2->mNgbGridPtr->mx;
				my1 = ngbGridInfo1->mNgbGridPtr->my;		my2 = ngbGridInfo2->mNgbGridPtr->my;
				dir1 = ngbGridInfo1->mDir;							dir2 = ngbGridInfo2->mDir;

				fprintf(fptrMAZ, "mx=%3d \t my=%3d \t dir =%3s\t \n", mx1, my1, toString((MAZDirectionType)dir1));
				fprintf(fptrMAZ, "mx=%3d \t my=%3d \t dir =%3s\t \n", mx2, my2, toString((MAZDirectionType)dir2));
			}
		}
	}

	fclose(fptrMAZ);
}


/** MazeGenerator: toString()
    static function that turns 'MAZDirectionType' to related string value
*/
char* MazeGenerator::toString(MAZDirectionType dir)
{
	if (dir == MAZ_NORTH) return "N";
	else if (dir == MAZ_EAST) return "E";
	else if (dir == MAZ_SOUTH) return "S";
	else if (dir == MAZ_WEST) return "W";
	else if (dir == MAZ_CENTER) return "C";
	else return " ";
}

/** MazeGenerator: toOpposite()
*/
MAZDirectionType MazeGenerator::toOpposite(MAZDirectionType dir)
{
	if (dir == MAZ_NORTH) return MAZ_SOUTH;
	else if (dir == MAZ_EAST) return MAZ_WEST;
	else if (dir == MAZ_SOUTH) return MAZ_NORTH;
	else if (dir == MAZ_WEST) return MAZ_EAST;
	else if (dir == MAZ_CENTER) return MAZ_CENTER;
	else return MAZ_NO_DIRECTION;
}
