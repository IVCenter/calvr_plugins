/***************************************************************
* File Name: PathRecordEntryList.cpp
*
* Description:
*
* Written by ZHANG Lelin on Oct 15, 2010
*
***************************************************************/
#include "PathRecordEntryList.h"

#include <iostream>
#include <cstdio>

using namespace std;
using namespace osg;


// Constructor
PathRecordEntryList::PathRecordEntryList(): mItrIdx(0), mNumEntries(0), mPeriod(0.f), mRecordEntryArray(NULL)
{
    mDataBuffer = new double[PATH_RECORD_ENTRY_LIST_SIZE_BUFFER];
}


// Destructor
PathRecordEntryList::~PathRecordEntryList()
{
    delete []mDataBuffer;
}


/***************************************************************
*  Function: loadFile()
***************************************************************/
void PathRecordEntryList::loadFile(FILE* filePtr)
{
    int counter = 0;
    double val;
    while (1)
    {
	//if (fscanf(filePtr, " %f\n", &val) < 0) break;
	if(fread(&val, sizeof(double),1,filePtr) <= 0) break;
	mDataBuffer[counter] = val;
	if (++counter >= PATH_RECORD_ENTRY_LIST_SIZE_BUFFER)
	{
	    cerr << "WARNING: PathRecordEntryList: Buffer overflow in reading file." << endl;
	    break;
	}
    }

    /* clear and new data to 'mRecordEntryArray' */
    mNumEntries = counter / 18;
    mRecordEntryArray = new PathRecordEntry[mNumEntries];
    for (int i = 0; i < mNumEntries; i++) 
    {
	int off = i * 18;
	mRecordEntryArray[i].t = mDataBuffer[off];
	mRecordEntryArray[i].s = mDataBuffer[off + 1];
	/*mRecordEntryArray[i].xmat = Matrixd
		(mDataBuffer[off + 2],  mDataBuffer[off + 3],  mDataBuffer[off + 4],  mDataBuffer[off + 5],
		 mDataBuffer[off + 6],  mDataBuffer[off + 7],  mDataBuffer[off + 8],  mDataBuffer[off + 9],
		 mDataBuffer[off + 10], mDataBuffer[off + 11], mDataBuffer[off + 12], mDataBuffer[off + 13],
		 mDataBuffer[off + 14], mDataBuffer[off + 15], mDataBuffer[off + 16], mDataBuffer[off + 17]);*/

        mRecordEntryArray[i].xmat = Matrixd((double*)&mDataBuffer[off + 2]);
    }

    if (mNumEntries <= 0) 
    {
	mPeriod = 0;
	return;
    }
    mPeriod = mRecordEntryArray[mNumEntries - 1].t;
}


/***************************************************************
*  Function: lookupRecordEntry()
***************************************************************/
bool PathRecordEntryList::lookupRecordEntry(const double &t, double &scale, Matrixd &xMat)
{
    if (mNumEntries <= 1 || mPeriod == 0) return false;

    //double clamptime = t - (int)(t / mPeriod) * mPeriod;
    double clamptime = t;
    if(clamptime < 0)
    {
	clamptime = 0;
    }
    else if(clamptime > mPeriod)
    {
	clamptime = mRecordEntryArray[mNumEntries-1].t;
    }
    
    mItrIdx = 0;
    while (clamptime > mRecordEntryArray[mItrIdx].t)
    {
	if (++mItrIdx >= mNumEntries) mItrIdx = 0;
    }
    interpolateRecordEntry(clamptime, scale, xMat);
    return true;
}


/***************************************************************
*  Function: interpolateRecordEntry()
***************************************************************/
void PathRecordEntryList::interpolateRecordEntry(const double &clamptime, double &scale, Matrixd &xmat)
{
    //std::cerr << "Size mat: " << sizeof(Matrixd::value_type) << " vec: " << sizeof(Vec3d::value_type) << std::endl;
    double t1, t2, s1, s2, w1, w2;
    Matrixd xmat1, xmat2;

    //xmat = mRecordEntryArray[mItrIdx].xmat;
    //scale = mRecordEntryArray[mItrIdx].s;
    //return;
    /* load adjacent record entries */
    if (mItrIdx == 0) 
    {
	t1 = mRecordEntryArray[0].t;
	s1 = mRecordEntryArray[0].s;
	xmat1 = mRecordEntryArray[0].xmat;
    }
    else 
    {
	t1 = mRecordEntryArray[mItrIdx - 1].t;
	s1 = mRecordEntryArray[mItrIdx - 1].s;
	xmat1 = mRecordEntryArray[mItrIdx - 1].xmat;
    }
    t2 = mRecordEntryArray[mItrIdx].t;
    s2 = mRecordEntryArray[mItrIdx].s;
    xmat2 = mRecordEntryArray[mItrIdx].xmat;

    //w1 = (t2 - clamptime) / (t2 - t1);
    w2 = (clamptime - t1) / (t2 - t1);
    w1 = 1.0 - w2;

    for(int i = 0; i < 16; i++)
    {
	xmat.ptr()[i] = w1 * xmat1.ptr()[i] + w2 * xmat2.ptr()[i];
    }

    /* perform linear interpolation scale and matrix */
    /*Vec3d x1, y1, z1, p1, x2, y2, z2, p2, ix, iy, iz, ip;

    x1 = Vec3d(xmat1(0, 0), xmat1(1, 0), xmat1(2, 0));
    y1 = Vec3d(xmat1(0, 1), xmat1(1, 1), xmat1(2, 1));
    z1 = Vec3d(xmat1(0, 2), xmat1(1, 2), xmat1(2, 2));
    p1 = Vec3d(xmat1(3, 0), xmat1(3, 1), xmat1(3, 2));

    x2 = Vec3d(xmat2(0, 0), xmat2(1, 0), xmat2(2, 0));
    y2 = Vec3d(xmat2(0, 1), xmat2(1, 1), xmat2(2, 1));
    z2 = Vec3d(xmat2(0, 2), xmat2(1, 2), xmat2(2, 2));
    p2 = Vec3d(xmat2(3, 0), xmat2(3, 1), xmat2(3, 2));

    ix = x1 * w1 + x2 * w2;	ix.normalize();
    iy = y1 * w1 + y2 * w2;	iy.normalize();
    iz = z1 * w1 + z2 * w2;	iz.normalize();
    ip = p1 * w1 + p2 * w2;*/

    scale = s1 * w1 + s2 * w2;
    /*xmat = Matrixd( ix.x(), iy.x(), iz.x(), 0, 
		    ix.y(), iy.y(), iz.y(), 0, 
		    ix.z(), iy.z(), iz.z(), 0, 
		    ip.x(), ip.y(), ip.z(), 1);*/
}

































