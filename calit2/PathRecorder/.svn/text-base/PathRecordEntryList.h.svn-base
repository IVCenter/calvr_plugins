/***************************************************************
* File Name: PathRecordEntryList.h
*
* Class Name: PathRecordEntryList
*
***************************************************************/
#ifndef _PATH_RECORD_ENTRY_LIST_H_
#define _PATH_RECORD_ENTRY_LIST_H_


// C++
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Open scene graph
#include <osg/Matrixd>
#include <osg/Vec3>


using namespace std;
using namespace osg;


#define PATH_RECORD_ENTRY_LIST_SIZE_BUFFER	512001


/***************************************************************
* Class Name: PathRecordEntryList
***************************************************************/
class PathRecordEntryList
{
  public:
    PathRecordEntryList();
    ~PathRecordEntryList();

    struct PathRecordEntry
    {
	PathRecordEntry(): t(0.f), s(0.f), xmat() {}

	double t;		// time stamp
	double s;		// scale value
	Matrixd	xmat;		// xform matrix
    };

    const double &getPeriodical() { return mPeriod; }
    void loadFile(FILE* filePtr);
    bool lookupRecordEntry(const double &timer, double &scale, Matrixd &xMat);
    void interpolateRecordEntry(const double &clamptime, double &scale, Matrixd &xmat);

  protected:
    int mItrIdx;
    int mNumEntries;
    double mPeriod;
    double *mDataBuffer;
    PathRecordEntry *mRecordEntryArray;
};

#endif
