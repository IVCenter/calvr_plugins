/***************************************************************
* File Name: DOGeometryCloner.cpp
*
* Description:
*
* Written by ZHANG Lelin on May 4, 2011
*
***************************************************************/
#include "DOGeometryCloner.h"


using namespace std;
using namespace osg;


// Constructor
DOGeometryCloner::DOGeometryCloner(): mEnabledFlag(false)
{
}


/***************************************************************
* Function: setDOGeometryCollectorPtr()
***************************************************************/
void DOGeometryCloner::setDOGeometryCollectorPtr(DOGeometryCollector *geomCollectorPtr)
{
    mDOGeometryCollectorPtr = geomCollectorPtr;
}


/***************************************************************
* Function: pushClonedObjects()
*
* Called at the first time that 'CLONE' icon is clicked, copy
* all selected geometry from 'mDOGeometryCollectorPtr' and push
* them into the 'mDupGeodeShapeVector' and 'mOrgGeometryVector'
* temporarily.
*
***************************************************************/
void DOGeometryCloner::pushClonedObjects()
{
    if (!mEnabledFlag) return;

    /* If no specific surface is selected, then make copy of all geodes in 'geodeShapeVector', 
       and register them in global scene graph. */
    if (mDOGeometryCollectorPtr->mMode == DOGeometryCollector::COLLECT_GEODE)
    {
	CAVEGeodeShapeVector& geodeShapeVector = mDOGeometryCollectorPtr->getGeodeShapeVector();
	const int nGeodes = geodeShapeVector.size();
 
	if (nGeodes > 0)
	{
	    for (int i = 0; i < nGeodes; i++)
	    {
		CAVEGeodeShape *clonedGeode = new CAVEGeodeShape(geodeShapeVector[i]);
		mDupGeodeShapeVector.push_back(clonedGeode);

		CAVEGroupShape *clonedGroup = new CAVEGroupShape(clonedGeode);
		mDOShapeSwitch->addChild(clonedGroup);
	    }   
	}
    }
}


/***************************************************************
* Function: popClonedObjects()
*
* Add current geometries stored in 'mDupGeodeShapeVector' and
* 'mOrgGeometryVector' to 'mDOGeometryCollectorPtr' selections.
* Deselected the existing ones.
*
***************************************************************/
void DOGeometryCloner::popClonedObjects()
{
    if (mDupGeodeShapeVector.size() > 0)
    {
	CAVEGeodeShapeVector& geodeShapeVector = mDOGeometryCollectorPtr->getGeodeShapeVector();
	const int nGeodes = geodeShapeVector.size();

	if (mDupGeodeShapeVector.size() == nGeodes)
	{
	    /* de-select all existing geodes in 'mDOGeometryCollectorPtr' */
	    for (int i = 0; i < nGeodes; i++)
	    {
		CAVEGeodeShape *orgGeode = geodeShapeVector[i];
		mDOGeometryCollectorPtr->toggleCAVEGeodeShape(orgGeode);
	    }

	    /* select all cloned geodes from 'mDupGeodeShapeVector' */
	    for (int i = 0; i < nGeodes; i++)
	    {
		CAVEGeodeShape *dupGeode = mDupGeodeShapeVector[i];
		mDOGeometryCollectorPtr->toggleCAVEGeodeShape(dupGeode);
	    }
	}
	// else cerr << "Warning DOGeometryCloner::popClonedObjects(): 'Un-matched' vector sizes." << endl;

	mDupGeodeShapeVector.clear();
    }
}











