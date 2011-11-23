/***************************************************************
* File Name: CAVEGroupShape.cpp
*
* Description:  
*
* Written by ZHANG Lelin on Jan 26, 2011
*
***************************************************************/
#include "CAVEGroupShape.h"


using namespace std;
using namespace osg;


// Default Constructor
CAVEGroupShape::CAVEGroupShape()
{
    mCAVEGeodeShapeVector.clear();
}

// Constructore using single CAVEGeodeShape object
CAVEGroupShape::CAVEGroupShape(CAVEGeodeShape *shape)
{
    addCAVEGeodeShape(shape);
}


/***************************************************************
* Function: addCAVEGeodeShape()
***************************************************************/
void CAVEGroupShape::addCAVEGeodeShape(CAVEGeodeShape *shape)
{
    mCAVEGeodeShapeVector.push_back(shape);
    addChild(shape);
}


/***************************************************************
* Function: getCAVEGeodeShape()
***************************************************************/
CAVEGeodeShape *CAVEGroupShape::getCAVEGeodeShape(const int &idx)
{
    if (idx < mCAVEGeodeShapeVector.size()) return mCAVEGeodeShapeVector[idx];
    else return NULL;
}


