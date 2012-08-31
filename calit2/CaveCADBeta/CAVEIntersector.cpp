/***************************************************************
* File Name: CAVEIntersector.cpp
*
* Description: Implementation of base intersection class
*
* Written by ZHANG Lelin on Oct 20, 2010
*
***************************************************************/
#include "CAVEIntersector.h"


using namespace osg;
using namespace std;


float CAVEIntersector::gMaxScope(1.0e12);


// Constructor
CAVEIntersector::CAVEIntersector()
{
    mRootNode = NULL;
    mTargetNode = NULL;
}


// Destructor
CAVEIntersector::~CAVEIntersector()
{
}


/***************************************************************
* Function: test()
*  
*  Perform intersection test with eye ray from viewer's position
*  to pointer's position in world coordinates. Return 'false' if
*  nothing is intersected. Intersected node is set to 'mHitNode'
*
***************************************************************/
bool CAVEIntersector::test(const osg::Vec3 pointerOrg, const osg::Vec3 pointerPos)
{
    if (!mRootNode) 
        return false;

    /* generate eye ray */
    LineSegment* eyeRay = new osg::LineSegment();
    Vec3 eyeVect = pointerPos - pointerOrg;
    eyeVect.normalize();
    eyeRay->set(pointerOrg, pointerOrg + eyeVect * gMaxScope);

    osgUtil::IntersectVisitor findIntersectVisitor;
    osgUtil::IntersectVisitor::HitList hitList;

    /* preform intersection test */
    findIntersectVisitor.addLineSegment(eyeRay);
    mRootNode->accept(findIntersectVisitor);  
    hitList = findIntersectVisitor.getHitList(eyeRay);
  
    if (!hitList.empty()) 
    {
        mHit = hitList.front();

        /* Intersection test returns true value on the following two conditions:
           There is no target node, or hit node is the same as target node.
        */
        if (!mTargetNode) 
            return true;

        Node *hitNode = dynamic_cast <Node*> (mHit.getGeode());

        if (hitNode == mTargetNode) 
            return true;
    }
    return false;
}


/***************************************************************
* Function: getHitNode()
***************************************************************/
Node *CAVEIntersector::getHitNode()
{
    Node *hitNode = dynamic_cast <Node*> (mHit.getGeode());
    return hitNode;
}







