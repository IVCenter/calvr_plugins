/***************************************************************
* File Name: CAVEIntersector.h
*
* Class Name: CAVEIntersector
*
* Description: Prototype of base intersector class that inherited
*  by state intersector and object intersector.
*
***************************************************************/
#ifndef _CAVE_INTERSECTOR_H_
#define _CAVE_INTERSECTOR_H_


// C++
#include <stdlib.h>
#include <math.h>
#include <list>
#include <iostream>

// Open scene graph
#include <osg/Drawable>
#include <osg/Geometry>
#include <osg/Group>
#include <osg/Vec3>
#include <osgUtil/IntersectVisitor>


class CAVEIntersector
{
  public:
    CAVEIntersector();
    virtual ~CAVEIntersector();

    void loadRootTargetNode(osg::Node *rNode, osg::Node *tNode) 
    { 
        mRootNode = rNode;  
        mTargetNode = tNode; 
    }
    bool test(const osg::Vec3 pointerOrg, const osg::Vec3 pointerPos);
    osg::Vec3 getWorldHitPosition() { return mHit.getWorldIntersectPoint(); }
    osg::Vec3 getWorldHitNormal() { return mHit.getWorldIntersectNormal(); }
    osg::Node *getHitNode();

  protected:
    osg::Node *mRootNode, *mTargetNode;
    osgUtil::Hit mHit;

    static float gMaxScope;
};


#endif
