/***************************************************************
* Animation File Name: ANIMRefXYPlane.cpp
*
* Description: Create reference ground
*
* Written by ZHANG Lelin on Sep 29, 2010
*
***************************************************************/
#include "ANIMRefXYPlane.h"

using namespace std;
using namespace osg;

namespace CAVEAnimationModeler
{

/***************************************************************
* Function: ANIMCreateRefXYPlane()
*
***************************************************************/
MatrixTransform *ANIMCreateRefXYPlane()
{
    MatrixTransform *XYPlaneTrans = new MatrixTransform;
    Node* XYPlaneNode = osgDB::readNodeFile(ANIMDataDir() + "VRMLFiles/XYPlane.WRL");
    XYPlaneTrans->addChild(XYPlaneNode);
    return XYPlaneTrans;
}

};
