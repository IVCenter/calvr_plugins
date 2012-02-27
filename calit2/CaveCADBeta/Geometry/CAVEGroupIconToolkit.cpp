/***************************************************************
* File Name: CAVEGroupIconToolkit.cpp
*
* Description:  
*
* Written by ZHANG Lelin on Jan 26, 2011
*
***************************************************************/
#include "CAVEGroupIconToolkit.h"


using namespace std;
using namespace osg;

/* initial bounding radius of manipulator */
float CAVEGroupIconToolkit::gManipulatorBoundRadius(CAVEGeodeIcon::gSphereBoundRadius);

/* manipulator boundary hint */
CAVEGroupIconToolkit *CAVEGroupIconToolkit::gMoveInstancePtr(NULL);
CAVEGroupIconToolkit *CAVEGroupIconToolkit::gCloneInstancePtr(NULL);
CAVEGroupIconToolkit *CAVEGroupIconToolkit::gRotateInstancePtr(NULL);
CAVEGroupIconToolkit *CAVEGroupIconToolkit::gManipulateInstancePtr(NULL);


// Constructor
CAVEGroupIconToolkit::CAVEGroupIconToolkit(const CAVEGeodeIconToolkit::Type &typ): mType(typ)
{
    mCAVEGeodeIconVector.clear();
    mMatrixTransVector.clear();

    switch (typ)
    {
	case CAVEGeodeIconToolkit::MOVE: initGroupMove(); break;
	case CAVEGeodeIconToolkit::CLONE: initGroupClone(); break;
	case CAVEGeodeIconToolkit::ROTATE: initGroupRotate(); break;
	case CAVEGeodeIconToolkit::MANIPULATE: initGroupManipulate(); break;
	default: break;
    };
}


/***************************************************************
* Function: initGroupMove()
***************************************************************/
void CAVEGroupIconToolkit::initGroupMove()
{
    CAVEGeodeIconToolkitMove *iconMoveX = new CAVEGeodeIconToolkitMove(CAVEGeodeIconToolkitMove::LEFT_RIGHT);
    CAVEGeodeIconToolkitMove *iconMoveY = new CAVEGeodeIconToolkitMove(CAVEGeodeIconToolkitMove::FRONT_BACK);
    CAVEGeodeIconToolkitMove *iconMoveZ = new CAVEGeodeIconToolkitMove(CAVEGeodeIconToolkitMove::UP_DOWN);

    mCAVEGeodeIconVector.push_back(iconMoveX);
    mCAVEGeodeIconVector.push_back(iconMoveY);
    mCAVEGeodeIconVector.push_back(iconMoveZ);

    MatrixTransform *iconMoveTransX = new MatrixTransform;
    MatrixTransform *iconMoveTransY = new MatrixTransform;
    MatrixTransform *iconMoveTransZ = new MatrixTransform;

    iconMoveTransX->addChild(iconMoveX);	iconMoveX->setMatrixTrans(iconMoveTransX);
    iconMoveTransY->addChild(iconMoveY);	iconMoveY->setMatrixTrans(iconMoveTransY);
    iconMoveTransZ->addChild(iconMoveZ);	iconMoveZ->setMatrixTrans(iconMoveTransZ);

    addChild(iconMoveTransX);
    addChild(iconMoveTransY);
    addChild(iconMoveTransZ);

    gMoveInstancePtr = this;
}


/***************************************************************
* Function: initGroupClone()
***************************************************************/
void CAVEGroupIconToolkit::initGroupClone()
{
    CAVEGeodeIconToolkitClone *iconCloneX = new CAVEGeodeIconToolkitClone(CAVEGeodeIconToolkitMove::LEFT_RIGHT);
    CAVEGeodeIconToolkitClone *iconCloneY = new CAVEGeodeIconToolkitClone(CAVEGeodeIconToolkitMove::FRONT_BACK);
    CAVEGeodeIconToolkitClone *iconCloneZ = new CAVEGeodeIconToolkitClone(CAVEGeodeIconToolkitMove::UP_DOWN);

    mCAVEGeodeIconVector.push_back(iconCloneX);
    mCAVEGeodeIconVector.push_back(iconCloneY);
    mCAVEGeodeIconVector.push_back(iconCloneZ);

    MatrixTransform *iconCloneTransX = new MatrixTransform;
    MatrixTransform *iconCloneTransY = new MatrixTransform;
    MatrixTransform *iconCloneTransZ = new MatrixTransform;

    iconCloneTransX->addChild(iconCloneX);	iconCloneX->setMatrixTrans(iconCloneTransX);
    iconCloneTransY->addChild(iconCloneY);	iconCloneY->setMatrixTrans(iconCloneTransY);
    iconCloneTransZ->addChild(iconCloneZ);	iconCloneZ->setMatrixTrans(iconCloneTransZ);

    addChild(iconCloneTransX);
    addChild(iconCloneTransY);
    addChild(iconCloneTransZ);

    gCloneInstancePtr = this;
}


/***************************************************************
* Function: initGroupRotate()
***************************************************************/
void CAVEGroupIconToolkit::initGroupRotate()
{
    CAVEGeodeIconToolkitRotate *iconRotateX = new CAVEGeodeIconToolkitRotate(CAVEGeodeIconToolkitRotate::X_AXIS);
    CAVEGeodeIconToolkitRotate *iconRotateY = new CAVEGeodeIconToolkitRotate(CAVEGeodeIconToolkitRotate::Y_AXIS);
    CAVEGeodeIconToolkitRotate *iconRotateZ = new CAVEGeodeIconToolkitRotate(CAVEGeodeIconToolkitRotate::Z_AXIS);

    mCAVEGeodeIconVector.push_back(iconRotateX);
    mCAVEGeodeIconVector.push_back(iconRotateY);
    mCAVEGeodeIconVector.push_back(iconRotateZ);

    MatrixTransform *iconRotateTransX = new MatrixTransform;
    MatrixTransform *iconRotateTransY = new MatrixTransform;
    MatrixTransform *iconRotateTransZ = new MatrixTransform;

    iconRotateTransX->addChild(iconRotateX);	iconRotateX->setMatrixTrans(iconRotateTransX);
    iconRotateTransY->addChild(iconRotateY);	iconRotateY->setMatrixTrans(iconRotateTransY);
    iconRotateTransZ->addChild(iconRotateZ);	iconRotateZ->setMatrixTrans(iconRotateTransZ);

    addChild(iconRotateTransX);
    addChild(iconRotateTransY);
    addChild(iconRotateTransZ);

    gRotateInstancePtr = this;
}


/***************************************************************
* Function: initGroupManipulate()
***************************************************************/
void CAVEGroupIconToolkit::initGroupManipulate()
{
    /* manipulation points are located on 8 corners, 12 edges and 6 surfaces of the wireframe */
    CAVEGeodeIconToolkitManipulate **iconManipulateArray = new CAVEGeodeIconToolkitManipulate*[26];

    const CAVEGeodeIconToolkitManipulate::CtrlPtType typCorner = CAVEGeodeIconToolkitManipulate::CORNER;
    const CAVEGeodeIconToolkitManipulate::CtrlPtType typEdge = CAVEGeodeIconToolkitManipulate::EDGE;
    const CAVEGeodeIconToolkitManipulate::CtrlPtType typFace = CAVEGeodeIconToolkitManipulate::FACE;

    iconManipulateArray[0] = new CAVEGeodeIconToolkitManipulate(Vec3(-1, -1, -1), typCorner);
    iconManipulateArray[1] = new CAVEGeodeIconToolkitManipulate(Vec3( 1, -1, -1), typCorner);
    iconManipulateArray[2] = new CAVEGeodeIconToolkitManipulate(Vec3(-1,  1, -1), typCorner);
    iconManipulateArray[3] = new CAVEGeodeIconToolkitManipulate(Vec3( 1,  1, -1), typCorner);
    iconManipulateArray[4] = new CAVEGeodeIconToolkitManipulate(Vec3(-1, -1,  1), typCorner);
    iconManipulateArray[5] = new CAVEGeodeIconToolkitManipulate(Vec3( 1, -1,  1), typCorner);
    iconManipulateArray[6] = new CAVEGeodeIconToolkitManipulate(Vec3(-1,  1,  1), typCorner);
    iconManipulateArray[7] = new CAVEGeodeIconToolkitManipulate(Vec3( 1,  1,  1), typCorner);

    iconManipulateArray[ 8] = new CAVEGeodeIconToolkitManipulate(Vec3(-1, -1, 0), typEdge);
    iconManipulateArray[ 9] = new CAVEGeodeIconToolkitManipulate(Vec3(-1,  1, 0), typEdge);
    iconManipulateArray[10] = new CAVEGeodeIconToolkitManipulate(Vec3( 1, -1, 0), typEdge);
    iconManipulateArray[11] = new CAVEGeodeIconToolkitManipulate(Vec3( 1,  1, 0), typEdge);
    iconManipulateArray[12] = new CAVEGeodeIconToolkitManipulate(Vec3(-1, 0, -1), typEdge);
    iconManipulateArray[13] = new CAVEGeodeIconToolkitManipulate(Vec3(-1, 0,  1), typEdge);
    iconManipulateArray[14] = new CAVEGeodeIconToolkitManipulate(Vec3( 1, 0, -1), typEdge);
    iconManipulateArray[15] = new CAVEGeodeIconToolkitManipulate(Vec3( 1, 0,  1), typEdge);
    iconManipulateArray[16] = new CAVEGeodeIconToolkitManipulate(Vec3(0, -1, -1), typEdge);
    iconManipulateArray[17] = new CAVEGeodeIconToolkitManipulate(Vec3(0, -1,  1), typEdge);
    iconManipulateArray[18] = new CAVEGeodeIconToolkitManipulate(Vec3(0,  1, -1), typEdge);
    iconManipulateArray[19] = new CAVEGeodeIconToolkitManipulate(Vec3(0,  1,  1), typEdge);

    iconManipulateArray[20] = new CAVEGeodeIconToolkitManipulate(Vec3( 1, 0, 0), typFace);
    iconManipulateArray[21] = new CAVEGeodeIconToolkitManipulate(Vec3(-1, 0, 0), typFace);
    iconManipulateArray[22] = new CAVEGeodeIconToolkitManipulate(Vec3(0,  1, 0), typFace);
    iconManipulateArray[23] = new CAVEGeodeIconToolkitManipulate(Vec3(0, -1, 0), typFace);
    iconManipulateArray[24] = new CAVEGeodeIconToolkitManipulate(Vec3(0, 0,  1), typFace);
    iconManipulateArray[25] = new CAVEGeodeIconToolkitManipulate(Vec3(0, 0, -1), typFace);

    /* add icon geodes to upper level of matrix transforms */
    MatrixTransform **iconMatrixTransArray = new MatrixTransform*[26];
    for (int i = 0; i < 26; i++)
    {
	iconMatrixTransArray[i] = new MatrixTransform;
	addChild(iconMatrixTransArray[i]);
	iconMatrixTransArray[i]->addChild(iconManipulateArray[i]);

	mCAVEGeodeIconVector.push_back(iconManipulateArray[i]);
	mMatrixTransVector.push_back(iconMatrixTransArray[i]);
    }

    gManipulateInstancePtr = this;
}


/***************************************************************
* Function: setManipulatorBoundRadius()
*
* 'gManipulatorBoundRadius' is used in 'setManipulatorBound()'
*  as reference scale to adjust manipulator sizes
*
***************************************************************/
void CAVEGroupIconToolkit::setManipulatorBoundRadius(const BoundingBox& bb)
{
    gManipulatorBoundRadius = bb.radius();
}

/***************************************************************
* Function: setManipulatorBound()
*
* 'manipulatorBound': bounding sizes of manipulator in Design 
*  State space, updated each time when new geode is selected or
*  snapping action is finished.
*
***************************************************************/
void CAVEGroupIconToolkit::setManipulatorBound(const BoundingBox& bb)
{
    if (gManipulatorBoundRadius <= 0) return;

    Vec3 manipulatorBound;
    manipulatorBound.x() = (CAVEGeodeIcon::gSphereBoundRadius) / gManipulatorBoundRadius * (bb.xMax() - bb.xMin()) * 0.5;
    manipulatorBound.y() = (CAVEGeodeIcon::gSphereBoundRadius) / gManipulatorBoundRadius * (bb.yMax() - bb.yMin()) * 0.5;
    manipulatorBound.z() = (CAVEGeodeIcon::gSphereBoundRadius) / gManipulatorBoundRadius * (bb.zMax() - bb.zMin()) * 0.5;

    /* update bounding sizes and reset matrix transforms under 'gManipulateInstancePtr' */
    if (!gManipulateInstancePtr) return;
    for (int i = 0; i < 26; i++)
    {
	CAVEGeodeIconToolkitManipulate *iconPtr = dynamic_cast <CAVEGeodeIconToolkitManipulate*> 
		(gManipulateInstancePtr->mCAVEGeodeIconVector[i]);
	if (iconPtr)
	{
	    iconPtr->updateBoundingVect(manipulatorBound);
	    iconPtr->setMatrixTrans(gManipulateInstancePtr->mMatrixTransVector[i]);
	}
    }
}










