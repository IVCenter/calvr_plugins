/***************************************************************
* File Name: CAVEGeodeIconToolkit.cpp
*
* Description:  
*
* Written by ZHANG Lelin on Jan 19, 2011
*
***************************************************************/
#include "CAVEGeodeIconToolkit.h"


using namespace std;
using namespace osg;


/* default colors and transparencies: values might be over written in specific constructors */
const float CAVEGeodeIconToolkit::gAlphaNormal(0.5f);
const float CAVEGeodeIconToolkit::gAlphaSelected(1.0f);
const float CAVEGeodeIconToolkit::gAlphaUnselected(0.2f);
const Vec3 CAVEGeodeIconToolkit::gDiffuseColor(Vec3(0.0f, 1.0f, 0.0f));
const Vec3 CAVEGeodeIconToolkit::gSpecularColor(Vec3(0.0f, 1.0f, 0.0f));


/* osg object sizes of 'CAVEGeodeIconToolkitMove' */
const float CAVEGeodeIconToolkitMove::gBodyRadius(0.03f);
const float CAVEGeodeIconToolkitMove::gBodyLength(0.18f);
const float CAVEGeodeIconToolkitMove::gArrowRadius(0.06f);
const float CAVEGeodeIconToolkitMove::gArrowLength(0.15f);


/* osg object sizes of 'CAVEGeodeIconToolkitRotate' */
const float CAVEGeodeIconToolkitRotate::gPanelRadius(0.3f);
const float CAVEGeodeIconToolkitRotate::gPanelThickness(0.002f);


/* osg object sizes of 'CAVEGeodeIconToolkitManipulate' */
const float CAVEGeodeIconToolkitManipulate::gCtrlPtSize(0.04f);


//Constructor
CAVEGeodeIconToolkit::CAVEGeodeIconToolkit()
{
    applyColor(gDiffuseColor, gSpecularColor, gAlphaNormal);
    applyTexture("");
}

//Destructor
CAVEGeodeIconToolkit::~CAVEGeodeIconToolkit()
{
}


/***************************************************************
* Function: pressed()
***************************************************************/
void CAVEGeodeIconToolkit::pressed()
{
    applyAlpha(gAlphaSelected);
}


/***************************************************************
* Function: released()
***************************************************************/
void CAVEGeodeIconToolkit::released()
{
    applyAlpha(gAlphaNormal);
}


/***************************************************************
* Class: CAVEGeodeIconToolkitMove
***************************************************************/
CAVEGeodeIconToolkitMove::CAVEGeodeIconToolkitMove(const FaceOrientation &typ)
{
    mType = MOVE;
    mFaceOrient = typ;

    initBaseGeometry();
}


void CAVEGeodeIconToolkitMove::initBaseGeometry()
{
    const float offz= (CAVEGeodeIcon::gSphereBoundRadius) * 0.6;

    mConeUp = new Cone();
    mConeDrawableUp = new ShapeDrawable(mConeUp);
    mConeUp->setRadius(gArrowRadius);
    mConeUp->setHeight(gArrowLength);
    mConeUp->setCenter(Vec3(0, 0, offz + gBodyLength + gArrowLength * 0.25));
    addDrawable(mConeDrawableUp);

    mCylinderUp = new Cylinder();
    mCylinderDrawableUp = new ShapeDrawable(mCylinderUp);
    mCylinderUp->setRadius(gBodyRadius);
    mCylinderUp->setHeight(gBodyLength);
    mCylinderUp->setCenter(Vec3(0, 0, offz + gBodyLength * 0.5));
    addDrawable(mCylinderDrawableUp);

    mConeDown = new Cone();
    mConeDrawableDown = new ShapeDrawable(mConeDown);
    mConeDown->setRadius(gArrowRadius);
    mConeDown->setHeight(gArrowLength);
    mConeDown->setRotation(Quat(M_PI, Vec3(1, 0, 0)));
    mConeDown->setCenter(Vec3(0, 0, -(offz + gBodyLength + gArrowLength * 0.25)));
    addDrawable(mConeDrawableDown);

    mCylinderDown = new Cylinder();
    mCylinderDrawableDown = new ShapeDrawable(mCylinderDown);
    mCylinderDown->setRadius(gBodyRadius);
    mCylinderDown->setHeight(gBodyLength);
    mCylinderDown->setCenter(Vec3(0, 0, -(offz + gBodyLength * 0.5)));
    addDrawable(mCylinderDrawableDown);

    /* apply discrimitive colors to different types of icons */
    if (mFaceOrient == FRONT_BACK) 
    {
        applyColor(Vec3(1, 0, 0), Vec3(1, 0, 0), gAlphaNormal);
        applyTexture("");
    }
    else if (mFaceOrient == LEFT_RIGHT) 
    {
        applyColor(Vec3(0, 1, 0), Vec3(0, 1, 0), gAlphaNormal);
        applyTexture("");
    }
    else if (mFaceOrient == UP_DOWN) 
    {
        applyColor(Vec3(0, 0, 1), Vec3(0, 0, 1), gAlphaNormal);
        applyTexture("");
    }
}


/***************************************************************
* Class: CAVEGeodeIconToolkitClone
***************************************************************/
CAVEGeodeIconToolkitClone::CAVEGeodeIconToolkitClone(const FaceOrientation &typ)
{
    /* change type back to 'CLONE' and add double arrows */
    mType = CLONE;
    mFaceOrient = typ;

    initBaseGeometry();

    const float offz= (CAVEGeodeIcon::gSphereBoundRadius) * 0.75;	// use differenc offset from 'MOVE'

    mExConeUp = new Cone();
    mExConeDrawableUp = new ShapeDrawable(mExConeUp);
    mExConeUp->setRadius(gArrowRadius);
    mExConeUp->setHeight(gArrowLength);
    mExConeUp->setCenter(Vec3(0, 0, offz + gBodyLength + gArrowLength * 0.25));
    addDrawable(mExConeDrawableUp);

    mExConeDown = new Cone();
    mExConeDrawableDown = new ShapeDrawable(mExConeDown);
    mExConeDown->setRadius(gArrowRadius);
    mExConeDown->setHeight(gArrowLength);
    mExConeDown->setRotation(Quat(M_PI, Vec3(1, 0, 0)));
    mExConeDown->setCenter(Vec3(0, 0, -(offz + gBodyLength + gArrowLength * 0.25)));
    addDrawable(mExConeDrawableDown);
}


void CAVEGeodeIconToolkitMove::setMatrixTrans(MatrixTransform *matTrans)
{
    Matrixd rotMat;

    if (mFaceOrient == FRONT_BACK)
    {
	rotMat = Matrixd(Quat(M_PI * 0.5, Vec3(1, 0, 0)));
    }
    else if (mFaceOrient == LEFT_RIGHT)
    {
	rotMat = Matrixd(Quat(M_PI * 0.5, Vec3(0, 1, 0)));
    }
    else rotMat.makeIdentity();

    matTrans->setMatrix(rotMat);
}


/***************************************************************
* Class: CAVEGeodeIconToolkitRotate
***************************************************************/
CAVEGeodeIconToolkitRotate::CAVEGeodeIconToolkitRotate(const AxisAlignment &typ)
{
    mType = ROTATE;
    mAxisAlignment = typ;

    mPanelCylinder = new Cylinder();
    mPanelDrawable = new ShapeDrawable(mPanelCylinder);
    mPanelCylinder->setRadius(gPanelRadius);
    mPanelCylinder->setHeight(gPanelThickness);
    mPanelCylinder->setCenter(Vec3(0, 0, 0));
    addDrawable(mPanelDrawable);

    /* use texture map for round rotation panels */
    applyColor(Vec3(0.8, 0.8, 0.8), Vec3(0.8, 0.8, 0.8), gAlphaNormal);
    applyTexture(getDataDir() + "Textures/RoundRuler.JPG");
}


void CAVEGeodeIconToolkitRotate::setMatrixTrans(osg::MatrixTransform *matTrans)
{
    Matrixd rotMat;

    if (mAxisAlignment == X_AXIS)
    {
	rotMat = Matrixd(Quat(M_PI * 0.5, Vec3(0, 1, 0)));
    }
    else if (mAxisAlignment == Y_AXIS)
    {
	rotMat = Matrixd(Quat(M_PI * 0.5, Vec3(1, 0, 0)));
    }
    else rotMat.makeIdentity();

    matTrans->setMatrix(rotMat);
}


/***************************************************************
* Class: CAVEGeodeIconToolkitManipulate
***************************************************************/
CAVEGeodeIconToolkitManipulate::CAVEGeodeIconToolkitManipulate(const Vec3 &scalingDir, const CtrlPtType &ctrlPtTyp)
{
    mType = MANIPULATE;
    mCtrlPtType = ctrlPtTyp;

    mScalingDir = scalingDir;
    mBoundingVect = Vec3(0, 0, 0);	// 'mBoundingVect' is not set until an actual geode is selected

    /* create osg geometry objects */
    mCtrlPtBox = new Box(Vec3(0, 0, 0), gCtrlPtSize);
    mCtrlPtBoxDrawable = new ShapeDrawable(mCtrlPtBox);
    addDrawable(mCtrlPtBoxDrawable);

    /* set color texture to discriminate between types of control points */
    if (ctrlPtTyp == CORNER) 
    {
        applyColor(Vec3(1, 0, 0), Vec3(1, 0, 0), gAlphaNormal);
        applyTexture("");
    }
    else if (ctrlPtTyp == EDGE) 
    {
        applyColor(Vec3(0, 1, 0), Vec3(0, 1, 0), gAlphaNormal);
        applyTexture("");
    }
    else if (ctrlPtTyp == FACE) 
    {
        applyColor(Vec3(0, 0, 1), Vec3(0, 0, 1), gAlphaNormal);
        applyTexture("");
    }
}


void CAVEGeodeIconToolkitManipulate::setMatrixTrans(MatrixTransform *matTrans)
{
    Matrixd offsetMat;
    offsetMat.makeTranslate(Vec3(mScalingDir.x() * mBoundingVect.x(), 
				 mScalingDir.y() * mBoundingVect.y(), 
				 mScalingDir.z() * mBoundingVect.z()));
    matTrans->setMatrix(offsetMat);
}

