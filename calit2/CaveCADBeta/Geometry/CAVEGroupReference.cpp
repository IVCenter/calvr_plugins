/***************************************************************
* File Name: CAVEGroupReference.cpp
*
* Description:  
*
* Written by ZHANG Lelin on Dec 1, 2010
*
***************************************************************/
#include "CAVEGroupReference.h"


using namespace std;
using namespace osg;


osg::Vec3 CAVEGroupReference::gPointerDir(Vec3(0, 1, 0));
const float CAVEGroupReferenceAxis::gCharSize(0.02f);
const float CAVEGroupReferenceAxis::gCharDepth(0.003f);


// Constructor
CAVEGroupReference::CAVEGroupReference(): mCenter(osg::Vec3(0, 0, 0))
{
    mMatrixTrans = new MatrixTransform;
    addChild(mMatrixTrans);
}


/***************************************************************
* Function: setCenterPos()
***************************************************************/
void CAVEGroupReference::setCenterPos(const Vec3 &center)
{
    mCenter = center;
    Matrixf transMat;
    transMat.makeTranslate(mCenter);
    mMatrixTrans->setMatrix(transMat);
}


/***************************************************************
* Function: setPointerDir()
***************************************************************/
void CAVEGroupReference::setPointerDir(const osg::Vec3 &pointerDir)
{
    /* update pointer front direction by projecting it onto XY plane */
    Vec3 pointerFront = pointerDir;
    pointerFront.z() = 0;
    pointerFront.normalize();

    gPointerDir = pointerFront;
}


// Constructor
CAVEGroupReferenceAxis::CAVEGroupReferenceAxis(): mVisibleFlag(false), mTextEnabledFlag(true)
{
    /* create axis associated geometry */
    mAxisSwitch = new Switch;

    mXDirTrans = new MatrixTransform;
    mYDirTrans = new MatrixTransform;
    mZDirTrans = new MatrixTransform;

    mXAxisGeode = new CAVEGeodeReferenceAxis();
    mYAxisGeode = new CAVEGeodeReferenceAxis();
    mZAxisGeode = new CAVEGeodeReferenceAxis();

    mXAxisGeode->setType(CAVEGeodeReferenceAxis::POS_X, &mXDirTrans);
    mYAxisGeode->setType(CAVEGeodeReferenceAxis::POS_Y, &mYDirTrans);
    mZAxisGeode->setType(CAVEGeodeReferenceAxis::POS_Z, &mZDirTrans);

    mMatrixTrans->addChild(mAxisSwitch);

    mAxisSwitch->addChild(mXDirTrans);
    mAxisSwitch->addChild(mYDirTrans);
    mAxisSwitch->addChild(mZDirTrans);

    mXDirTrans->addChild(mXAxisGeode);
    mYDirTrans->addChild(mYAxisGeode);
    mZDirTrans->addChild(mZAxisGeode);

    mAxisSwitch->setAllChildrenOff();

    /* create 3D texts geometry */
    if (mTextEnabledFlag)
    {
        mGridInfoTextTrans = new MatrixTransform;
        mXAxisTextTrans = new MatrixTransform;
        mYAxisTextTrans = new MatrixTransform;
        mZAxisTextTrans = new MatrixTransform;

        Geode *gridInfoTextGeode = createText3D(&mGridInfoText);
        Geode *xAxisTextGeode = createText3D(&mXAxisText);
        Geode *yAxisTextGeode = createText3D(&mYAxisText);
        Geode *zAxisTextGeode = createText3D(&mZAxisText);

        mGridInfoTextTrans->addChild(gridInfoTextGeode);
        mXAxisTextTrans->addChild(xAxisTextGeode);
        mYAxisTextTrans->addChild(yAxisTextGeode);
        mZAxisTextTrans->addChild(zAxisTextGeode);

        mAxisSwitch->addChild(mGridInfoTextTrans);
        mAxisSwitch->addChild(mXAxisTextTrans);
        mAxisSwitch->addChild(mYAxisTextTrans);
        mAxisSwitch->addChild(mZAxisTextTrans);
    }
}


/***************************************************************
* Function: setAxisMasking()
***************************************************************/
void CAVEGroupReferenceAxis::setAxisMasking(bool flag)
{
    mVisibleFlag = flag;
    if (flag) mAxisSwitch->setAllChildrenOn();
    else mAxisSwitch->setAllChildrenOff();
}


/***************************************************************
* Function: updateUnitGridSizeInfo()
***************************************************************/
void CAVEGroupReferenceAxis::updateUnitGridSizeInfo(const string &infoStr)
{
    if (mTextEnabledFlag)
    {
        mGridInfoText->setText("Unit size = " + infoStr);
        mGridInfoText->setPosition(Vec3(0, -gCharSize, -gCharSize));

        /* align the text to viewer's front direction */
        Matrixd rotMat;
        rotMat.makeRotate(Vec3(0, 1, 0), gPointerDir);
        mGridInfoTextTrans->setMatrix(Matrixd(rotMat)); 
    }
}


/***************************************************************
* Function: updateDiagonal()
*
* 'wireFrameVect': Lengths vector of the wire frame, used to
*  decide the size of axis system. 'solidShapeVect': Diagonal
*  vector of the actual solid shape, used to print out numbers
*  of dimensions that showed on each axis.
*
***************************************************************/
void CAVEGroupReferenceAxis::updateDiagonal(const osg::Vec3 &wireFrameVect, const osg::Vec3 &solidShapeVect)
{
    /* update axis */
    if (wireFrameVect.x() >= 0) 
        mXAxisGeode->setType(CAVEGeodeReferenceAxis::POS_X, &mXDirTrans);
    else 
        mXAxisGeode->setType(CAVEGeodeReferenceAxis::NEG_X, &mXDirTrans);

    if (wireFrameVect.y() >= 0) 
        mYAxisGeode->setType(CAVEGeodeReferenceAxis::POS_Y, &mYDirTrans);
    else 
        mYAxisGeode->setType(CAVEGeodeReferenceAxis::NEG_Y, &mYDirTrans);

    if (wireFrameVect.z() >= 0) 
        mZAxisGeode->setType(CAVEGeodeReferenceAxis::POS_Z, &mZDirTrans);
    else 
        mZAxisGeode->setType(CAVEGeodeReferenceAxis::NEG_Z, &mZDirTrans);

    mXAxisGeode->resize(wireFrameVect.x());
    mYAxisGeode->resize(wireFrameVect.y());
    mZAxisGeode->resize(wireFrameVect.z());

    /* update text */
    if (mTextEnabledFlag)
    {
        char lenstr[64];
        const float threshold = CAVEGeodeSnapWireframe::gSnappingUnitDist;
        const float lx = solidShapeVect.x() > 0 ? solidShapeVect.x(): -solidShapeVect.x();
        const float ly = solidShapeVect.y() > 0 ? solidShapeVect.y(): -solidShapeVect.y();
        const float lz = solidShapeVect.z() > 0 ? solidShapeVect.z(): -solidShapeVect.z();

        Matrixd rotMat;
        rotMat.makeRotate(Vec3(0, 1, 0), gPointerDir);

        if (lx >= threshold)
        {
            sprintf(lenstr, "%3.2f m", lx);
            mXAxisText->setText(string(lenstr));

            /* apply pointer oriented rotation and offset along the axis */
            Matrixd transMat;
            transMat.makeTranslate(Vec3(wireFrameVect.x(), 0, gCharSize));
            mXAxisTextTrans->setMatrix(rotMat * transMat);
        } 
        else 
        {
            mXAxisText->setText("");
        }

        if (ly >= threshold)
        {
            sprintf(lenstr, "%3.2f m", ly);
            mYAxisText->setText(string(lenstr));

            /* apply pointer oriented rotation and offset along the axis */
            Matrixd transMat;
            transMat.makeTranslate(Vec3(0, wireFrameVect.y(), gCharSize));
            mYAxisTextTrans->setMatrix(rotMat * transMat);
        } 
        else 
        {
            mYAxisText->setText("");
        }

        if (lz >= threshold)
        {
            sprintf(lenstr, "%3.2f m", lz);
            mZAxisText->setText(string(lenstr));

            /* apply pointer oriented rotation and offset along the axis */
            Matrixd transMat;
            transMat.makeTranslate(Vec3(0, -gCharSize, wireFrameVect.z()));
            mZAxisTextTrans->setMatrix(rotMat * transMat);
        } 
        else 
        {
            mZAxisText->setText("");
        }
    }
}


/***************************************************************
* Function: createText()
***************************************************************/
Geode *CAVEGroupReferenceAxis::createText3D(osgText::Text3D **text)
{
    Geode *textGeode = new Geode;
    *text = new osgText::Text3D;
    textGeode->addDrawable(*text);

    (*text)->setFont(CAVEGeode::getDataDir() + "Fonts/TN.ttf");
    (*text)->setCharacterSize(gCharSize, 0.7);
    (*text)->setCharacterDepth(gCharDepth);
    (*text)->setPosition(Vec3(0, 0, 0));
    (*text)->setAlignment(osgText::Text3D::CENTER_BOTTOM);
    (*text)->setDrawMode(osgText::Text3D::TEXT);
    (*text)->setAxisAlignment(osgText::Text3D::XZ_PLANE);
    (*text)->setRenderMode(osgText::Text3D::PER_GLYPH);
    (*text)->setText("");

    Material *material = new Material;
    material->setDiffuse(Material::FRONT_AND_BACK, Vec4(0, 1, 0, 1));
    material->setAmbient(Material::FRONT_AND_BACK, Vec4(0, 1, 0, 1));
    material->setAlpha(Material::FRONT_AND_BACK, 1.0f);

    StateSet *stateset = textGeode->getOrCreateStateSet();
    stateset->setAttributeAndModes(material, StateAttribute::OVERRIDE | StateAttribute::ON);
    stateset->setMode(GL_BLEND, StateAttribute::OVERRIDE | StateAttribute::ON );
    stateset->setRenderingHint(StateSet::TRANSPARENT_BIN);

    return textGeode;
}


// Constructor
CAVEGroupReferencePlane::CAVEGroupReferencePlane()
{
    mUnitGridSize = SnapLevelController::getInitSnappingLength();

    mXYPlaneSwitch = new Switch;
    mXZPlaneSwitch = new Switch;
    mYZPlaneSwitch = new Switch;

    mXYPlaneTrans = new MatrixTransform;
    mXZPlaneTrans = new MatrixTransform;
    mYZPlaneTrans = new MatrixTransform;

    mXYPlaneGeode = new CAVEGeodeReferencePlane;
    mXZPlaneGeode = new CAVEGeodeReferencePlane;
    mYZPlaneGeode = new CAVEGeodeReferencePlane;

    mMatrixTrans->addChild(mXYPlaneSwitch);
    mMatrixTrans->addChild(mXZPlaneSwitch);
    mMatrixTrans->addChild(mYZPlaneSwitch);

    mXYPlaneSwitch->addChild(mXYPlaneTrans);
    mXZPlaneSwitch->addChild(mXZPlaneTrans);
    mYZPlaneSwitch->addChild(mYZPlaneTrans);

    mXYPlaneTrans->addChild(mXYPlaneGeode);	
    mXZPlaneTrans->addChild(mXZPlaneGeode);	
    mYZPlaneTrans->addChild(mYZPlaneGeode);

    Matrixf rotXZMat, rotYZMat;
    rotXZMat.makeRotate(M_PI / 2, Vec3(1, 0, 0));
    rotYZMat.makeRotate(M_PI / 2, Vec3(0, 1, 0));
    mXZPlaneTrans->setMatrix(rotXZMat);
    mYZPlaneTrans->setMatrix(rotYZMat);

    mXYPlaneGeode->setGridColor(CAVEGeodeReferencePlane::BLUE);
    mXZPlaneGeode->setGridColor(CAVEGeodeReferencePlane::GREEN);
    mYZPlaneGeode->setGridColor(CAVEGeodeReferencePlane::GREEN);

    mXYPlaneGeode->setAlpha(0.4f);
    mXZPlaneGeode->setAlpha(0.4f);
    mYZPlaneGeode->setAlpha(0.4f);

    float len = CAVEGeodeReferencePlane::gSideLength;
    mXZPlaneGeode->resize(len, len * 0.005, mUnitGridSize);
    mYZPlaneGeode->resize(len * 0.005, len, mUnitGridSize);

    setPlaneMasking(false, false, false);
}


/***************************************************************
* Function: setCenterPos()
***************************************************************/
void CAVEGroupReferencePlane::setCenterPos(const osg::Vec3 &center, bool noSnap)
{
    Vec3 centerRounded;
    if (noSnap)
    {
        centerRounded = center;
    }
    else
    {
        /* snap 'center' vector with respect to mUnitGridSize */
        float snapUnitX, snapUnitY, snapUnitZ;
        snapUnitX = snapUnitY = snapUnitZ = mUnitGridSize;
        if (center.x() < 0) 
            snapUnitX = -mUnitGridSize;
        if (center.y() < 0) 
            snapUnitY = -mUnitGridSize;
        if (center.z() < 0) 
            snapUnitZ = -mUnitGridSize;
        int xSeg = (int)(abs((int)((center.x() + 0.5 * snapUnitX) / mUnitGridSize)));
        int ySeg = (int)(abs((int)((center.y() + 0.5 * snapUnitY) / mUnitGridSize)));
        int zSeg = (int)(abs((int)((center.z() + 0.5 * snapUnitZ) / mUnitGridSize)));
        centerRounded.x() = xSeg * snapUnitX;
        centerRounded.y() = ySeg * snapUnitY;
        centerRounded.z() = zSeg * snapUnitZ;
    

        /* set highlights of either XZ plane or YZ plane */
        Vec3 offset = centerRounded - mCenter;
        float offx = offset.x() * offset.x();
        float offy = offset.y() * offset.y();

        if (offx > offy) 
            mYZPlaneGeode->setAlpha(1.0f);
        else 
            mYZPlaneGeode->setAlpha(0.4f);

        if (offx < offy) 
            mXZPlaneGeode->setAlpha(1.0f);
        else 
            mXZPlaneGeode->setAlpha(0.4f);
    } 

    mCenter = centerRounded;
    Matrixf transMat;
    transMat.makeTranslate(mCenter);
    mMatrixTrans->setMatrix(transMat);
}


/***************************************************************
* Function: setUnitGridSize()
***************************************************************/
void CAVEGroupReferencePlane::setUnitGridSize(const float &gridsize)
{
    float len = CAVEGeodeReferencePlane::gSideLength;
    mXYPlaneGeode->resize(len, len, gridsize);
    mUnitGridSize = gridsize;
}


/***************************************************************
* Function: setPlaneMasking()
***************************************************************/
void CAVEGroupReferencePlane::setPlaneMasking(bool flagXY, bool flagXZ, bool flagYZ)
{
    if (flagXY) mXYPlaneSwitch->setAllChildrenOn();
    else mXYPlaneSwitch->setAllChildrenOff();

    if (flagXZ) mXZPlaneSwitch->setAllChildrenOn();
    else mXZPlaneSwitch->setAllChildrenOff();

    if (flagYZ) mYZPlaneSwitch->setAllChildrenOn();
    else mYZPlaneSwitch->setAllChildrenOff();
}

