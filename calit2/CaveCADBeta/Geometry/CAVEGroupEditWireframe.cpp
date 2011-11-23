/***************************************************************
* File Name: CAVEGroupEditWireframe.cpp
*
* Description:  
*
* Written by ZHANG Lelin on Feb 2, 2011
*
***************************************************************/
#include "CAVEGroupEditWireframe.h"


using namespace std;
using namespace osg;


const float CAVEGroupEditWireframe::gCharSize(0.08f);
const float CAVEGroupEditWireframe::gCharDepth(0.012f);

osg::Vec3 CAVEGroupEditWireframe::gPointerDir(Vec3(0, 1, 0));


// Constructor: CAVEGroupEditWireframe
CAVEGroupEditWireframe::CAVEGroupEditWireframe(): mPrimaryFlag(false), mBoundingRadius(0.f),
	mMoveSVect(Vec3s(0, 0, 0)), mRotateSVect(Vec3s(0, 0, 0)), mScaleNumSegs(0), mScaleUnitVect(Vec3(0, 0, 0))
{
    /* create editting info text */
    mEditInfoTextTrans = new MatrixTransform();
    mEditInfoTextSwitch = new Switch();
    mEditInfoTextSwitch->setAllChildrenOff();
    Geode *textGeode = createText3D(&mEditInfoText);
    mEditInfoTextSwitch->addChild(textGeode);
    mEditInfoTextTrans->addChild(mEditInfoTextSwitch);
    addChild(mEditInfoTextTrans);
}


/***************************************************************
* Function: resetInfoOrientation()
***************************************************************/
void CAVEGroupEditWireframe::resetInfoOrientation()
{
    /* update translation matrix of 'mEditInfoTextTrans', align the text to viewer's front direction */
    Matrixd textRotMat, textTransMat;
    textTransMat.makeTranslate(Vec3(0, -mBoundingRadius * 0.5f, mBoundingRadius * 0.866f));
    textRotMat.makeRotate(Vec3(0, 1, 0), gPointerDir);
    mEditInfoTextTrans->setMatrix(mBoundSphereScaleMat * textTransMat * textRotMat);
}


/***************************************************************
* Function: setPointerDir()
***************************************************************/
void CAVEGroupEditWireframe::setPointerDir(const osg::Vec3 &pointerDir)
{
    /* update pointer front direction by projecting it onto XY plane */
    Vec3 pointerFront = pointerDir;
    pointerFront.z() = 0;
    pointerFront.normalize();

    gPointerDir = pointerFront;
}


/***************************************************************
* Function: createText()
***************************************************************/
Geode *CAVEGroupEditWireframe::createText3D(osgText::Text3D **text)
{
    Geode *textGeode = new Geode;
    *text = new osgText::Text3D;
    textGeode->addDrawable(*text);

    (*text)->setFont(CAVEGeode::getDataDir() + "Fonts/TN.ttf");
    (*text)->setCharacterSize(gCharSize, 0.7);
    (*text)->setCharacterDepth(gCharDepth);
    (*text)->setLineSpacing(0.25f);
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
















