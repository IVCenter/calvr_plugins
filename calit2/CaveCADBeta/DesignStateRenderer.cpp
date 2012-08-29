/***************************************************************
* File Name: DesignStateRenderer.cpp
*
* Description: DesignStateRenderer only renders non-intersecatable
* objects attached to the virtaul sphere. Positions of input device
* will be passed to DesignObjectHandler where intersectable icons
* are rendered and intersection tests are performed.
*
* Written by ZHANG Lelin on Sep 15, 2010
*
***************************************************************/
#include "DesignStateRenderer.h"

using namespace std;
using namespace osg;


DesignStateRenderer *DesignStateRenderer::gDSRendererPtr(NULL);

//Constructor
DesignStateRenderer::DesignStateRenderer(osg::Group* designStateRootGroup): mFlagVisible(false),
				mViewOrg(Vec3(0, -0.5, 0)), mViewDir(Vec3(0, 1, 0))
{
    /* initialize design state objects that registered in 'mDSList'*/
    mDSParticleSystem = new DesignStateParticleSystem();
    mDSVirtualSphere = new DSVirtualSphere();
    mDSVirtualEarth = new DSVirtualEarth();
    mDSParamountSwitch = new DSParamountSwitch();
    mDSSketchBook = new DSSketchBook();
    mDSGeometryCreator = new DSGeometryCreator();
    mDSTexturePallette = new DSTexturePallette();
    mActiveDSPtr = mDSGeometryCreator;//VirtualSphere;

    /* push state object transforms to list, attach them to scene graph */
    mDSList.clear();
    //mDSList.push_back(mDSVirtualSphere);
    mDSList.push_back(mDSVirtualEarth);
    mDSList.push_back(mDSParamountSwitch);
    mDSList.push_back(mDSSketchBook);
    mDSList.push_back(mDSGeometryCreator);
    mDSList.push_back(mDSTexturePallette);
    mActiveDSItr = mDSList.begin();

    mDSRootTrans = new MatrixTransform();
    //mDSRootTrans->addChild(mDSParticleSystem);
    //mDSRootTrans->addChild(mDSVirtualSphere);
    mDSRootTrans->addChild(mDSVirtualEarth);
    mDSRootTrans->addChild(mDSParamountSwitch);
    mDSRootTrans->addChild(mDSSketchBook);
    mDSRootTrans->addChild(mDSGeometryCreator);
    mDSRootTrans->addChild(mDSTexturePallette);
    designStateRootGroup->addChild(mDSRootTrans);

    for (DesignStateList::iterator itrDS = mDSList.begin(); itrDS != mDSList.end(); itrDS++)
    {
        (*itrDS)->setParticleSystemPtr(mDSParticleSystem);
    }

    /* initialize design state objects that NOT registered in 'mDSList'*/
    mDSGeometryEditor = new DSGeometryEditor();
    mDSGeometryEditor->setParticleSystemPtr(mDSParticleSystem);
    mDSRootTrans->addChild(mDSGeometryEditor);

    mDSGeometryEditor->addUpperDesignState(mDSGeometryCreator);
    mDSGeometryEditor->setUpperStateSwitchCallback(switchToUpperDesignState);
    mDSGeometryCreator->addLowerDesignState(mDSGeometryEditor);
    mDSGeometryCreator->setLowerStateSwitchCallback(switchToLowerDesignState);

    /* disable all virtual objects */
    mDSParticleSystem->setEmitterEnabled(false);
    mActiveDSPtr->setObjectEnabled(false);

    /* create two directional light sources for all DS objects */
    designStateRootGroup->addChild(createDirectionalLights(designStateRootGroup->getOrCreateStateSet()));

    /* set instance pointer to 'this', which will be used in all static callbacks */
    gDSRendererPtr = this;

    DesignStateBase::setDesignStateRootGroupPtr(designStateRootGroup);
}


//Destructor
DesignStateRenderer::~DesignStateRenderer()
{
}


/***************************************************************
* Function: setDesignObjectHandlerPtr()
***************************************************************/
void DesignStateRenderer::setDesignObjectHandlerPtr(DesignObjectHandler *designObjectHandler)
{
    DesignStateBase::setDesignObjectRootGroupPtr(designObjectHandler->getIntersectableSceneGraphPtr());

    mDSGeometryCreator->setDesignObjectHandlerPtr(designObjectHandler);
    mDSGeometryEditor->setDesignObjectHandlerPtr(designObjectHandler);
}


/***************************************************************
* Function: setScenicHandlerPtr()
***************************************************************/
void DesignStateRenderer::setScenicHandlerPtr(VirtualScenicHandler *virtualScenicHandler)
{
    mDSVirtualEarth->setScenicHandlerPtr(virtualScenicHandler);
    mDSParamountSwitch->setScenicHandlerPtr(virtualScenicHandler);
    mDSSketchBook->setScenicHandlerPtr(virtualScenicHandler);
}


/***************************************************************
* Function: setAudioConfigHandlerPtr()
***************************************************************/
void DesignStateRenderer::setAudioConfigHandlerPtr(AudioConfigHandler *audioConfigHandler)
{
    mDSGeometryCreator->setAudioConfigHandlerPtr(audioConfigHandler);
    mDSGeometryEditor->setAudioConfigHandlerPtr(audioConfigHandler);
    mDSTexturePallette->setAudioConfigHandlerPtr(audioConfigHandler);
}


/***************************************************************
* Function: setVisible()
***************************************************************/
void DesignStateRenderer::setVisible(bool flag)
{
    mFlagVisible = flag;
//    mActiveDSPtr->setObjectEnabled(flag);

    mDSVirtualEarth->setObjectEnabled(flag);
    mDSParamountSwitch->setObjectEnabled(flag);
    mDSSketchBook->setObjectEnabled(flag);
    mDSGeometryCreator->setObjectEnabled(flag);
    mDSTexturePallette->setObjectEnabled(flag);

    /* align state spheres with viewer's position and front orientation */
    if (flag) resetPose();
}


/***************************************************************
* Function: toggleDSVisibility()
***************************************************************/
void DesignStateRenderer::toggleDSVisibility()
{
    mFlagVisible = !mFlagVisible;
    setVisible(mFlagVisible);
}


/***************************************************************
* Function: switchToPrevState()
*
* Turn the particle system on when switching to a new state
*
***************************************************************/
void DesignStateRenderer::switchToPrevState()
{
    if (!mFlagVisible || mDSList.size() <= 0) return;

    if (!mActiveDSPtr->isLocked())
    {
        mActiveDSPtr->setObjectEnabled(false);
        if (++mActiveDSItr == mDSList.end()) 
        {
            mActiveDSItr = mDSList.begin();
        }
        mActiveDSPtr = dynamic_cast <DesignStateBase*> (*mActiveDSItr);
        if (mActiveDSPtr)
        {
            mActiveDSPtr->setObjectEnabled(true);
            resetPose();
        }
    }
    // else: pass the button event to existing rendering state
}


/***************************************************************
* Function: switchToNextState()
*
* Turn the particle system on when switching to a new state
*
***************************************************************/
void DesignStateRenderer::switchToNextState()
{
    if (!mFlagVisible || mDSList.size() <= 0) return;

    if (!mActiveDSPtr->isLocked())
    {
        mActiveDSPtr->setObjectEnabled(false);
        if (mActiveDSItr == mDSList.begin()) 
        {
            mActiveDSItr = mDSList.end();
        }
        mActiveDSItr--;
        mActiveDSPtr = dynamic_cast <DesignStateBase*> (*mActiveDSItr);
        if (mActiveDSPtr)
        {
            mActiveDSPtr->setObjectEnabled(true);
            resetPose();
        }
    }
    // else: pass the button event to existing rendering state
}


/***************************************************************
* Static Function: switchToLowerState()
*
* Will change pointer value of 'mActiveDSPtr', but 'mActiveDSItr'
* would remain the same in order to trace position in 'mDSList'.
*
***************************************************************/
void DesignStateRenderer::switchToLowerDesignState(const int &idx)
{
    if (!gDSRendererPtr) return;
    if (!(gDSRendererPtr->mFlagVisible) || gDSRendererPtr->mDSList.size() <= 0 
	|| gDSRendererPtr->mActiveDSPtr->isLocked()) return;

    DesignStateBase **dsPtr = &(gDSRendererPtr->mActiveDSPtr);
    (*dsPtr)->setObjectEnabled(false);
    (*dsPtr) = (*dsPtr)->getLowerDesignState(idx);
    (*dsPtr)->setObjectEnabled(true);
}


/***************************************************************
* Static Function: switchToUpperState()
***************************************************************/
void DesignStateRenderer::switchToUpperDesignState(const int &idx)
{
    if (!gDSRendererPtr) return;
    if (!(gDSRendererPtr->mFlagVisible) || gDSRendererPtr->mDSList.size() <= 0 
	|| gDSRendererPtr->mActiveDSPtr->isLocked()) return;

    DesignStateBase **dsPtr = &(gDSRendererPtr->mActiveDSPtr);
    (*dsPtr)->setObjectEnabled(false);
    (*dsPtr) = (*dsPtr)->getUpperDesignState(idx);
    (*dsPtr)->setObjectEnabled(true);
}


/***************************************************************
* Function: switchToPrevSubState()
***************************************************************/
void DesignStateRenderer::switchToPrevSubState()
{
    mActiveDSPtr->switchToPrevSubState();
}


/***************************************************************
* Function: switchToNextSubState()
***************************************************************/
void DesignStateRenderer::switchToNextSubState()
{
    mActiveDSPtr->switchToNextSubState();
}


/***************************************************************
* Function: inputDevMoveEvent()
*
* Description: Update virtual ball's position
*
***************************************************************/
void DesignStateRenderer::inputDevMoveEvent(const Vec3 &pointerOrg, const Vec3 &pointerPos)
{
    mActiveDSPtr->inputDevMoveEvent(pointerOrg, pointerPos); 
    
/*   mDSVirtualEarth->inputDevMoveEvent(pointerOrg, pointerPos);
    mDSParamountSwitch->inputDevMoveEvent(pointerOrg, pointerPos);
    mDSSketchBook->inputDevMoveEvent(pointerOrg, pointerPos);
    mDSGeometryCreator->inputDevMoveEvent(pointerOrg, pointerPos);
    mDSTexturePallette->inputDevMoveEvent(pointerOrg, pointerPos);
*/
    resetPose();
}


/***************************************************************
* Function: update()
***************************************************************/
void DesignStateRenderer::update(const osg::Vec3 &viewDir, const osg::Vec3 &viewPos)
{
    mViewOrg = viewPos;
    mViewDir = viewDir;

    mActiveDSPtr->update();
    if (mDSVirtualEarth) 
        mDSVirtualEarth->updateVSParameters(viewDir, viewPos);
}


/***************************************************************
* Function: inputDevPressEvent()
***************************************************************/
bool DesignStateRenderer::inputDevPressEvent(const Vec3 &pointerOrg, const Vec3 &pointerPos)
{
    if (mDSVirtualEarth->test(pointerOrg, pointerPos))
    {
        if (mActiveDSPtr != mDSVirtualEarth)
        {
            mActiveDSPtr->setObjectEnabled(false);
        }
        mActiveDSPtr = mDSVirtualEarth;
        mActiveDSPtr->setObjectEnabled(true);
    }
    else if (mDSParamountSwitch->test(pointerOrg, pointerPos))
    {
        if (mActiveDSPtr != mDSParamountSwitch)
            mActiveDSPtr->setObjectEnabled(false);
        mActiveDSPtr = mDSParamountSwitch;
        mActiveDSPtr->setObjectEnabled(true);
    }
    else if (mDSSketchBook->test(pointerOrg, pointerPos))
    {
        if (mActiveDSPtr != mDSSketchBook)
            mActiveDSPtr->setObjectEnabled(false);
        mActiveDSPtr = mDSSketchBook;
        mActiveDSPtr->setObjectEnabled(true);
    }
    else if (mDSGeometryCreator->test(pointerOrg, pointerPos))
    {
        if (mActiveDSPtr != mDSGeometryCreator)
            mActiveDSPtr->setObjectEnabled(false);
        mActiveDSPtr = mDSGeometryCreator;
        mActiveDSPtr->setObjectEnabled(true);
    }
    else if (mDSTexturePallette->test(pointerOrg, pointerPos))
    {
        if (mActiveDSPtr != mDSTexturePallette)
            mActiveDSPtr->setObjectEnabled(false);
        mActiveDSPtr = mDSTexturePallette;
        mActiveDSPtr->setObjectEnabled(true);
    }    

    return (mActiveDSPtr->inputDevPressEvent(pointerOrg, pointerPos));
}


/***************************************************************
* Function: inputDevReleaseEvent()
***************************************************************/
bool DesignStateRenderer::inputDevReleaseEvent()
{
    return (mActiveDSPtr->inputDevReleaseEvent());
}


/***************************************************************
* Function: resetPose()
***************************************************************/
void DesignStateRenderer::resetPose()
{
    /* calculate position and front direction: project viewer's front 
       to XY plane and use that as front orientation, to make sure 
       all design tools are aligned straight up when showing up. */

    float height = 0.5;

    Vec3 pos = mViewOrg + mViewDir * ANIM_VIRTUAL_SPHERE_DISTANCE * 0.5;
    pos[2] += height;
    Vec3 front = mViewDir;
    front.z() = 0;
    front.normalize();

    Matrixf transMat, rotMat;
    transMat.makeTranslate(pos);
    rotMat.makeRotate(Vec3(0, 1, 0), front);
    mDSRootTrans->setMatrix(rotMat * transMat);

    /* pass new position & orientation to static members of design state base */
    DesignStateBase::setDesignStateCenterPos(pos);
    DesignStateBase::setDesignStateFrontVect(front);

    /* update two directional lights' position */
    Vec4 leftDir = Vec4(1.0f, -1.0f, 1.0f, 0.0f);
    Vec4 rightDir = Vec4(-1.0f, -1.0f, 1.0f, 0.0f);
    leftDir = leftDir * rotMat;
    rightDir = rightDir * rotMat;
    mLeftDirLight->setPosition(leftDir);
    mRightDirLight->setPosition(rightDir);
}


/***************************************************************
* Function: createSunLight()
***************************************************************/
Group *DesignStateRenderer::createDirectionalLights(osg::StateSet *stateset)
{
    Group* lightGroup = new Group;
    LightSource *leftLS = new LightSource;
    LightSource *rightLS = new LightSource;
    mLeftDirLight = new Light;
    mRightDirLight = new Light;

    /* Directional vector = (1.0f, -1.0f, 1.0f) in relative to viewer's position */
    mLeftDirLight->setLightNum(6);
    mLeftDirLight->setPosition(Vec4(1.0f, -1.0f, 1.0f, 0.0f));
    mLeftDirLight->setDiffuse(Vec4(1.0f,1.0f,1.0f,1.0f));
    mLeftDirLight->setSpecular(Vec4(0.0f,0.0f,0.0f,1.0f));
    mLeftDirLight->setAmbient(Vec4(0.5f,0.5f,0.5f,1.0f));

    /* Directional vector = (-1.0f, -1.0f, 1.0f) in relative to viewer's position */
    mRightDirLight->setLightNum(7);
    mRightDirLight->setPosition(Vec4(-1.0f, -1.0f, 1.0f, 0.0f));
    mRightDirLight->setDiffuse(Vec4(1.0f,1.0f,1.0f,1.0f));
    mRightDirLight->setSpecular(Vec4(0.0f,0.0f,0.0f,1.0f));
    mRightDirLight->setAmbient(Vec4(0.5f,0.5f,0.5f,1.0f));
  
    leftLS->setLight(mLeftDirLight);
    leftLS->setLocalStateSetModes(StateAttribute::ON);
    leftLS->setStateSetModes(*stateset, StateAttribute::ON);

    rightLS->setLight(mRightDirLight);
    rightLS->setLocalStateSetModes(StateAttribute::ON);
    rightLS->setStateSetModes(*stateset, StateAttribute::ON);

    lightGroup->addChild(leftLS);
    lightGroup->addChild(rightLS);

    return lightGroup;
}

