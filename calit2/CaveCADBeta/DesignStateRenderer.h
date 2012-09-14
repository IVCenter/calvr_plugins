/***************************************************************
* File Name: DesignStateRenderer.h
*
* Class Name: DesignStateRenderer
*
***************************************************************/

#ifndef _DESIGN_STATE_RENDERER_H_
#define _DESIGN_STATE_RENDERER_H_


// C++
#include <iostream>
#include <list>

// Open scene graph
#include <osg/Light>
#include <osg/LightSource>
#include <osg/MatrixTransform>
#include <osg/Group>
#include <osg/Switch>

// Local includes
#include "DesignStates/DesignStateBase.h"
#include "DesignStates/DesignStateParticleSystem.h"
#include "DesignStates/DSVirtualSphere.h"
#include "DesignStates/DSVirtualEarth.h"
#include "DesignStates/DSParamountSwitch.h"
#include "DesignStates/DSSketchBook.h"
#include "DesignStates/DSGeometryCreator.h"
#include "DesignStates/DSGeometryEditor.h"
#include "DesignStates/DSTexturePallette.h"
#include "DesignStates/DSViewpoints.h"


/***************************************************************
* Class: DesignStateRenderer
***************************************************************/
class DesignStateRenderer
{
  public:
    DesignStateRenderer(osg::Group* designStateRootGroup);
    ~DesignStateRenderer();

    void setDesignObjectHandlerPtr(DesignObjectHandler *designObjectHandler);
    void setScenicHandlerPtr(VirtualScenicHandler *virtualScenicHandler);
    void setAudioConfigHandlerPtr(AudioConfigHandler *audioConfigHandler);

    /* functions activated by button input events */
    void toggleDSVisibility();
    void setVisible(bool flag);
    void switchToPrevState();
    void switchToNextState();
    void switchToPrevSubState();
    void switchToNextSubState();

    /* switch between lower/upper states: static callback functions called from separate 
       design states, which enables design states that not registered in 'mDSList'. */
    static void switchToLowerDesignState(const int &idx);
    static void switchToUpperDesignState(const int &idx);

    /* functions triggered by pointers */
    void inputDevMoveEvent(const osg::Vec3 &pointerOrg, const osg::Vec3 &pointerPos);
    bool inputDevPressEvent(const Vec3 &pointerOrg, const Vec3 &pointerPos);
    bool inputDevReleaseEvent();

    void update(const osg::Vec3 &viewDir, const osg::Vec3 &viewPos);

  protected:
    bool mFlagVisible;
    osg::MatrixTransform* mDSRootTrans;
    osg::Vec3 mViewOrg, mViewDir;
    osg::Light *mLeftDirLight, *mRightDirLight;

    /* design state objects */
    DesignStateParticleSystem *mDSParticleSystem;
    DesignStateList::iterator mActiveDSItr;
    DesignStateList mDSList;

    /* design states that registered in 'mDSList' */
    DesignStateBase *mActiveDSPtr;
    DSVirtualSphere *mDSVirtualSphere;
    DSVirtualEarth *mDSVirtualEarth;
    DSParamountSwitch *mDSParamountSwitch;
    DSSketchBook *mDSSketchBook;
    DSGeometryCreator *mDSGeometryCreator;
    DSTexturePallette *mDSTexturePallette;
    DSViewpoints *mDSViewpoints;

    /* design states that NOT registered in 'mDSList' */
    DSGeometryEditor *mDSGeometryEditor;

    /* reset position and rotation of DSRootTrans based on viewer's viewport */
    void resetPose();

    osg::Group *createDirectionalLights(osg::StateSet *stateset);

    static DesignStateRenderer *gDSRendererPtr;
};


#endif
