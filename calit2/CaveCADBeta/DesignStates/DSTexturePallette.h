/***************************************************************
* File Name: DSTexturePallette.h
*
* Description: Derived class from DesignStateBase
*
***************************************************************/

#ifndef _DS_TEXTURE_PALLETTE_H_
#define _DS_TEXTURE_PALLETTE_H_


// Local include
#include "DesignStateBase.h"
#include "../AudioConfigHandler.h"
#include "../TrackballController.h"
#include "../AnimationModeler/ANIMTexturePallette.h"
#include "../ColorSelector.h"

/***************************************************************
* Class: DSTexturePallette
***************************************************************/
class DSTexturePallette: public DesignStateBase
{
  public:
    DSTexturePallette();
    ~DSTexturePallette();

    // virtual functions inherited from base class
    virtual void setObjectEnabled(bool flag);
    virtual void switchToPrevSubState();
    virtual void switchToNextSubState();

    void inputDevMoveEvent(const osg::Vec3 &pointerOrg, const osg::Vec3 &pointerPos);
    bool inputDevPressEvent(const osg::Vec3 &pointerOrg, const osg::Vec3 &pointerPos);
    bool inputDevReleaseEvent();
    void update();
    void setHighlight(bool isHighlighted, const osg::Vec3 &pointerOrg, const osg::Vec3 &pointerPos);

    // definition of texturing states
    enum TexturingState
    {
        IDLE,
        SELECT_TEXTURE,
        APPLY_TEXTURE
    };

    void setAudioConfigHandlerPtr(AudioConfigHandler *audioConfigHandler) { mAudioConfigHandler = audioConfigHandler; }

  protected:

    // OSG objects as decendents in DSTexturePallette
    osg::Switch *mIdleStateSwitch, *mSelectStateSwitch, *mAlphaTurnerSwitch;
    osg::ShapeDrawable *mSD;
    ColorSelector *mColorSelector;
    osg::PositionAttitudeTransform *mButtonFwd, *mButtonBwd;
    bool mIsOpen;
    std::vector<osg::PositionAttitudeTransform*> mFwdVec, mBwdVec, mHighlightVec;

    int mTexIndex, mNumTexs, mColorIndex;
    CAVEAnimationModeler::ANIMTexturePalletteIdleEntry *mTextureStatesIdleEntry;
    CAVEAnimationModeler::ANIMTexturePalletteSelectEntry **mTextureStatesSelectEntryArray;

    std::vector<CAVEAnimationModeler::ANIMTexturePalletteSelectEntry*> mTextureEntries, mColorEntries;

    // states and pointers that handles device inputs
    TexturingState mTexturingState;
    DesignObjectHandler *mDesignObjectHandler;

    // audio config handler: update audio parameters as textures are adapted
    AudioConfigHandler *mAudioConfigHandler;

    void resetIntersectionRootTarget();
    void texturingStateTransitionHandle(const TexturingState& prevState, const TexturingState& nextState);
};


#endif

