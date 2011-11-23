/***************************************************************
* Animation File Name: ANIMGeometryEditor.cpp
*
* Description: Load geometry creator shapes & objects
*
* Written by ZHANG Lelin on Jan 18, 2011
*
***************************************************************/
#include "ANIMGeometryEditor.h"

using namespace std;
using namespace osg;

namespace CAVEAnimationModeler
{


/***************************************************************
* Function: ANIMLoadGeometryEditorIconToolkits()
***************************************************************/
void ANIMLoadGeometryEditorIconToolkits(osg::MatrixTransform **iconToolkitTrans, 
				int &numToolkits, ANIMIconToolkitSwitchEntry ***iconToolkitSwitchEntryArray)
{
    *iconToolkitTrans = new MatrixTransform;

    /* with reference to the number of editting types defined in 'CAVEGroupIconToolkit' */
    numToolkits = 4;

    /* initialize toolkit switch entry array and add switch members to 'iconToolkitSwitch' */
    *iconToolkitSwitchEntryArray = new ANIMIconToolkitSwitchEntry*[numToolkits];
    for (int i = 0; i < numToolkits; i++)
    {
	(*iconToolkitSwitchEntryArray)[i] = new ANIMIconToolkitSwitchEntry;
	ANIMCreateSingleIconToolkitSwitchAnimation(i, &((*iconToolkitSwitchEntryArray)[i]));

	/* each type of editting tool contains one 'CAVEGroupIconToolkit' instance, which will be
	   added to the root group of 'iconToolkitSwitch' on top */
	(*iconToolkitTrans)->addChild((*iconToolkitSwitchEntryArray)[i]->mSwitch);
    }
}


/***************************************************************
* Function: ANIMCreateSingleIconToolkitSwitchAnimation()
***************************************************************/
void ANIMCreateSingleIconToolkitSwitchAnimation(int idx, ANIMIconToolkitSwitchEntry **iconToolkitEntry)
{
    (*iconToolkitEntry)->mSwitch = new Switch;
    PositionAttitudeTransform *iconToolkitPATransFwd = new PositionAttitudeTransform;
    PositionAttitudeTransform *iconToolkitPATransBwd = new PositionAttitudeTransform;

    /* 'mSwitch' has two decendents for foward and backward animations */
    (*iconToolkitEntry)->mSwitch->setAllChildrenOff();
    (*iconToolkitEntry)->mSwitch->addChild(iconToolkitPATransFwd);
    (*iconToolkitEntry)->mSwitch->addChild(iconToolkitPATransBwd);

    /* attach the same 'CAVEGroupIconToolkit' object to both 'PositionAttitudeTransform' */
    const CAVEGeodeIconToolkit::Type type = (CAVEGeodeIconToolkit::Type) idx;
    CAVEGroupIconToolkit *groupIconToolkit = new CAVEGroupIconToolkit(type);
    iconToolkitPATransFwd->addChild(groupIconToolkit);
    iconToolkitPATransBwd->addChild(groupIconToolkit);

    /* set up the forward / backward scale animation paths for toolkit icons */
    AnimationPath* animScaleFwd = new AnimationPath;
    AnimationPath* animScaleBwd = new AnimationPath;
    animScaleFwd->setLoopMode(AnimationPath::NO_LOOPING);
    animScaleBwd->setLoopMode(AnimationPath::NO_LOOPING);
   
    Vec3 scaleFwd, scaleBwd;
    float step = 1.f / ANIM_GEOMETRY_EDITOR_TOOLKIT_SHOWUP_SAMPS;
    for (int i = 0; i < ANIM_GEOMETRY_EDITOR_TOOLKIT_SHOWUP_SAMPS + 1; i++)
    {
	float val = i * step;
	scaleFwd = Vec3(val, val, val);
	scaleBwd = Vec3(1.f-val, 1.f-val, 1.f-val);
	animScaleFwd->insert(val, AnimationPath::ControlPoint(Vec3(), Quat(), scaleFwd));
	animScaleBwd->insert(val, AnimationPath::ControlPoint(Vec3(), Quat(), scaleBwd));
    }

    (*iconToolkitEntry)->mFwdAnimCallback = new AnimationPathCallback(animScaleFwd, 
					0.0, 1.f / ANIM_GEOMETRY_EDITOR_TOOLKIT_SHOWUP_TIME);
    (*iconToolkitEntry)->mBwdAnimCallback = new AnimationPathCallback(animScaleBwd, 
					0.0, 1.f / ANIM_GEOMETRY_EDITOR_TOOLKIT_SHOWUP_TIME);
    iconToolkitPATransFwd->setUpdateCallback((*iconToolkitEntry)->mFwdAnimCallback);
    iconToolkitPATransBwd->setUpdateCallback((*iconToolkitEntry)->mBwdAnimCallback);
}


};







