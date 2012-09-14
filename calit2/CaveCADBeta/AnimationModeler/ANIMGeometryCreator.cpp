/***************************************************************
* Animation File Name: ANIMGeometryCreator.cpp
*
* Description: Load geometry creator shapes & objects
*
* Written by ZHANG Lelin on Nov 10, 2010
*
***************************************************************/
#include "ANIMGeometryCreator.h"

using namespace std;
using namespace osg;

namespace CAVEAnimationModeler
{

/***************************************************************
* Function: ANIMLoadGeometryCreator()
*
* xformScaleFwd: Root transform node for inflating geometries
* xformScaleBwd: Root transform node for shrinking geometries
* sphereExteriorSwitch: Switch control for single exterior sphere
*
***************************************************************/
void ANIMLoadGeometryCreator(PositionAttitudeTransform** xformScaleFwd, PositionAttitudeTransform** xformScaleBwd,
			     Switch **sphereExteriorSwitch, Geode **sphereExteriorGeode,
			     int &numTypes, ANIMShapeSwitchEntry ***shapeSwitchEntryArray)
{
    *xformScaleFwd = new PositionAttitudeTransform;
    *xformScaleBwd = new PositionAttitudeTransform;

    MatrixTransform *geomCreatorTrans = new MatrixTransform;

    MatrixTransform *sphereExteriorTrans = new MatrixTransform;
    *sphereExteriorSwitch = new Switch;
    Switch *createBoxSwitch = new Switch;
    Switch *createCylinderSwitch = new Switch;

    (*xformScaleFwd)->addChild(geomCreatorTrans);
    (*xformScaleBwd)->addChild(geomCreatorTrans);

    geomCreatorTrans->addChild(*sphereExteriorSwitch);
    geomCreatorTrans->addChild(createBoxSwitch);
    geomCreatorTrans->addChild(createCylinderSwitch);
    
    osg::Vec3 pos(-1, 0, 0);

    // create drawables, geodes and attach them to animation switches
    *sphereExteriorGeode = new Geode();
    Sphere *sphere = new Sphere(osg::Vec3(), ANIM_VIRTUAL_SPHERE_RADIUS);
    ShapeDrawable *sphereDrawable = new ShapeDrawable(sphere);
    (*sphereExteriorGeode)->addDrawable(sphereDrawable);


    Box *box = new Box(osg::Vec3(0.1, 0, 0), ANIM_VIRTUAL_SPHERE_RADIUS / 1.9);
    (*sphereExteriorGeode)->addDrawable(new ShapeDrawable(box));

    float r = ANIM_VIRTUAL_SPHERE_RADIUS / 3.0;
    Cylinder *cylinder = new Cylinder(osg::Vec3(-0.05, 0, -0.05), r, r * 2);
    (*sphereExteriorGeode)->addDrawable(new ShapeDrawable(cylinder));
    
    Cone *cone = new osg::Cone(osg::Vec3(0, -0.1, 0.05), r, r * 2);
    (*sphereExteriorGeode)->addDrawable(new ShapeDrawable(cone));


    Material *transmaterial = new Material;
    transmaterial->setDiffuse(Material::FRONT_AND_BACK, Vec4(1, 1, 1, 1));
    transmaterial->setAlpha(Material::FRONT_AND_BACK, 0.6f);

    Image* envMap = osgDB::readImageFile(ANIMDataDir() + "Textures/ShapeContainer.JPG");
    Texture2D* envTex = new Texture2D(envMap);    
    
    StateSet *sphereStateSet = (sphereDrawable)->getOrCreateStateSet();
    sphereStateSet->setMode(GL_BLEND, StateAttribute::OVERRIDE | StateAttribute::ON );
    sphereStateSet->setRenderingHint(StateSet::TRANSPARENT_BIN);
    sphereStateSet->setAttributeAndModes(transmaterial, StateAttribute::OVERRIDE | StateAttribute::ON);
    sphereStateSet->setTextureAttributeAndModes(0, envTex, StateAttribute::ON);
    sphereStateSet->setMode(GL_CULL_FACE, StateAttribute::ON);

    sphereExteriorTrans->addChild(*sphereExteriorGeode);
    (*sphereExteriorSwitch)->addChild(sphereExteriorTrans);
    (*sphereExteriorSwitch)->setAllChildrenOn();

    // write into shape switch entry array record
    numTypes = 2;
    *shapeSwitchEntryArray = new ANIMShapeSwitchEntry*[numTypes];
    (*shapeSwitchEntryArray)[0] = new ANIMShapeSwitchEntry;
    (*shapeSwitchEntryArray)[1] = new ANIMShapeSwitchEntry;
    (*shapeSwitchEntryArray)[0]->mSwitch = createBoxSwitch;
    (*shapeSwitchEntryArray)[1]->mSwitch = createCylinderSwitch;

    ANIMCreateSingleShapeSwitchAnimation(&((*shapeSwitchEntryArray)[0]), CAVEGeodeShape::BOX);
    ANIMCreateSingleShapeSwitchAnimation(&((*shapeSwitchEntryArray)[1]), CAVEGeodeShape::CYLINDER);

    /* set up the forward / backward scale animation paths for geometry creator */
    AnimationPath* animationPathScaleFwd = new AnimationPath;
    AnimationPath* animationPathScaleBwd = new AnimationPath;
    animationPathScaleFwd->setLoopMode(AnimationPath::NO_LOOPING);
    animationPathScaleBwd->setLoopMode(AnimationPath::NO_LOOPING);
   
    Vec3 scaleFwd, scaleBwd;
    float step = 1.f / ANIM_VIRTUAL_SPHERE_NUM_SAMPS;
    for (int i = 0; i < ANIM_VIRTUAL_SPHERE_NUM_SAMPS + 1; i++)
    {
        float val = i * step;
        scaleFwd = Vec3(val, val, val);
        scaleBwd = Vec3(1.f-val, 1.f-val, 1.f-val);
        animationPathScaleFwd->insert(val, AnimationPath::ControlPoint(pos, Quat(), scaleFwd));
        animationPathScaleBwd->insert(val, AnimationPath::ControlPoint(pos, Quat(), scaleBwd));
    }

    AnimationPathCallback *animCallbackFwd = new AnimationPathCallback(animationPathScaleFwd, 
						0.0, 1.f / ANIM_VIRTUAL_SPHERE_LAPSE_TIME);
    AnimationPathCallback *animCallbackBwd = new AnimationPathCallback(animationPathScaleBwd, 
						0.0, 1.f / ANIM_VIRTUAL_SPHERE_LAPSE_TIME); 
    (*xformScaleFwd)->setUpdateCallback(animCallbackFwd);
    (*xformScaleBwd)->setUpdateCallback(animCallbackBwd);
}


/***************************************************************
* Function: ANIMCreateSingleShapeSwitchAnimation()
***************************************************************/
void ANIMCreateSingleShapeSwitchAnimation(ANIMShapeSwitchEntry **shapeEntry, const CAVEGeodeShape::Type &typ)
{
    PositionAttitudeTransform *flipUpFwdTrans = new PositionAttitudeTransform;
    PositionAttitudeTransform *flipDownFwdTrans = new PositionAttitudeTransform;
    PositionAttitudeTransform *flipUpBwdTrans = new PositionAttitudeTransform;
    PositionAttitudeTransform *flipDownBwdTrans = new PositionAttitudeTransform;

    (*shapeEntry)->mSwitch->addChild(flipUpFwdTrans);		// child #0
    (*shapeEntry)->mSwitch->addChild(flipDownFwdTrans);		// child #1
    (*shapeEntry)->mSwitch->addChild(flipUpBwdTrans);		// child #2
    (*shapeEntry)->mSwitch->addChild(flipDownBwdTrans);		// child #3
    (*shapeEntry)->mSwitch->setAllChildrenOff();

    osg::Vec3 pos(0, 0, 0);

    /* create shape geode based on 'ANIMShapeSwitchEntry::Type' */
    Geode *shapeGeode = new Geode;
    if (typ == CAVEGeodeShape::BOX)
    {
        Box *box = new Box(osg::Vec3(), ANIM_VIRTUAL_SPHERE_RADIUS / 0.9);
        shapeGeode->addDrawable(new ShapeDrawable(box));
    }
    else if (typ == CAVEGeodeShape::CYLINDER)
    {
        float r = ANIM_VIRTUAL_SPHERE_RADIUS / 1.5;
        Cylinder *cylinder = new Cylinder(osg::Vec3(), r, r * 2);
        shapeGeode->addDrawable(new ShapeDrawable(cylinder));
    }
    flipUpFwdTrans->addChild(shapeGeode);
    flipDownFwdTrans->addChild(shapeGeode);
    flipUpBwdTrans->addChild(shapeGeode);
    flipDownBwdTrans->addChild(shapeGeode);

    /* set up flip up / flip down animation paths for shape switch */
    AnimationPath* animationFlipUpFwd = new AnimationPath;
    AnimationPath* animationFlipDownFwd = new AnimationPath;
    AnimationPath* animationFlipUpBwd = new AnimationPath;
    AnimationPath* animationFlipDownBwd = new AnimationPath;
    animationFlipUpFwd->setLoopMode(AnimationPath::NO_LOOPING);
    animationFlipDownFwd->setLoopMode(AnimationPath::NO_LOOPING);
    animationFlipUpBwd->setLoopMode(AnimationPath::NO_LOOPING);
    animationFlipDownBwd->setLoopMode(AnimationPath::NO_LOOPING);
   
    Vec3 scaleUpVect, scaleDownVect;
    Quat flipUpFwdQuat, flipDownFwdQuat, flipUpBwdQuat, flipDownBwdQuat;
    float timestep = ANIM_GEOMETRY_CREATOR_SHAPE_FLIP_TIME / ANIM_GEOMETRY_CREATOR_SHAPE_FLIP_SAMPS;
    float scalestep = 1.f / ANIM_GEOMETRY_CREATOR_SHAPE_FLIP_SAMPS;
    float anglestep = M_PI * 0.5 / ANIM_GEOMETRY_CREATOR_SHAPE_FLIP_SAMPS;

    if (typ == CAVEGeodeShape::BOX)
        pos[2] -= 0.5;
    else if (typ == CAVEGeodeShape::CYLINDER)
        pos[2] -= 1.0;

    osg::Vec3 diff, startPos(0,0,0), fwd, bwd;

    for (int i = 0; i < ANIM_GEOMETRY_CREATOR_SHAPE_FLIP_SAMPS + 1; i++)
    {
        float t = i * timestep;
        float val = i * scalestep;
        scaleUpVect = Vec3(val, val, val);
        scaleDownVect = Vec3(1.f-val, 1.f-val, 1.f-val);

        flipUpFwdQuat = Quat(i * anglestep - M_PI / 2, Vec3(1, 0, 0));
        flipDownFwdQuat = Quat(i * anglestep, Vec3(1, 0, 0));
        flipUpBwdQuat = Quat(i * anglestep - M_PI / 2, Vec3(-1, 0, 0));
        flipDownBwdQuat = Quat(i * anglestep, Vec3(-1, 0, 0));

        diff = startPos - pos;
        
        for (int j = 0; j < 3; ++j)
            diff[j] *= val; 
        fwd = startPos - diff;
        bwd = pos + diff;

        animationFlipUpFwd->insert(t, AnimationPath::ControlPoint(fwd, flipUpFwdQuat, scaleUpVect));
        animationFlipDownFwd->insert(t, AnimationPath::ControlPoint(fwd, flipDownFwdQuat, scaleUpVect));
        animationFlipUpBwd->insert(t, AnimationPath::ControlPoint(bwd, flipUpBwdQuat, scaleDownVect));
        animationFlipDownBwd->insert(t, AnimationPath::ControlPoint(bwd, flipDownBwdQuat, scaleDownVect));
    }

    AnimationPathCallback *animCallbackFlipUpFwd = new AnimationPathCallback(animationFlipUpFwd, 
						0.0, 1.f / ANIM_GEOMETRY_CREATOR_SHAPE_FLIP_TIME);
    AnimationPathCallback *animCallbackFlipDownFwd = new AnimationPathCallback(animationFlipDownFwd, 
						0.0, 1.f / ANIM_GEOMETRY_CREATOR_SHAPE_FLIP_TIME);
    AnimationPathCallback *animCallbackFlipUpBwd = new AnimationPathCallback(animationFlipUpBwd, 
						0.0, 1.f / ANIM_GEOMETRY_CREATOR_SHAPE_FLIP_TIME);
    AnimationPathCallback *animCallbackFlipDownBwd = new AnimationPathCallback(animationFlipDownBwd, 
						0.0, 1.f / ANIM_GEOMETRY_CREATOR_SHAPE_FLIP_TIME);
    flipUpFwdTrans->setUpdateCallback(animCallbackFlipUpFwd);
    flipDownFwdTrans->setUpdateCallback(animCallbackFlipDownFwd);
    flipUpBwdTrans->setUpdateCallback(animCallbackFlipUpBwd);
    flipDownBwdTrans->setUpdateCallback(animCallbackFlipDownBwd);

    /* write into shape switch entry array record*/
    (*shapeEntry)->mFlipUpFwdAnim = animCallbackFlipUpFwd;
    (*shapeEntry)->mFlipDownFwdAnim = animCallbackFlipDownFwd;
    (*shapeEntry)->mFlipUpBwdAnim = animCallbackFlipUpBwd;
    (*shapeEntry)->mFlipDownBwdAnim = animCallbackFlipDownBwd;
}


/***************************************************************
* Function: ANIMLoadGeometryCreatorReference()
*
* Description: Add wireframe geodes and solid shape geodes to
* switches. Orders should be in accordance with those defined
* in 'ANIMShapeSwitchEntry'
*
***************************************************************/
void ANIMLoadGeometryCreatorReference(Switch **snapWireframeSwitch, Switch **snapSolidshapeSwitch)
{
    *snapWireframeSwitch = new Switch();
    CAVEGeodeSnapWireframeBox *snapWireframeBox = new CAVEGeodeSnapWireframeBox();
    CAVEGeodeSnapWireframeCylinder * snapWireframeCylinder = new CAVEGeodeSnapWireframeCylinder();

    (*snapWireframeSwitch)->addChild(snapWireframeBox);
    (*snapWireframeSwitch)->addChild(snapWireframeCylinder);
    (*snapWireframeSwitch)->setAllChildrenOff();

    *snapSolidshapeSwitch = new Switch();
    (*snapSolidshapeSwitch)->addChild(new CAVEGeodeSnapSolidshapeBox());
    (*snapSolidshapeSwitch)->addChild(new CAVEGeodeSnapSolidshapeCylinder());
    (*snapSolidshapeSwitch)->setAllChildrenOff();
}

};

