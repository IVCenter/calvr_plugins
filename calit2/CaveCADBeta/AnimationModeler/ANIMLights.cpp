/***************************************************************
* Animation File Name: ANIMViewpoints.cpp
*
* Description: Create animated model of virtual sphere
*
* Written by ZHANG Lelin on Sep 15, 2010
*
***************************************************************/
#include "ANIMLights.h"
#include <osgText/Text3D>

using namespace std;
using namespace osg;

namespace CAVEAnimationModeler
{


/***************************************************************
* Function: ANIMCreateViewpoints()
*
***************************************************************/
void ANIMCreateLights(std::vector<osg::PositionAttitudeTransform*>* fwdVec,
                            std::vector<osg::PositionAttitudeTransform*>* bwdVec)
{
    Geode* sphereGeode = new Geode();
    Sphere* virtualSphere = new Sphere();
    ShapeDrawable* sphereDrawable = new ShapeDrawable(virtualSphere);

    virtualSphere->setRadius(ANIM_VIRTUAL_SPHERE_RADIUS);

    StateSet* stateset = new StateSet();
    stateset->setMode(GL_BLEND, StateAttribute::OVERRIDE | StateAttribute::ON);
    stateset->setMode(GL_CULL_FACE, StateAttribute::OVERRIDE | StateAttribute::ON);
    stateset->setRenderingHint(StateSet::TRANSPARENT_BIN);
    sphereGeode->setStateSet(stateset);

    osgDB::ReaderWriter::Options *options = new osgDB::ReaderWriter::Options();
    options->setObjectCacheHint(osgDB::ReaderWriter::Options::CACHE_NONE);

    osg::Node *node;
    osg::MatrixTransform *objScale;
    int numItems = 2;
    for (int i = 0; i < numItems; ++i)
    {
        osg::Vec3 startPos(1.0, 0, 0);
        osg::Vec3 pos(1.0, 0, -i*0.5);

        virtualSphere = new osg::Sphere(osg::Vec3(), ANIM_VIRTUAL_SPHERE_RADIUS);
        sphereDrawable = new osg::ShapeDrawable(virtualSphere);

        osg::Vec4 color;
        if (i == 0) // root menu item geometry
        {
            color = osg::Vec4(0.2, 0.2, 1, 0.5);
            node = NULL; 
            objScale = NULL;
            
            objScale = new osg::MatrixTransform();
            node = new osg::Node();

            osg::Cone * cone = new osg::Cone(osg::Vec3(), 0.2, 0.2);  
            osg::ShapeDrawable * sd = new osg::ShapeDrawable(cone);
            sd->setColor(color);
            osg::Geode * geode = new osg::Geode();
            geode->addDrawable(sd);
            objScale->addChild(geode);
        }
        else // child menu item geometries
        {
            color = osg::Vec4(0.2, 0.2, 1, 0.5);

            node = new osg::Node();
            objScale = new osg::MatrixTransform();
            
            osg::Box * box = new osg::Box(osg::Vec3(), 0.2);
            osg::ShapeDrawable * sd = new osg::ShapeDrawable(box);
            sd->setColor(color);
            osg::Geode * geode = new osg::Geode();
            geode->addDrawable(sd);
            objScale->addChild(geode);
        }

        stateset = sphereDrawable->getOrCreateStateSet();
        stateset->setMode(GL_BLEND, StateAttribute::ON);
        stateset->setMode(GL_CULL_FACE, StateAttribute::ON);
        stateset->setRenderingHint(StateSet::TRANSPARENT_BIN);

        sphereDrawable->setColor(color);
        sphereGeode = new osg::Geode();
        sphereGeode->addDrawable(sphereDrawable);
 
        AnimationPath* animationPathScaleFwd = new AnimationPath;
        AnimationPath* animationPathScaleBwd = new AnimationPath;
        animationPathScaleFwd->setLoopMode(AnimationPath::NO_LOOPING);
        animationPathScaleBwd->setLoopMode(AnimationPath::NO_LOOPING);

        Vec3 scaleFwd, scaleBwd;
        float step = 1.f / ANIM_VIRTUAL_SPHERE_NUM_SAMPS;
        for (int j = 0; j < ANIM_VIRTUAL_SPHERE_NUM_SAMPS + 1; j++)
        {
            float val = j * step;
            scaleFwd = Vec3(val, val, val);
            scaleBwd = Vec3(1-val, 1-val, 1-val);

            osg::Vec3 diff = startPos - pos;
            osg::Vec3 fwdVec, bwdVec;

            for (int i = 0; i < 3; ++i)
                diff[i] *= val;

            fwdVec = startPos - diff;
            bwdVec = pos + diff;

            animationPathScaleFwd->insert(val, AnimationPath::ControlPoint(fwdVec, Quat(), scaleFwd));
            animationPathScaleBwd->insert(val, AnimationPath::ControlPoint(bwdVec, Quat(), scaleBwd));
        }

        AnimationPathCallback *animCallbackFwd = new AnimationPathCallback(animationPathScaleFwd, 
                            0.0, 1.f / ANIM_VIRTUAL_SPHERE_LAPSE_TIME);
        AnimationPathCallback *animCallbackBwd = new AnimationPathCallback(animationPathScaleBwd, 
                            0.0, 1.f / ANIM_VIRTUAL_SPHERE_LAPSE_TIME); 
        
        osg::PositionAttitudeTransform *fwd, *bwd;
        fwd = new osg::PositionAttitudeTransform();
        bwd = new osg::PositionAttitudeTransform();
    
        fwd->addChild(sphereGeode);
        bwd->addChild(sphereGeode);

        if (objScale && node)
        {
            fwd->addChild(objScale);
            bwd->addChild(objScale);
        }

        fwd->setUpdateCallback(animCallbackFwd);
        bwd->setUpdateCallback(animCallbackBwd);

        fwdVec->push_back(fwd);
        bwdVec->push_back(bwd);
    }
}

};

