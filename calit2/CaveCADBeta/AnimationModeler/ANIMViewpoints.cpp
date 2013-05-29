/***************************************************************
* Animation File Name: ANIMViewpoints.cpp
*
* Description: Create animated model of virtual sphere
*
* Written by ZHANG Lelin on Sep 15, 2010
*
***************************************************************/
#include "ANIMViewpoints.h"
#include <osgText/Text3D>

using namespace std;
using namespace osg;

namespace CAVEAnimationModeler
{


/***************************************************************
* Function: ANIMCreateViewpoints()
*
***************************************************************/
void ANIMCreateViewpoints(std::vector<osg::PositionAttitudeTransform*>* fwdVec, 
                          std::vector<osg::PositionAttitudeTransform*>* bwdVec)
{
    Geode* sphereGeode = new Geode();
    Sphere* virtualSphere = new Sphere();
    ShapeDrawable* sphereDrawable = new ShapeDrawable(virtualSphere);
    
    virtualSphere->setRadius(ANIM_VIRTUAL_SPHERE_RADIUS);

    // apply shaders to geode stateset 
    sphereDrawable->setColor(osg::Vec4(1, 0, 0, 0.5));

    StateSet* stateset = new StateSet();
    stateset->setMode(GL_BLEND, StateAttribute::OVERRIDE | StateAttribute::ON);
    stateset->setMode(GL_CULL_FACE, StateAttribute::OVERRIDE | StateAttribute::ON);
    stateset->setRenderingHint(StateSet::TRANSPARENT_BIN);
    sphereGeode->setStateSet(stateset);
    
    int numViews = 2;
    for (int i = 0; i < numViews; ++i)
    {
        osg::Vec3 startPos(0, 0, 0);   
        osg::Vec3 pos(0, 0, -i*0.5);

        virtualSphere = new osg::Sphere(osg::Vec3(), ANIM_VIRTUAL_SPHERE_RADIUS);
        sphereDrawable = new osg::ShapeDrawable(virtualSphere);

        osg::Vec4 color;
        std::string str;
        if (i == 0)
        {
            color = osg::Vec4(0, 0, 1, 0.5);
            str = "Views";
        }
        else if (i == numViews - 1)
        {
            color = osg::Vec4(1, 0, 0, 0.5);
            str = "Save";
        }
        else
        {
            char buf[10];
            sprintf(buf, "%d\0", i);
            str = std::string(buf);
        }

        stateset = sphereDrawable->getOrCreateStateSet();
        stateset->setMode(GL_BLEND, StateAttribute::ON);
        stateset->setMode(GL_CULL_FACE, StateAttribute::ON);
        stateset->setRenderingHint(StateSet::TRANSPARENT_BIN);

        sphereDrawable->setColor(color);
        sphereGeode = new osg::Geode();
        //sphereGeode->addDrawable(sphereDrawable);

        osg::ref_ptr<osgText::Text> text = new osgText::Text();
        text->setText(str);
        text->setCharacterSize(0.1);
        text->setDrawMode(osgText::Text::TEXT);
        text->setAxisAlignment(osgText::Text::XZ_PLANE);
        text->setPosition(osg::Vec3(-0.1, -0.4, -0.1));
        text->setColor(osg::Vec4(1,1,1,1));
        text->setFont(cvr::CalVR::instance()->getResourceDir() + "/resources/arial.ttf");

        sphereGeode->addDrawable(text);

 
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

            animationPathScaleFwd->insert(val, AnimationPath::ControlPoint(fwdVec, Quat(), scaleFwd));//pos, Quat(), scaleFwd));
            animationPathScaleBwd->insert(val, AnimationPath::ControlPoint(bwdVec, Quat(), scaleBwd));
        }

        AnimationPathCallback *animCallbackFwd = new AnimationPathCallback(animationPathScaleFwd, 
                            0.0, 1.f / ANIM_VIRTUAL_SPHERE_LAPSE_TIME);
        AnimationPathCallback *animCallbackBwd = new AnimationPathCallback(animationPathScaleBwd, 
                            0.0, 1.f / ANIM_VIRTUAL_SPHERE_LAPSE_TIME); 
        
        osg::PositionAttitudeTransform *fwd, *bwd;
        fwd = new osg::PositionAttitudeTransform();
        bwd = new osg::PositionAttitudeTransform();

        Node* frameNode = osgDB::readNodeFile(ANIMDataDir() + "VRMLFiles/ParamountFrame.WRL"); 
        fwd->addChild(frameNode);
        bwd->addChild(frameNode);

        fwd->addChild(sphereGeode);
        bwd->addChild(sphereGeode);     

        fwd->setUpdateCallback(animCallbackFwd);
        bwd->setUpdateCallback(animCallbackBwd);

        fwdVec->push_back(fwd);
        bwdVec->push_back(bwd);
        
     
        //(*xformScaleFwd)->addChild(sphereGeode);
        //(*xformScaleBwd)->addChild(sphereGeode);
    }

/*
    Program* shaderProg = new Program;
    stateset->setAttribute(shaderProg);
    shaderProg->addShader(Shader::readShaderFile(Shader::VERTEX, ANIMDataDir() + "Shaders/VirtualSphere.vert"));
    shaderProg->addShader(Shader::readShaderFile(Shader::FRAGMENT, ANIMDataDir() + "Shaders/VirtualSphere.frag"));

    Image* envMap = osgDB::readImageFile(ANIMDataDir() + "Textures/EnvMap.JPG");
    Texture2D* envTex = new Texture2D(envMap);    
    stateset->setTextureAttributeAndModes(0, envTex, StateAttribute::ON);

    Uniform* envMapSampler = new Uniform("EnvMap", 0);
    stateset->addUniform(envMapSampler);

    Uniform* baseColorUniform = new Uniform("BaseColor", Vec3(0.2, 1.0, 0.2));
    stateset->addUniform(baseColorUniform);

    Uniform* lightPosUniform = new Uniform("LightPos", Vec4(1.0, 0.0, 0.2, 0.0));
    stateset->addUniform(lightPosUniform);
*/
}


void ANIMAddViewpoint(std::vector<osg::PositionAttitudeTransform*>* fwdVec, 
                      std::vector<osg::PositionAttitudeTransform*>* bwdVec)
{
    Geode* sphereGeode;// = new Geode();
    Sphere* virtualSphere;// = new Sphere();
    ShapeDrawable* sphereDrawable;// = new ShapeDrawable(virtualSphere);
    //virtualSphere->setRadius(ANIM_VIRTUAL_SPHERE_RADIUS);

    // apply shaders to geode stateset 
    //sphereDrawable->setColor(osg::Vec4(1, 0, 0, 0.5));

    //stateset->setMode(GL_BLEND, StateAttribute::OVERRIDE | StateAttribute::ON);
    //stateset->setMode(GL_CULL_FACE, StateAttribute::OVERRIDE | StateAttribute::ON);
    //stateset->setRenderingHint(StateSet::TRANSPARENT_BIN);
    //sphereGeode->setStateSet(stateset);

    int numViews = fwdVec->size();
    //std::vector<osg::PositionAttitudeTransform*>::iterator it = fwdVec->begin();
    fwdVec->clear();
    bwdVec->clear();

    for (int i = 0; i < numViews + 1; ++i)
    {
        osg::Vec3 startPos(0, 0, 0);
        osg::Vec3 pos(0, 0, -i*0.5);

        virtualSphere = new osg::Sphere(osg::Vec3(), ANIM_VIRTUAL_SPHERE_RADIUS);
        sphereDrawable = new osg::ShapeDrawable(virtualSphere);

        osg::Vec4 color;
        std::string str;
        if (i == 0)
        {
            color = osg::Vec4(0, 0, 1, 0.5); // blue
            str = "Views";
        }
        else if (i == numViews)
        {
            color = osg::Vec4(1, 0, 0, 0.5); // red
            str = "Save";
        }
        else
        {
            color = osg::Vec4(1, 1, 0, 0.5); // yellow
            char buf[10];
            sprintf(buf, "%d\0", i);
            str = std::string(buf);
        }

        StateSet* stateset;        
        stateset = sphereDrawable->getOrCreateStateSet();
        stateset->setMode(GL_BLEND, StateAttribute::ON);
        stateset->setMode(GL_CULL_FACE, StateAttribute::ON);
        stateset->setMode(GL_LIGHTING, osg::StateAttribute::ON);
        stateset->setRenderingHint(StateSet::TRANSPARENT_BIN);

        sphereDrawable->setColor(color);
        sphereGeode = new osg::Geode();
        //sphereGeode->addDrawable(sphereDrawable);

        osg::ref_ptr<osgText::Text> text = new osgText::Text();
        text->setText(str);
        text->setCharacterSize(0.1);
        text->setDrawMode(osgText::Text::TEXT);
        text->setAxisAlignment(osgText::Text::XZ_PLANE);
        text->setPosition(osg::Vec3(-0.1, -0.4, -0.1));
        text->setColor(osg::Vec4(1,1,1,1));

        text->setFont(cvr::CalVR::instance()->getResourceDir() + "/resources/arial.ttf");
        if (i != 0 && i != numViews)
            text->setPosition(osg::Vec3(-0.03, -0.4, -0.1));
        sphereGeode->addDrawable(text);

       
/*        if (i != 0 && i != numViews)
        {
            osgText::Text3D * textNode = new osgText::Text3D();
            char buf[2];
            sprintf(buf, "%d", i);
            textNode->setText(std::string(buf));

            textNode->setCharacterSize(30);
            textNode->setCharacterDepth(15);
            textNode->setDrawMode(osgText::Text3D::TEXT);
            //textNode->setAlignment(osgText::Text3D::CENTER_CENTER);
            textNode->setPosition(osg::Vec3(0,0,0));
            textNode->setColor(osg::Vec4(1,1,1,1));
            textNode->setAxisAlignment(osgText::Text3D::XZ_PLANE);
            //textNode->setMaximumWidth(1000);
            //textNode->getOrCreateStateSet()->setRenderingHint(StateAttribute::PROTECTED | osg::StateSet::OPAQUE_BIN);
            //textNode->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::ON);

            osg::StateSet *ss = textNode->getOrCreateStateSet();
            ss->setRenderingHint(osg::StateSet::OPAQUE_BIN);
            //ss->setMode(GL_LIGHTING, osg::StateAttribute::ON);
            //ss->setMode(GL_BLEND, osg::StateAttribute::ON);
            //ss->setMode(GL_LIGHTING, osg::StateAttribute::PROTECTED | osg::StateAttribute::ON);
            //ss->setMode(GL_DEPTH_TEST, osg::StateAttribute::PROTECTED | osg::StateAttribute::OFF);
            //ss->setRenderBinDetails(0, "Render Bin");

            //sphereGeode->addDrawable(textNode);
        }*/

        AnimationPath* animationPathScaleFwd = new AnimationPath();
        AnimationPath* animationPathScaleBwd = new AnimationPath();
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
            osg::Vec3 fwd, bwd;
 
            for (int i = 0; i < 3; ++i)
                diff[i] *= val;

            fwd = startPos - diff;
            bwd = pos + diff;

            animationPathScaleFwd->insert(val, AnimationPath::ControlPoint(fwd, Quat(), scaleFwd));
            animationPathScaleBwd->insert(val, AnimationPath::ControlPoint(bwd, Quat(), scaleBwd));
        }

        AnimationPathCallback *animCallbackFwd = new AnimationPathCallback(animationPathScaleFwd, 
                            0.0, 1.f / ANIM_VIRTUAL_SPHERE_LAPSE_TIME);
        AnimationPathCallback *animCallbackBwd = new AnimationPathCallback(animationPathScaleBwd, 
                            0.0, 1.f / ANIM_VIRTUAL_SPHERE_LAPSE_TIME); 
 
        osg::PositionAttitudeTransform *fwd, *bwd;
        fwd = new osg::PositionAttitudeTransform();
        bwd = new osg::PositionAttitudeTransform();

        Node* frameNode = osgDB::readNodeFile(ANIMDataDir() + "VRMLFiles/ParamountFrame.WRL"); 
        fwd->addChild(frameNode);
        bwd->addChild(frameNode);

        fwd->addChild(sphereGeode);
        bwd->addChild(sphereGeode);

        fwd->setUpdateCallback(animCallbackFwd);
        bwd->setUpdateCallback(animCallbackBwd);

        fwdVec->push_back(fwd);
        bwdVec->push_back(bwd);
    }

/*
    Program* shaderProg = new Program;
    stateset->setAttribute(shaderProg);
    shaderProg->addShader(Shader::readShaderFile(Shader::VERTEX, ANIMDataDir() + "Shaders/VirtualSphere.vert"));
    shaderProg->addShader(Shader::readShaderFile(Shader::FRAGMENT, ANIMDataDir() + "Shaders/VirtualSphere.frag"));

    Image* envMap = osgDB::readImageFile(ANIMDataDir() + "Textures/EnvMap.JPG");
    Texture2D* envTex = new Texture2D(envMap);    
    stateset->setTextureAttributeAndModes(0, envTex, StateAttribute::ON);

    Uniform* envMapSampler = new Uniform("EnvMap", 0);
    stateset->addUniform(envMapSampler);

    Uniform* baseColorUniform = new Uniform("BaseColor", Vec3(0.2, 1.0, 0.2));
    stateset->addUniform(baseColorUniform);

    Uniform* lightPosUniform = new Uniform("LightPos", Vec4(1.0, 0.0, 0.2, 0.0));
    stateset->addUniform(lightPosUniform);
*/
}


/*void ANIMLoadViewpoints(Switch **selectStateSwitch, Switch **alphaTurnerSwitch,
				    int &numViews, ANIMTexturePalletteSelectEntry ***textureStatesEntryArray)
{
    *selectStateSwitch = new Switch;
    *alphaTurnerSwitch = new Switch;

    (*selectStateSwitch)->setAllChildrenOn();
    (*alphaTurnerSwitch)->setAllChildrenOn();

    // load alpha turner nodes for 'APPLY_TEXTURE' state 
    MatrixTransform *horizontalTrans = new MatrixTransform;
    MatrixTransform *verticalTrans = new MatrixTransform;

    Node* alphaTurnerNode = osgDB::readNodeFile(ANIMDataDir() + "VRMLFiles/AlphaTurner.WRL");

    horizontalTrans->addChild(alphaTurnerNode);
    verticalTrans->addChild(alphaTurnerNode);

    (*alphaTurnerSwitch)->addChild(horizontalTrans);
    (*alphaTurnerSwitch)->addChild(verticalTrans);

    std::vector<osg::Vec3> colors;
    colors.push_back(osg::Vec3(1, 0, 0));
    colors.push_back(osg::Vec3(1, 0.5, 0));
    colors.push_back(osg::Vec3(1, 1, 0));
    colors.push_back(osg::Vec3(0, 1, 0));
    colors.push_back(osg::Vec3(0, 1, 1));
    colors.push_back(osg::Vec3(0, 0, 1));
    
    osg::ShapeDrawable *sd;
    osg::Sphere *sphere;
    osg::Geode *geode;

    for (int i = 0; i < 3; ++i)
    {
        sphere = new osg::Sphere(osg::Vec3(0, 0, -i), 0.5);
        sd = new osg::ShapeDrawable(sphere); 
        sd->setColor(colors[i]);
        geode->addDrawable(sd);

        (*selectStateSwitch)->addChild((*textureStatesEntryArray)[i]->mEntrySwitch);
    }

}*/

};

