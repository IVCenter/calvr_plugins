/***************************************************************
* Animation File Name: ANIMParamountPaintFrames.cpp
*
* Description: Load paramount paint frames switch objects
*
* Written by ZHANG Lelin on Nov 2, 2010
*
***************************************************************/
#include "ANIMParamountPaintFrames.h"

using namespace std;
using namespace osg;

namespace CAVEAnimationModeler
{


/***************************************************************
* Function: ANIMLoadParamountPaintFrames()
***************************************************************/
void ANIMLoadParamountPaintFrames(PositionAttitudeTransform** xformScaleFwd, PositionAttitudeTransform** xformScaleBwd,
				int &numParas, float &paraswitchRadius, ANIMParamountSwitchEntry ***paraEntryArray)
{
    *xformScaleFwd = new PositionAttitudeTransform;
    *xformScaleBwd = new PositionAttitudeTransform;

    // Load panorama filenames from config file
    bool isFile = true;
    int i = 0;
    std::string filename = "", dir, path, thumb;
    std::vector<std::string> filenames, thumbFiles;
    
    dir = cvr::ConfigManager::getEntry("dir", "Plugin.CaveCADBeta.Panoramas", "/home/cehughes");
    dir = dir + "/";
    path = "Plugin.CaveCADBeta.Panoramas.0";
    filename = cvr::ConfigManager::getEntry(path, "", &isFile);
    thumb = cvr::ConfigManager::getEntry("thumb", path, "");

    while (isFile)
    {
        filenames.push_back(dir + filename);
        thumbFiles.push_back(dir + thumb);

        i++;
        char buf[50];
        sprintf(buf, "Plugin.CaveCADBeta.Panoramas.%d", i);
        std::string path = std::string(buf);
        filename = cvr::ConfigManager::getEntry(path, "", &isFile);
        thumb = cvr::ConfigManager::getEntry("thumb", path, "");
    }

    numParas = i;

    // create tree structured paramount switch entry array 
    *paraEntryArray = new ANIMParamountSwitchEntry*[numParas];
    for (int i = 0; i < numParas; i++)
    {
        (*paraEntryArray)[i] = new ANIMParamountSwitchEntry;
        MatrixTransform *matTrans = new MatrixTransform;
        (*xformScaleFwd)->addChild(matTrans);
        (*xformScaleBwd)->addChild(matTrans);

        (*paraEntryArray)[i]->mMatrixTrans = matTrans;
    }

    // set initial position of paint frame arrays 
    Matrixd transMat;
    float intvlAngle = M_PI * 2 / numParas;

    // load frame node from VRML file 
    Node* frameNode = osgDB::readNodeFile(ANIMDataDir() + "VRMLFiles/ParamountFrame.WRL"); 
    paraswitchRadius = frameNode->getBound().radius() * numParas * 0.5 / M_PI;
  
    for (int i = 0; i < numParas; i++)
    {
        Geode* paintGeode = ANIMCreateParamountPaintGeode(thumbFiles[i]);

        float phi = i * intvlAngle;
        Vec3 transVec = Vec3(0, -cos(phi) * paraswitchRadius, sin(phi) * paraswitchRadius);
        transMat.makeTranslate(transVec);
        (*paraEntryArray)[i]->mMatrixTrans->setMatrix(transMat);

        Switch *singleParaSwitch = new Switch;
        PositionAttitudeTransform *zoominTrans = new PositionAttitudeTransform;
        PositionAttitudeTransform *zoomoutTrans = new PositionAttitudeTransform; 

        zoominTrans->addChild(frameNode);	zoomoutTrans->addChild(frameNode);
        zoominTrans->addChild(paintGeode);	zoomoutTrans->addChild(paintGeode);
        singleParaSwitch->addChild(zoominTrans);
        singleParaSwitch->addChild(zoomoutTrans);
        (*paraEntryArray)[i]->mMatrixTrans->addChild(singleParaSwitch);
        if (i == 0) 
            singleParaSwitch->setSingleChildOn(0);
        else 
            singleParaSwitch->setSingleChildOn(1);

        /* set up zoom in / zoom out animation paths for each paramount paint & frame */
        AnimationPathCallback *zoomInCallback, *zoomOutCallback;
        ANIMCreateParamountPaintFrameAnimation(&zoomInCallback, &zoomOutCallback);
        zoominTrans->setUpdateCallback(zoomInCallback);
        zoomoutTrans->setUpdateCallback(zoomOutCallback);

        /* write into paramount switch entry array record */
        (*paraEntryArray)[i]->mSwitch = singleParaSwitch;
        (*paraEntryArray)[i]->mZoomInAnim = zoomInCallback;
        (*paraEntryArray)[i]->mZoomOutAnim = zoomOutCallback;
        (*paraEntryArray)[i]->mPaintGeode = paintGeode;
        (*paraEntryArray)[i]->mTexFilename = filenames[i];//ANIMDataDir() + string("Textures/Paramounts/Paramount") 
                                   //+ filenames[i];//+ string(idxStr) + string(".JPG");
    }

    /* set up the forward / backward scale animation paths for paramount root switch */
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
        scaleBwd = Vec3(1-val, 1-val, 1-val);
        animationPathScaleFwd->insert(val, AnimationPath::ControlPoint(Vec3(),Quat(), scaleFwd));
        animationPathScaleBwd->insert(val, AnimationPath::ControlPoint(Vec3(),Quat(), scaleBwd));
    }

    AnimationPathCallback *animCallbackFwd = new AnimationPathCallback(animationPathScaleFwd, 
						0.0, 1.f / ANIM_VIRTUAL_SPHERE_LAPSE_TIME);
    AnimationPathCallback *animCallbackBwd = new AnimationPathCallback(animationPathScaleBwd, 
						0.0, 1.f / ANIM_VIRTUAL_SPHERE_LAPSE_TIME); 
    (*xformScaleFwd)->setUpdateCallback(animCallbackFwd);
    (*xformScaleBwd)->setUpdateCallback(animCallbackBwd);
}


/***************************************************************
* Function: ANIMCreateParamountPaintFrameAnimation()
***************************************************************/
void ANIMCreateParamountPaintFrameAnimation(AnimationPathCallback **zoomInCallback,
					    AnimationPathCallback **zoomOutCallback)
{
    /* set up zoom in / zoom out animation paths for separate paramount paint & frame */
    AnimationPath* animationPathZoomIn = new AnimationPath;
    AnimationPath* animationPathZoomOut = new AnimationPath;
    animationPathZoomIn->setLoopMode(AnimationPath::NO_LOOPING);
    animationPathZoomOut->setLoopMode(AnimationPath::NO_LOOPING);
   
    Vec3 zoomInVect, zoomOutVect;
    float zoomstep = (ANIM_PARA_PAINT_FRAME_ZOOM_FACTOR - 1.0f) / ANIM_PARA_PAINT_FRAME_ZOOM_SAMPS;
    float timestep = 1.0f / ANIM_PARA_PAINT_FRAME_ZOOM_SAMPS;
    for (int i = 0; i < ANIM_PARA_PAINT_FRAME_ZOOM_SAMPS + 1; i++)
    {
        float val = i * zoomstep;
        zoomInVect = Vec3(val + 1.0f, val + 1.0f, val + 1.0f);
        zoomOutVect = Vec3(ANIM_PARA_PAINT_FRAME_ZOOM_FACTOR - val, 
                ANIM_PARA_PAINT_FRAME_ZOOM_FACTOR - val, 
                ANIM_PARA_PAINT_FRAME_ZOOM_FACTOR - val);
        animationPathZoomIn->insert(i * timestep, AnimationPath::ControlPoint(Vec3(0,0,0), Quat(), zoomInVect));
        animationPathZoomOut->insert(i * timestep, AnimationPath::ControlPoint(Vec3(0,0,0), Quat(), zoomOutVect));
    }
    *zoomInCallback = new AnimationPathCallback(animationPathZoomIn, 
						0.0, 1.f / ANIM_PARA_PAINT_FRAME_LAPSE_TIME);
    *zoomOutCallback = new AnimationPathCallback(animationPathZoomOut, 
						0.0, 1.f / ANIM_PARA_PAINT_FRAME_LAPSE_TIME);
}


/***************************************************************
* Function: ANIMCreateParamountPaintGeode()
***************************************************************/
Geode *ANIMCreateParamountPaintGeode(const string &texfilename)
{
    /* coordinates of page object */
    Vec3 topleft = Vec3(-0.176, -0.044, 0.126);
    Vec3 bottomleft = Vec3(-0.176, -0.044, -0.126);
    Vec3 bottomright = Vec3(0.176, -0.044, -0.126);
    Vec3 topright = Vec3(0.176, -0.044, 0.126);

    /* create page pain geometry */
    Geode *paintGeode = new Geode;
    Geometry *paintGeometry = new Geometry();
    Vec3Array* vertices = new Vec3Array;
    Vec2Array* texcoords = new Vec2Array(4);
    Vec3Array* normals = new Vec3Array;

    vertices->push_back(topleft);	(*texcoords)[0].set(0, 1);
    vertices->push_back(bottomleft);	(*texcoords)[1].set(0, 0);
    vertices->push_back(bottomright);	(*texcoords)[2].set(1, 0);
    vertices->push_back(topright);	(*texcoords)[3].set(1, 1);
    
    for (int i = 0; i < 4; i++) 
    {
        normals->push_back(Vec3(0, -1, 0));
    }

    DrawElementsUInt* rectangle = new DrawElementsUInt(PrimitiveSet::POLYGON, 0);
    rectangle->push_back(0);	rectangle->push_back(1);
    rectangle->push_back(2);	rectangle->push_back(3);

    paintGeometry->addPrimitiveSet(rectangle);
    paintGeometry->setVertexArray(vertices);
    paintGeometry->setTexCoordArray(0, texcoords);
    paintGeometry->setNormalArray(normals);
    paintGeometry->setNormalBinding(Geometry::BIND_PER_VERTEX);
    paintGeode->addDrawable(paintGeometry);

    /* apply image textures to paint geode */
    Image* imgPaintPreview = osgDB::readImageFile(texfilename);
    Texture2D* texPaintPreview = new Texture2D(imgPaintPreview); 

    Material* material = new Material;
    material->setDiffuse(Material::FRONT_AND_BACK, Vec4(1.0f, 1.0f, 1.0f, 1.0f));
    material->setAlpha(Material::FRONT_AND_BACK, 1.0f);

    StateSet *paintStateSet = paintGeode->getOrCreateStateSet();
    paintStateSet->setTextureAttributeAndModes(0, texPaintPreview, StateAttribute::ON);
    paintStateSet->setAttributeAndModes(material, StateAttribute::OVERRIDE | StateAttribute::ON);
    paintStateSet->setMode(GL_BLEND, StateAttribute::OVERRIDE | StateAttribute::ON );
    paintStateSet->setRenderingHint(StateSet::TRANSPARENT_BIN);

    return paintGeode;
}


};

