/***************************************************************
* Animation File Name: ANIMSketchBook.cpp
*
* Description: Load sketch book and pages objects
*
* Written by ZHANG Lelin on Nov 4, 2010
*
***************************************************************/
#include "ANIMSketchBook.h"

using namespace std;
using namespace osg;

namespace CAVEAnimationModeler
{


/***************************************************************
* Function: ANIMLoadSketchBook()
***************************************************************/
void ANIMLoadSketchBook(PositionAttitudeTransform** xformScaleFwd, PositionAttitudeTransform** xformScaleBwd,
			int &numPages, ANIMPageEntry ***pageEntryArray)
{
    *xformScaleFwd = new PositionAttitudeTransform;
    *xformScaleBwd = new PositionAttitudeTransform;

    MatrixTransform *sketchbookTrans = new MatrixTransform;
    Matrixf transMat, scaleMat;
    transMat.makeTranslate(Vec3(0, 0, ANIM_VIRTUAL_SPHERE_RADIUS));
    scaleMat.makeScale(Vec3(ANIM_PARA_PAINT_FRAME_ZOOM_FACTOR,
			    ANIM_PARA_PAINT_FRAME_ZOOM_FACTOR,
			    ANIM_PARA_PAINT_FRAME_ZOOM_FACTOR));
    sketchbookTrans->setMatrix(transMat * scaleMat);

    (*xformScaleFwd)->addChild(sketchbookTrans);
    (*xformScaleBwd)->addChild(sketchbookTrans);

    // load sketch book node from VRML file, create page geodes 
    Node* sketchbookNode = osgDB::readNodeFile(ANIMDataDir() + "VRMLFiles/SketchBook.WRL");
    sketchbookTrans->addChild(sketchbookNode);


    // Load floorplan filenames from config file
    bool isFile = true;
    int j = 0, numTex;
    std::string file = "", dir, path;
    std::vector<std::string> filenames;
    
    dir = cvr::ConfigManager::getEntry("dir", "Plugin.CaveCADBeta.Floorplans", "/home/cehughes");
    dir = dir + "/";
    path = "Plugin.CaveCADBeta.Floorplans.0";
    file = cvr::ConfigManager::getEntry(path, "", &isFile);

    while (isFile)
    {
        filenames.push_back(dir + file);
        j++;
        char buf[50];
        sprintf(buf, "Plugin.CaveCADBeta.Floorplans.%d", j);
        std::string path = std::string(buf);
        file = cvr::ConfigManager::getEntry(path, "", &isFile);
    }
    
    numPages = j;

    //numPages = 3;
    
    // create tree structured page entry array 
    *pageEntryArray = new ANIMPageEntry*[numPages];

    for (int i = 0; i < numPages; i++)
    {
/*        char idxStr[16];
        if (i < 10) 
        {
            sprintf(idxStr, "0%d", i);
        }
        else if (i < 100) 
        {
            sprintf(idxStr, "%d", i);
        }*/
        string filename = filenames[i];//ANIMDataDir() + "Textures/Floorplans/Floorplan" + string(idxStr) + string(".JPG");

        (*pageEntryArray)[i] = new ANIMPageEntry;
        Switch *singlePageSwitch = new Switch;
        PositionAttitudeTransform *flipUpTrans = new PositionAttitudeTransform;
        PositionAttitudeTransform *flipDownTrans = new PositionAttitudeTransform;

        sketchbookTrans->addChild(singlePageSwitch);
        singlePageSwitch->addChild(flipUpTrans);
        singlePageSwitch->addChild(flipDownTrans);
        singlePageSwitch->setAllChildrenOff();

        // set up flip up / flip down animation paths for each page 
        Geode *flipUpGeode, *flipDownGeode;
        AnimationPathCallback *flipUpCallback, *flipDownCallback;
        ANIMCreateSinglePageGeodeAnimation(filename, &flipUpGeode, &flipDownGeode, &flipUpCallback, &flipDownCallback);

        flipUpTrans->addChild(flipUpGeode);
        flipUpTrans->setUpdateCallback(flipUpCallback);
        flipDownTrans->addChild(flipDownGeode);
        flipDownTrans->setUpdateCallback(flipDownCallback);

        // write into page entry array record 
        (*pageEntryArray)[i]->mSwitch = singlePageSwitch;
        (*pageEntryArray)[i]->mFlipUpAnim = flipUpCallback;
        (*pageEntryArray)[i]->mFlipDownAnim = flipDownCallback;
        (*pageEntryArray)[i]->mPageGeode = flipDownGeode;
        (*pageEntryArray)[i]->mTexFilename = filename;
    }

    (*pageEntryArray)[0]->mSwitch->setSingleChildOn(1);
    (*pageEntryArray)[1]->mSwitch->setSingleChildOn(0);

    // size of floorplan 
    



    // FIX THIS - put sizes in the config file or something
    float alt = -2.9f;
    (*pageEntryArray)[0]->mLength = 32;		(*pageEntryArray)[0]->mWidth = 16;	(*pageEntryArray)[0]->mAlti = alt;
    (*pageEntryArray)[1]->mLength = 128;	(*pageEntryArray)[1]->mWidth = 128;	(*pageEntryArray)[1]->mAlti = alt;
    (*pageEntryArray)[2]->mLength = 32;		(*pageEntryArray)[2]->mWidth = 16;	(*pageEntryArray)[2]->mAlti = alt;
    (*pageEntryArray)[3]->mLength = 32;		(*pageEntryArray)[3]->mWidth = 32;	(*pageEntryArray)[3]->mAlti = alt;




    // set up the forward / backward scale animation paths for sketch book root switch 
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
* Function: ANIMCreateSinglePageGeodeAnimation()
***************************************************************/
void ANIMCreateSinglePageGeodeAnimation(const string& texfilename, Geode **flipUpGeode, Geode **flipDownGeode,
					AnimationPathCallback **flipUpCallback, AnimationPathCallback **flipDownCallback)
{
    /* coordinates of page object */
    Vec3 topleft = Vec3(-0.19, 0, 0);
    Vec3 bottomleft = Vec3(-0.19, 0, -0.28);
    Vec3 bottomright = Vec3( 0.19, 0, -0.28);
    Vec3 topright = Vec3( 0.19, 0, 0);
    Vec3 start = Vec3(0, -0.004, 0);
    Vec3 end = Vec3(0, 0.008, 0);
    float pageH = 0.28, pageW = 0.38;

    /* create page pain geometry */
    *flipUpGeode = new Geode;
    *flipDownGeode = new Geode;

    Geometry *pageGeometry = new Geometry();
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

    pageGeometry->addPrimitiveSet(rectangle);
    pageGeometry->setVertexArray(vertices);
    pageGeometry->setTexCoordArray(0, texcoords);
    pageGeometry->setNormalArray(normals);
    pageGeometry->setNormalBinding(Geometry::BIND_PER_VERTEX);

    (*flipUpGeode)->addDrawable(pageGeometry);
    (*flipDownGeode)->addDrawable(pageGeometry);

    /* apply image textures to page geodes */
    Image* imgFloorplan = osgDB::readImageFile(texfilename);
    int imgW = imgFloorplan->s();
    int imgH = imgFloorplan->t();
    Texture2D* texFloorplan = new Texture2D(imgFloorplan); 
    texFloorplan->setWrap(Texture::WRAP_S, Texture::CLAMP);
    texFloorplan->setWrap(Texture::WRAP_T, Texture::CLAMP);

    float imgRatio = (float) imgW / imgH;
    float pageRatio = pageW / pageH;
    if (imgRatio <= pageRatio)
    {
        (*texcoords)[0].set((1.0 - pageRatio / imgRatio) * 0.5, 1);
        (*texcoords)[1].set((1.0 - pageRatio / imgRatio) * 0.5, 0);
        (*texcoords)[2].set((1.0 + pageRatio / imgRatio) * 0.5, 0);
        (*texcoords)[3].set((1.0 + pageRatio / imgRatio) * 0.5, 1);
    }
    else
    {
        (*texcoords)[0].set(0, (1.0 + imgRatio / pageRatio) * 0.5);
        (*texcoords)[1].set(0, (1.0 - imgRatio / pageRatio) * 0.5);
        (*texcoords)[2].set(1, (1.0 - imgRatio / pageRatio) * 0.5);
        (*texcoords)[3].set(1, (1.0 + imgRatio / pageRatio) * 0.5);
    }

    Material *transmaterial = new Material;
    transmaterial->setDiffuse(Material::FRONT_AND_BACK, Vec4(1, 1, 1, 1));
    transmaterial->setAlpha(Material::FRONT_AND_BACK, 0.8f);

    Material *solidmaterial = new Material;
    solidmaterial->setDiffuse(Material::FRONT_AND_BACK, Vec4(1, 1, 1, 1));
    solidmaterial->setAlpha(Material::FRONT_AND_BACK, 1.0f);

    StateSet *flipUpStateSet = (*flipUpGeode)->getOrCreateStateSet();
    flipUpStateSet->setTextureAttributeAndModes(0, texFloorplan, StateAttribute::ON);
    flipUpStateSet->setMode(GL_BLEND, StateAttribute::OVERRIDE | StateAttribute::ON );
    flipUpStateSet->setRenderingHint(StateSet::TRANSPARENT_BIN);
    flipUpStateSet->setAttributeAndModes(transmaterial, StateAttribute::OVERRIDE | StateAttribute::ON);

    StateSet *flipDownStateSet = (*flipDownGeode)->getOrCreateStateSet();
    flipDownStateSet->setTextureAttributeAndModes(0, texFloorplan, StateAttribute::ON);
    flipDownStateSet->setMode(GL_BLEND, StateAttribute::OVERRIDE | StateAttribute::ON );
    flipDownStateSet->setRenderingHint(StateSet::TRANSPARENT_BIN);
    flipDownStateSet->setAttributeAndModes(solidmaterial, StateAttribute::OVERRIDE | StateAttribute::ON);

    /* create page flipping animation call backs */
    AnimationPath* animationPathFlipUp = new AnimationPath;
    AnimationPath* animationPathFlipDown = new AnimationPath;
    animationPathFlipUp->setLoopMode(AnimationPath::NO_LOOPING);
    animationPathFlipDown->setLoopMode(AnimationPath::NO_LOOPING);

    Vec3 flipUpOffset, flipDownOffset;
    Quat flipUpQuat, flipDownQuat;
    Vec3 offsetstep = (end - start) / ANIM_SKETCH_BOOK_PAGE_FLIP_SAMPS;
    float anglestep = M_PI * 2 / ANIM_SKETCH_BOOK_PAGE_FLIP_SAMPS;
    float timestep = 1.0f / ANIM_SKETCH_BOOK_PAGE_FLIP_SAMPS;
    for (int i = 0; i < ANIM_SKETCH_BOOK_PAGE_FLIP_SAMPS + 1; i++)
    {
        float val = i * timestep;
        flipUpOffset = start + offsetstep * i;
        flipDownOffset = end - offsetstep * i;
        flipUpQuat = Quat(i * anglestep, Vec3(-1, 0, 0));
        flipDownQuat = Quat(i * anglestep, Vec3(1, 0, 0));
        animationPathFlipUp->insert(val, AnimationPath::ControlPoint(flipUpOffset, flipUpQuat, Vec3(1, 1, 1)));
        animationPathFlipDown->insert(val, AnimationPath::ControlPoint(flipDownOffset, flipDownQuat, Vec3(1, 1, 1)));
    }

    *flipUpCallback = new AnimationPathCallback(animationPathFlipUp, 0.0, 1.0f / ANIM_SKETCH_BOOK_PAGE_FLIP_TIME);
    *flipDownCallback = new AnimationPathCallback(animationPathFlipDown, 0.0, 1.0f / ANIM_SKETCH_BOOK_PAGE_FLIP_TIME);
}


};

