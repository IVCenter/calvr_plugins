/***************************************************************
* Animation File Name: ANIMTexturePallette.cpp
*
* Description: Create objects associated with DSTexturePallette
*
* Written by ZHANG Lelin on Jan 12, 2011
*
***************************************************************/
#include "ANIMTexturePallette.h"

using namespace std;
using namespace osg;

namespace CAVEAnimationModeler
{


/***************************************************************
* Function: ANIMLoadTexturePalletteRoot()
*
* xformScaleFwd: Animation transform for inflating geometries
* xformScaleBwd: Animation transform for shrinking geometries
*
***************************************************************/
void ANIMLoadTexturePalletteRoot(osg::PositionAttitudeTransform** xformScaleFwd,
                                 osg::PositionAttitudeTransform** xformScaleBwd,
                                 osg::PositionAttitudeTransform** buttonScaleFwd,
                                 osg::PositionAttitudeTransform** buttonScaleBwd)
{
    *xformScaleFwd = new PositionAttitudeTransform;
    *xformScaleBwd = new PositionAttitudeTransform;

    // create zoom in & out animations for 'DSTexturePallette' root switch
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
        animationPathScaleFwd->insert(val, AnimationPath::ControlPoint(Vec3(-0.5, 0, 0),Quat(), scaleFwd));
        animationPathScaleBwd->insert(val, AnimationPath::ControlPoint(Vec3(-0.5, 0, 0),Quat(), scaleBwd));
    }

    AnimationPathCallback *animCallbackFwd = new AnimationPathCallback(animationPathScaleFwd, 
						0.0, 1.f / ANIM_VIRTUAL_SPHERE_LAPSE_TIME);
    AnimationPathCallback *animCallbackBwd = new AnimationPathCallback(animationPathScaleBwd, 
						0.0, 1.f / ANIM_VIRTUAL_SPHERE_LAPSE_TIME); 
    (*xformScaleFwd)->setUpdateCallback(animCallbackFwd);
    (*xformScaleBwd)->setUpdateCallback(animCallbackBwd);


    // create Save Color button
    osg::Geode *saveGeode = new osg::Geode();
    osg::Shape *saveSphere = new osg::Sphere(osg::Vec3(0, 0, 0), 0.20);
    osg::ShapeDrawable *saveSD = new osg::ShapeDrawable(saveSphere);
    saveSD->setColor(osg::Vec4(1.0, 1.0, 1.0, 0.5));
    saveGeode->addDrawable(saveSD);
    
    animationPathScaleFwd = new AnimationPath();
    animationPathScaleBwd = new AnimationPath();

    animationPathScaleFwd->setLoopMode(AnimationPath::NO_LOOPING);
    animationPathScaleBwd->setLoopMode(AnimationPath::NO_LOOPING);

    step = 1.f / ANIM_VIRTUAL_SPHERE_NUM_SAMPS;
    osg::Vec3 startPos, endPos;
    startPos = osg::Vec3(-0.5, 0, 0);
    endPos = osg::Vec3(-0.5, 0, -2.0);
    for (int i = 0; i < ANIM_VIRTUAL_SPHERE_NUM_SAMPS + 1; i++)
    {
        float val = i * step;
        scaleFwd = Vec3(val, val, val);
        scaleBwd = Vec3(1.f-val, 1.f-val, 1.f-val);

        osg::Vec3 diff, fwd, bwd;
        diff = osg::Vec3(endPos - startPos);

        for (int j = 0; j < 3; ++j)
        {
            fwd[j] = startPos[j] + val * diff[j];
            bwd[j] = startPos[j] + (1 - val) * diff[j];
        }

        animationPathScaleFwd->insert(val, AnimationPath::ControlPoint(fwd, Quat(), scaleFwd));
        animationPathScaleBwd->insert(val, AnimationPath::ControlPoint(bwd, Quat(), scaleBwd));
    }

    animCallbackFwd = new AnimationPathCallback(animationPathScaleFwd, 0.0, 1.f / ANIM_VIRTUAL_SPHERE_LAPSE_TIME);
    animCallbackBwd = new AnimationPathCallback(animationPathScaleBwd, 0.0, 1.f / ANIM_VIRTUAL_SPHERE_LAPSE_TIME); 
    
    (*buttonScaleFwd) = new osg::PositionAttitudeTransform();
    (*buttonScaleBwd) = new osg::PositionAttitudeTransform();

    (*buttonScaleFwd)->setUpdateCallback(animCallbackFwd);
    (*buttonScaleBwd)->setUpdateCallback(animCallbackBwd);

    (*buttonScaleFwd)->addChild(saveGeode);
    (*buttonScaleBwd)->addChild(saveGeode);
}


/***************************************************************
* Function: ANIMLoadTexturePalletteIdle()
***************************************************************/
void ANIMLoadTexturePalletteIdle(osg::Switch **idleStateSwitch, ANIMTexturePalletteIdleEntry **textureStateIdleEntry)
{
    *idleStateSwitch = new Switch;
    (*idleStateSwitch)->setAllChildrenOn();

    *textureStateIdleEntry = new ANIMTexturePalletteIdleEntry;
    (*textureStateIdleEntry)->mEntrySwitch = new Switch;

    // create drawables, geodes and texture mapping
    (*textureStateIdleEntry)->mEntryGeode = new Geode();
    Sphere *idleSphere = new Sphere(Vec3(0, 0, 0), ANIM_VIRTUAL_SPHERE_RADIUS);
    ShapeDrawable *idleSphereDrawable = new ShapeDrawable(idleSphere);
    ((*textureStateIdleEntry)->mEntryGeode)->addDrawable(idleSphereDrawable);

    Material *material = new Material;
    material->setDiffuse(Material::FRONT_AND_BACK, Vec4(1, 1, 1, 1));
    material->setSpecular(Material::FRONT_AND_BACK, Vec4(1, 1, 1, 1));
    material->setAlpha(Material::FRONT_AND_BACK, 1.f);

    Image *idleImage = osgDB::readImageFile(ANIMDataDir() + "Textures/Pallette/PalletteSigniture.BMP");
    Texture2D* idleTexture = new Texture2D(idleImage);    
    
    StateSet *idleStateSet = ((*textureStateIdleEntry)->mEntryGeode)->getOrCreateStateSet();
    idleStateSet->setMode(GL_BLEND, StateAttribute::OVERRIDE | StateAttribute::ON );
    idleStateSet->setRenderingHint(StateSet::TRANSPARENT_BIN);
    idleStateSet->setAttributeAndModes(material, StateAttribute::OVERRIDE | StateAttribute::ON);
    idleStateSet->setTextureAttributeAndModes(0, idleTexture, StateAttribute::ON);

    // create zoom in & out animations for 'DSTexturePallette': 'IDLE' state
    PositionAttitudeTransform *idleStatePATransFwd = new PositionAttitudeTransform;
    PositionAttitudeTransform *idleStatePATransBwd = new PositionAttitudeTransform;

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
        animationPathScaleFwd->insert(val, AnimationPath::ControlPoint(Vec3(0, 0, 0), Quat(), scaleFwd));
        animationPathScaleBwd->insert(val, AnimationPath::ControlPoint(Vec3(0, 0, 0), Quat(), scaleBwd));
    }

    (*textureStateIdleEntry)->mFwdAnim = new AnimationPathCallback(animationPathScaleFwd, 
						0.0, 1.f / ANIM_VIRTUAL_SPHERE_LAPSE_TIME);
    (*textureStateIdleEntry)->mBwdAnim = new AnimationPathCallback(animationPathScaleBwd, 
						0.0, 1.f / ANIM_VIRTUAL_SPHERE_LAPSE_TIME); 

    // setup scene graph tree
    (*idleStateSwitch)->addChild((*textureStateIdleEntry)->mEntrySwitch);

    (*textureStateIdleEntry)->mEntrySwitch->addChild(idleStatePATransFwd);
    (*textureStateIdleEntry)->mEntrySwitch->addChild(idleStatePATransBwd);

    idleStatePATransFwd->addChild((*textureStateIdleEntry)->mEntryGeode);
    idleStatePATransBwd->addChild((*textureStateIdleEntry)->mEntryGeode);

    idleStatePATransFwd->setUpdateCallback((*textureStateIdleEntry)->mFwdAnim);
    idleStatePATransBwd->setUpdateCallback((*textureStateIdleEntry)->mBwdAnim);
}


/***************************************************************
* Function: ANIMLoadTexturePalletteSelect()
*
* selectStateSwitch: Root Switch for all texture spheres
* alphaTurnerSwitch: Root Switch for alpha value indicator
* numTexs: Number of textures that loaded to DSTexturePallette
* textureStatesEntryArray: Data pointer to texture states
*
***************************************************************/
void ANIMLoadTexturePalletteSelect(Switch **selectStateSwitch, Switch **alphaTurnerSwitch,
				    int &numTexs, ANIMTexturePalletteSelectEntry ***textureStatesEntryArray)
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

    // initialize texture state entry record array and randomly generated showing up positions array
    numTexs = 24;
    *textureStatesEntryArray = new ANIMTexturePalletteSelectEntry*[numTexs];
    for (int i = 0; i < numTexs; i++) 
    {
        (*textureStatesEntryArray)[i] = new ANIMTexturePalletteSelectEntry;
    }
    Vec3 *showUpPosArray = new Vec3[numTexs];
    ANIMCreateRandomShowupPosArray(numTexs, &showUpPosArray);

    // plain white texture and color
    Vec3 voidColor = Vec3(1, 1, 1);
    string voidTex = ANIMDataDir() + "Textures/White.JPG";


    // Load texture filenames from config file
    bool isFile = true;
    int j = 0, numTex;
    std::string filename = "", dir, path;
    std::vector<std::string> filenames;
    
    dir = cvr::ConfigManager::getEntry("dir", "Plugin.CaveCADBeta.Textures", "/home/cehughes");
    dir = dir + "/";
    path = "Plugin.CaveCADBeta.Textures.0";
    filename = cvr::ConfigManager::getEntry(path, "", &isFile);

    while (isFile)
    {
        filenames.push_back(dir + filename);

        j++;
        char buf[50];
        sprintf(buf, "Plugin.CaveCADBeta.Textures.%d", j);
        std::string path = std::string(buf);
        filename = cvr::ConfigManager::getEntry(path, "", &isFile);
    }

    numTex = j;
    
    osg::Vec3 subpos(0, 0, -0.5);
    for (int i = 0; i < numTex; ++i)
    {
        ANIMCreateTextureEntryGeode(showUpPosArray[i] + subpos, voidColor, voidColor, filenames[i],
				&((*textureStatesEntryArray)[i]));
    }

    // create 10 texture entries
/*    ANIMCreateTextureEntryGeode(showUpPosArray[0], voidColor, voidColor, headerTex + "00.JPG", 
				&((*textureStatesEntryArray)[0]));
    ANIMCreateTextureEntryGeode(showUpPosArray[1], voidColor, voidColor, headerTex + "01.JPG", 
				&((*textureStatesEntryArray)[1]));
    ANIMCreateTextureEntryGeode(showUpPosArray[2], voidColor, voidColor, headerTex + "02.JPG", 
				&((*textureStatesEntryArray)[2]));
    ANIMCreateTextureEntryGeode(showUpPosArray[3], voidColor, voidColor, headerTex + "03.JPG", 
				&((*textureStatesEntryArray)[3]));
    ANIMCreateTextureEntryGeode(showUpPosArray[4], voidColor, voidColor, headerTex + "04.JPG", 
				&((*textureStatesEntryArray)[4]));
    ANIMCreateTextureEntryGeode(showUpPosArray[5], voidColor, voidColor, headerTex + "05.JPG", 
				&((*textureStatesEntryArray)[5]));
    ANIMCreateTextureEntryGeode(showUpPosArray[6], voidColor, voidColor, headerTex + "06.JPG", 
				&((*textureStatesEntryArray)[6]));
    ANIMCreateTextureEntryGeode(showUpPosArray[7], voidColor, voidColor, headerTex + "07.JPG", 
				&((*textureStatesEntryArray)[7]));
    ANIMCreateTextureEntryGeode(showUpPosArray[8], voidColor, voidColor, headerTex + "08.JPG", 
				&((*textureStatesEntryArray)[8]));
    ANIMCreateTextureEntryGeode(showUpPosArray[9], voidColor, voidColor, headerTex + "09.JPG", 
				&((*textureStatesEntryArray)[9]));
*/

    string headerTex = ANIMDataDir() + "Textures/Pallette/";
    // create 14 color entries
    std::vector<osg::Vec3> colors;
    colors.push_back(osg::Vec3(0.0, 0.0, 0.0)); // black
    colors.push_back(osg::Vec3(0.5, 0.5, 0.5)); // grey
    colors.push_back(osg::Vec3(1.0, 1.0, 1.0)); // white
    colors.push_back(osg::Vec3(1.0, 0.0, 0.0)); // red 
    colors.push_back(osg::Vec3(0.0, 1.0, 0.0)); // green

    colors.push_back(osg::Vec3(0.0, 0.0, 1.0)); // blue
    colors.push_back(osg::Vec3(1.0, 1.0, 0.0)); // yellow
    colors.push_back(osg::Vec3(1.0, 0.0, 1.0)); // pink
    colors.push_back(osg::Vec3(0.0, 1.0, 1.0)); // cyan
    colors.push_back(osg::Vec3(0.5, 0.0, 1.0)); // purple

    colors.push_back(osg::Vec3(1.0, 0.5, 0.0)); // orange
    colors.push_back(osg::Vec3(0.0, 0.5, 1.0)); // sky blue
    colors.push_back(osg::Vec3(1.0, 0.0, 0.5)); // bright pink
    colors.push_back(osg::Vec3(0.0, 0.5, 1.0)); // light bluegreenish
    
    subpos = osg::Vec3(0, 0, -1.5);
    for (int i = 0; i < colors.size(); ++i)
    {
        ANIMCreateTextureEntryGeode(showUpPosArray[i] + subpos, colors[i], colors[i], voidTex,
				&((*textureStatesEntryArray)[i+10]));
    }

/*    ANIMCreateTextureEntryGeode(showUpPosArray[10], Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0), voidTex,
				&((*textureStatesEntryArray)[10]));
    ANIMCreateTextureEntryGeode(showUpPosArray[11], Vec3(0.5, 0.5, 0.5), Vec3(0.5, 0.5, 0.5), voidTex,
				&((*textureStatesEntryArray)[11]));
    ANIMCreateTextureEntryGeode(showUpPosArray[12], Vec3(1.0, 1.0, 1.0), Vec3(1.0, 1.0, 1.0), voidTex,
				&((*textureStatesEntryArray)[12]));
    ANIMCreateTextureEntryGeode(showUpPosArray[13], Vec3(1.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0), voidTex,
				&((*textureStatesEntryArray)[13]));
    ANIMCreateTextureEntryGeode(showUpPosArray[14], Vec3(0.0, 1.0, 0.0), Vec3(0.0, 1.0, 0.0), voidTex,
				&((*textureStatesEntryArray)[14]));
    ANIMCreateTextureEntryGeode(showUpPosArray[15], Vec3(0.0, 0.0, 1.0), Vec3(0.0, 0.0, 1.0), voidTex,
				&((*textureStatesEntryArray)[15]));
    ANIMCreateTextureEntryGeode(showUpPosArray[16], Vec3(1.0, 1.0, 0.0), Vec3(1.0, 1.0, 0.0), voidTex,
				&((*textureStatesEntryArray)[16]));
    ANIMCreateTextureEntryGeode(showUpPosArray[17], Vec3(0.0, 1.0, 1.0), Vec3(0.0, 1.0, 1.0), voidTex,
				&((*textureStatesEntryArray)[17]));
    ANIMCreateTextureEntryGeode(showUpPosArray[18], Vec3(1.0, 0.0, 1.0), Vec3(1.0, 0.0, 1.0), voidTex,
				&((*textureStatesEntryArray)[18]));
    ANIMCreateTextureEntryGeode(showUpPosArray[19], Vec3(0.5, 1.0, 0.0), Vec3(0.5, 1.0, 0.0), voidTex,
				&((*textureStatesEntryArray)[19]));
    ANIMCreateTextureEntryGeode(showUpPosArray[20], Vec3(0.0, 0.5, 1.0), Vec3(0.0, 0.5, 1.0), voidTex,
				&((*textureStatesEntryArray)[20]));
    ANIMCreateTextureEntryGeode(showUpPosArray[21], Vec3(1.0, 1.0, 0.5), Vec3(1.0, 1.0, 0.5), voidTex,
				&((*textureStatesEntryArray)[21]));
    ANIMCreateTextureEntryGeode(showUpPosArray[22], Vec3(0.5, 1.0, 1.0), Vec3(0.5, 1.0, 1.0), voidTex,
				&((*textureStatesEntryArray)[22]));
    ANIMCreateTextureEntryGeode(showUpPosArray[23], Vec3(1.0, 1.0, 0.5), Vec3(1.0, 1.0, 0.5), voidTex,
				&((*textureStatesEntryArray)[23]));
*/
    // add entry animation objects to 'selectStateSwitch'
    for (int i = 0; i < numTexs; i++) 
    {
        (*selectStateSwitch)->addChild((*textureStatesEntryArray)[i]->mEntrySwitch);
    }

    //ANIMCreateTextureEntryGeode(subpos + osg::Vec3(0, 0, -1.75), osg::Vec3(1,1,1), osg::Vec3(1,1,1),
    //    voidTex, &((*textureStatesEntryArray)[24]));
    //(*selectStateSwitch)->addChild((*textureStatesEntryArray)[24]->mEntrySwitch);

    // get audio parameters of each material from audio configuration file
    FILE *audioInfoFilePtr = fopen((headerTex + string("AudioInfo.cfg")).c_str(), "r");
    if (!audioInfoFilePtr)
    {
        cerr << "CaveCADBeta ANIMLoadTexturePalletteSelect: Error opening audio config file." << endl;
        return;
    }

    char audioentry[128];
    for (int i = 0; i < numTexs; i++)
    {
        if (!feof(audioInfoFilePtr))
        {
            fgets(audioentry, 128, audioInfoFilePtr);
            string str = string(audioentry);
            str.erase(str.size()-1, 1);		// discard the last '/n' at the end of each line
            (*textureStatesEntryArray)[i]->setAudioInfo(str);
        }
    }

    if (audioInfoFilePtr) fclose((FILE*)audioInfoFilePtr);
}


/***************************************************************
* Function: ANIMCreateTextureEntryGeode()
*
* Called by function 'ANIMLoadTexturePalletteSelect()' only.
* writes in record of 'textureEntry' and creates three chains
* of animations given the 'showUpPos', diffuse & specular color
* and directory of the texture file.
*
***************************************************************/
void ANIMCreateTextureEntryGeode(const Vec3 &showUpPos, const Vec3 &diffuse, const Vec3 &specular,
				 const string &texfilename, ANIMTexturePalletteSelectEntry **textureEntry)
{
    (*textureEntry)->mEntrySwitch = new Switch();
    (*textureEntry)->mEntrySwitch->setAllChildrenOn();
    (*textureEntry)->mEntryGeode = new Geode();

    // set data record in protected field
    (*textureEntry)->setDiffuse(diffuse);
    (*textureEntry)->setSpecular(specular);
    (*textureEntry)->setTexFilename(texfilename);

    // create drawables, geodes and texture mapping
    Sphere *texEntrySphere = new Sphere(Vec3(0, 0, 0), ANIM_TEXTURE_PALLETTE_ENTRY_SPHERE_RADIUS);
    ShapeDrawable *texEntryDrawable = new ShapeDrawable(texEntrySphere);
    (*textureEntry)->mEntryGeode->addDrawable(texEntryDrawable);

    Material *entryMaterial = new Material;
    entryMaterial->setDiffuse(Material::FRONT_AND_BACK, Vec4(diffuse, 1.f));
    entryMaterial->setSpecular(Material::FRONT_AND_BACK, Vec4(specular, 1.f));
    entryMaterial->setAlpha(Material::FRONT_AND_BACK, 1.f);

    Image *entryImage = osgDB::readImageFile(texfilename);
    Texture2D* entryTexture = new Texture2D(entryImage);    
    
    StateSet *idleStateSet = (*textureEntry)->mEntryGeode->getOrCreateStateSet();
    idleStateSet->setMode(GL_BLEND, StateAttribute::OVERRIDE | StateAttribute::ON );
    idleStateSet->setRenderingHint(StateSet::TRANSPARENT_BIN);
    idleStateSet->setAttributeAndModes(entryMaterial, StateAttribute::OVERRIDE | StateAttribute::ON);
    idleStateSet->setTextureAttributeAndModes(0, entryTexture, StateAttribute::ON);

    // create three way animations
    PositionAttitudeTransform **statePATransArray = new PositionAttitudeTransform*[8];
    for (int i = 0; i < 8; i++)
    {
        statePATransArray[i] = new PositionAttitudeTransform;

        (*textureEntry)->mEntrySwitch->addChild(statePATransArray[i]);
        statePATransArray[i]->addChild((*textureEntry)->mEntryGeode);
    }

    // set up animation paths for four way transitions
    AnimationPath **animationPathsArray = new AnimationPath*[8];
    for (int i = 0; i < 8; i++)
    {
        animationPathsArray[i] = new AnimationPath;
        animationPathsArray[i]->setLoopMode(AnimationPath::NO_LOOPING);
    }

    Vec3 scaleFwd1, scaleBwd1, scaleFwd2, scaleBwd2, scaleFwd3, scaleBwd3,transFwd, transBwd;
    float step = 1.f / ANIM_VIRTUAL_SPHERE_NUM_SAMPS;
    float diff = (ANIM_VIRTUAL_SPHERE_RADIUS / ANIM_TEXTURE_PALLETTE_ENTRY_SPHERE_RADIUS - 1.0f);
    for (int i = 0; i < ANIM_VIRTUAL_SPHERE_NUM_SAMPS + 1; i++)
    {
        float val = i * step;
        scaleFwd1 = Vec3(val, val, val);
        scaleBwd1 = Vec3(1.f - val, 1.f - val, 1.f - val);
        scaleFwd2 = Vec3(1 + val * diff, 1 + val * diff, 1 + val * diff);
        scaleBwd2 = Vec3(1 + (1 - val) * diff, 1 + (1 - val) * diff, 1 + (1 - val) * diff);
        scaleFwd3 = Vec3((1 + diff) * val, (1 + diff) * val, (1 + diff) * val);
        scaleBwd3 = Vec3((1 + diff) * (1 - val), (1 + diff) * (1 - val), (1 + diff) * (1 - val));
        transFwd = showUpPos * val;
        transBwd = showUpPos * (1.f - val);
        
        // new menu positioning
        osg::Vec3 offset(0, 0, 0);
        transFwd += offset;
        transBwd += offset;

        animationPathsArray[0]->insert(val, AnimationPath::ControlPoint(transFwd, Quat(), scaleFwd1));
        animationPathsArray[1]->insert(val, AnimationPath::ControlPoint(transBwd, Quat(), scaleBwd1));
        animationPathsArray[2]->insert(val, AnimationPath::ControlPoint(showUpPos, Quat(), scaleBwd1));
        animationPathsArray[3]->insert(val, AnimationPath::ControlPoint(showUpPos, Quat(), scaleFwd1));
        animationPathsArray[4]->insert(val, AnimationPath::ControlPoint(transBwd, Quat(), scaleFwd2));
        animationPathsArray[5]->insert(val, AnimationPath::ControlPoint(transFwd, Quat(), scaleBwd2));
        animationPathsArray[6]->insert(val, AnimationPath::ControlPoint(Vec3(0, 0, 0), Quat(), scaleBwd3));
        animationPathsArray[7]->insert(val, AnimationPath::ControlPoint(Vec3(0, 0, 0), Quat(), scaleFwd3));
    }

    (*textureEntry)->mStateAnimationArray = new AnimationPathCallback*[8];
    for (int i = 0; i < 8; i++)
    {
        (*textureEntry)->mStateAnimationArray[i] =  new AnimationPathCallback(animationPathsArray[i], 
                                0.0, 1.f / ANIM_TEXTURE_PALLETTE_ANIMATION_TIME);
        statePATransArray[i]->setUpdateCallback((*textureEntry)->mStateAnimationArray[i]);
    }
}


/***************************************************************
* Function: ANIMCreateTextureEntryGeode()
*
* Called by function 'ANIMLoadTexturePalletteSelect()' only.
* writes in Vec3 array of 'showUpPosArray' using a sequence of
* randomly generated integers between 0 and 32768
*
***************************************************************/
void ANIMCreateRandomShowupPosArray(const int &numTexs, Vec3 **showUpPosArray)
{
    /*  The reason for using a pre-generated integer sequence rather than running a
	random generator is that different sequences might be generated on parallel
	rendering systems due to asynchronized clocks.
    */
    int randIntArray[128] = {	16441, 14510, 6757, 16176, 25835, 13200, 13102, 8554,
				12567, 10259, 1672, 19629, 15599, 11314, 20131, 14066, 
				15173, 5089, 31814, 26782, 4148, 5813, 23049, 9315, 
				30093, 19882, 6030, 17428, 28061, 461, 19959, 28176, 
				24337, 15280, 10967, 4185, 29384, 16015, 28434, 23389, 
				11910, 6510, 21285, 18324, 19900, 15183, 10903, 1625,
				881, 32495, 19518, 5992, 7896, 28048, 21848, 6532, 
				32227, 1897, 20859, 19979, 20350, 13185, 10626, 11328, 
				24887, 4041, 30655, 19003, 4612, 13717, 32733, 20103, 
				1625, 9581, 3333, 3771, 24426, 28831, 24795, 29758, 
				518, 24010, 10790, 14775, 26441, 12557, 10775, 4295, 
				916, 28503, 17189, 16538, 1679, 26543, 9392, 21318, 
				4278, 1103, 8886, 28885, 31097, 24721, 2304, 5129, 
				25290, 24522, 23893, 22047, 7494, 1127, 3141, 9146, 
				18978, 3479, 26762, 18081, 13884, 27917, 13467, 19624, 
				7122, 30076, 8907, 30103, 20067, 13712, 20447, 30425 };

    // Showing up positions are evenly distributed with the spherical region with radius 3R.
    for (int i = 0; i < numTexs; i++)
    {
        Vec3 pos = Vec3(randIntArray[i * 3], randIntArray[i * 3 + 1], randIntArray[i * 3 + 2]);
        pos = (pos / 16384.f - Vec3(1, 1, 1)) * ANIM_VIRTUAL_SPHERE_RADIUS * 3;
        (*showUpPosArray)[i] = pos;
    }
}


void ANIMLoadColorEntries(osg::Switch **selectStateSwitch, osg::Switch **alphaTurnerSwitch,
    std::vector<osg::Vec3> *colors, std::vector<ANIMTexturePalletteSelectEntry*> *colorEntries)
{
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

 
    Vec3 *showUpPosArray = new Vec3[colors->size()];
    ANIMCreateRandomShowupPosArray(colors->size(), &showUpPosArray);

    // plain white texture and color
    string voidTex = ANIMDataDir() + "Textures/White.JPG";

    osg::Vec3 subpos = osg::Vec3(0, 0, -1.5);
    for (int i = 0; i < colors->size(); ++i)
    {
        colorEntries->push_back(new ANIMTexturePalletteSelectEntry());
 
        ANIMCreateTextureEntryGeode(showUpPosArray[i] + subpos, (*colors)[i], (*colors)[i], voidTex,
				&((*colorEntries)[i]));

        (*selectStateSwitch)->addChild((*colorEntries)[i]->mEntrySwitch);
    }
}

void ANIMLoadTextureEntries(osg::Switch **selectStateSwitch, osg::Switch **alphaTurnerSwitch,
    std::vector<std::string> *filenames, std::vector<ANIMTexturePalletteSelectEntry*> *textureEntries)
{
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

 
    Vec3 *showUpPosArray = new Vec3[filenames->size()];
    ANIMCreateRandomShowupPosArray(filenames->size(), &showUpPosArray);

    Vec3 voidColor = Vec3(1, 1, 1);
    osg::Vec3 subpos(0, 0, -0.5);
    for (int i = 0; i < filenames->size(); ++i)
    {
        textureEntries->push_back(new ANIMTexturePalletteSelectEntry());

        ANIMCreateTextureEntryGeode(showUpPosArray[i] + subpos, voidColor, voidColor, (*filenames)[i],
				&((*textureEntries)[i]));

        (*selectStateSwitch)->addChild((*textureEntries)[i]->mEntrySwitch);
    }

}

void ANIMAddColorEntry(osg::Switch **selectStateSwitch, osg::Switch **alphaTurnerSwitch,
    osg::Vec3 color, std::vector<ANIMTexturePalletteSelectEntry*> *colorEntries)
{
    int size = colorEntries->size();
    osg::Vec3 *showUpPosArray = new osg::Vec3[size + 1];
    ANIMCreateRandomShowupPosArray(size + 1, &showUpPosArray);

    string voidTex = ANIMDataDir() + "Textures/White.JPG";
    osg::Vec3 subpos = osg::Vec3(0, 0, -1.5);

    colorEntries->push_back(new ANIMTexturePalletteSelectEntry());

    ANIMCreateTextureEntryGeode(showUpPosArray[size] + subpos, color, color, voidTex,
            &((*colorEntries)[size]));

    (*selectStateSwitch)->addChild((*colorEntries)[size]->mEntrySwitch);
}


};

