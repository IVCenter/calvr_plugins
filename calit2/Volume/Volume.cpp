#include "Volume.h"

#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/SceneManager.h>
#include <cvrKernel/PluginManager.h>
#include <cvrKernel/PluginHelper.h>
#include <cvrKernel/NodeMask.h>
#include <cvrMenu/MenuSystem.h>
#include <cvrKernel/ComController.h>
#include <cvrUtil/Intersection.h>
#include <iostream>

#include <osg/Node>
#include <osg/Geometry>
#include <osg/Notify>
#include <osg/Texture3D>
#include <osg/Texture1D>
#include <osg/TexGen>
#include <osg/Geode>
#include <osg/Billboard>
#include <osg/PositionAttitudeTransform>
#include <osg/ClipNode>
#include <osg/AlphaFunc>
#include <osg/TexGenNode>
#include <osg/TexEnv>
#include <osg/TexEnvCombine>
#include <osg/Material>
#include <osg/PrimitiveSet>
#include <osg/Endian>
#include <osg/BlendFunc>
#include <osg/BlendEquation>
#include <osg/TransferFunction>
#include <osg/ShapeDrawable>
#include <osg/PolygonMode>
#include <osg/Material>
#include <osg/MatrixTransform>
#include <osg/ShapeDrawable>
#include <osg/Point>
#include <osgUtil/Simplifier>

#include <osgDB/Registry>
#include <osgDB/ReadFile>
#include <osgDB/WriteFile>
#include <osgDB/FileUtils>
#include <osgDB/FileNameUtils>

#include <osg/ImageUtils>
#include <osgVolume/Volume>
#include <osgVolume/VolumeTile>
#include <osgVolume/RayTracedTechnique>
#include <osgVolume/FixedFunctionTechnique>

#include "cvrImageSequence.h"

#include <string.h>
#include <fstream>

using namespace std;
using namespace cvr;

// uid generator
int Volume::id;

CVRPLUGIN(Volume)


struct ScaleOperator
{
    ScaleOperator():_scale(1.0f) {}
    ScaleOperator(float scale):_scale(scale) {}
    ScaleOperator(const ScaleOperator& so):_scale(so._scale) {}

    ScaleOperator& operator = (const ScaleOperator& so) { _scale = so._scale; return *this; }

    float _scale;

    inline void luminance(float& l) const { l*= _scale; }
    inline void alpha(float& a) const { a*= _scale; }
    inline void luminance_alpha(float& l,float& a) const { l*= _scale; a*= _scale;  }
    inline void rgb(float& r,float& g,float& b) const { r*= _scale; g*=_scale; b*=_scale; }
    inline void rgba(float& r,float& g,float& b,float& a) const { r*= _scale; g*=_scale; b*=_scale; a*=_scale; }
};

Volume::Volume() : FileLoadCallback("xvf")
{
}

void Volume::clampToNearestValidPowerOfTwo(int& sizeX, int& sizeY, int& sizeZ, int s_maximumTextureSize, int t_maximumTextureSize, int r_maximumTextureSize)
{
    int s_nearestPowerOfTwo = 1;
    while(s_nearestPowerOfTwo<sizeX && s_nearestPowerOfTwo<s_maximumTextureSize) s_nearestPowerOfTwo*=2;

    int t_nearestPowerOfTwo = 1;
    while(t_nearestPowerOfTwo<sizeY && t_nearestPowerOfTwo<t_maximumTextureSize) t_nearestPowerOfTwo*=2;

    int r_nearestPowerOfTwo = 1;
    while(r_nearestPowerOfTwo<sizeZ && r_nearestPowerOfTwo<r_maximumTextureSize) r_nearestPowerOfTwo*=2;

    sizeX = s_nearestPowerOfTwo;
    sizeY = t_nearestPowerOfTwo;
    sizeZ = r_nearestPowerOfTwo;
}

struct Volume::volumeinfo* Volume::loadXVF(std::string filename, int &x, int &y, int &z, float &xMultiplier, float &yMultiplier, float &zMultiplier, 
                                int& sizeS, int& sizeT, int &sizeR, int& numberBytesPerComponent)
{
    struct volumeinfo* volinfo = NULL;

    vvVolDesc* vol = new vvVolDesc(filename.c_str());

	vvFileIO vvIO;
	vvFileIO::ErrorType type = vvIO.loadVolumeData(vol);
	if(type != vvFileIO::OK)
	{
	    std::cerr << "Error reading XVF file\n";
	    delete vol;
	    return volinfo;
	}

	// create a new volume
	volinfo = new struct volumeinfo;

    vol->toggleEndianness();
	x = vol->vox[0];
	y = vol->vox[1];
	z = vol->vox[2];
	xMultiplier = vol->dist[0];
	yMultiplier = vol->dist[1];
	zMultiplier = vol->dist[2];
	numberBytesPerComponent = vol->bpc;

    // create an image sequence even if single frame
    osg::ref_ptr<cvrImageSequence> imageSequence = new cvrImageSequence;
    imageSequence->setLength(vol->getStoredFrames());
    imageSequence->setLoopingMode( osg::ImageStream::LOOPING );
    imageSequence->setLength(vol->getStoredFrames() * 0.4f);
    imageSequence->pause();

    std::cerr << "Number of frames found " << vol->getStoredFrames() << std::endl;

    unsigned int r_offset = 0;
    unsigned int r_line = 0;
	GLenum dataType;

    for(int i = 0; i < vol->getStoredFrames(); i++)
    {
	    // allocate volume
	    osg::Image* image = allocateVolume(x, y, z, sizeS, sizeT, sizeR, numberBytesPerComponent, dataType);

        // read into voldesc data
        uchar* voldata = vol->getRaw(i);
        r_offset = (z < sizeR) ? sizeR/2 - z/2 : 0;
        r_line = x * numberBytesPerComponent;

        // read in data
        unsigned int counter = 0;
        for(int r=0;r<z;++r)
        {
            for(int t=0;t<y;++t)
            {
                char* data = (char*) image->data(0,t,r+r_offset);
                memcpy(data, voldata + (counter * r_line), r_line);
                counter++;
            }
        }
        imageSequence->addImage(image);
    }

    // assign image and transfer function
    volinfo->image = imageSequence.get();
    volinfo->r_offset = r_offset;
    volinfo->numberOfBytesPerComponent = numberBytesPerComponent;
    volinfo->datatype = dataType;
    volinfo->id = id++;
    volinfo->numberOfFrames = imageSequence->getNumImages();
    
    // compute transferFunction make adjustemnt for scalar shift
    osg::Vec4 minValue, maxValue;
    osg::computeMinMax(volinfo->image.get(), minValue, maxValue);
    std::cerr << "Min " << minValue.r() << " max value " << maxValue.r() << std::endl;
    
    // create a transferfunction
	volinfo->tf = new osg::TransferFunction1D();

    // create colorMap (access widget and get center and min max ranges)
    // Looks only for PyramidWidgets
    int size = vol->tf._widgets.count();
    vol->tf._widgets.first();
    float scalar = 1.0 / maxValue.r();

    std::cerr << "Scale value " << scalar << std::endl;

    // temporary transferfunction table
    volinfo->defaultTransferFunc = new vvTransFunc();

    for(int i = 0; i < size; i++)
    {
        vvTFWidget* w = vol->tf._widgets.getData();

        if( vvTFPyramid *pw = dynamic_cast<vvTFPyramid*>(w) )
        {
            volinfo->center = pw->_pos[0];
            volinfo->width = pw->_bottom[0];
            volinfo->defaultTransferFunc->_widgets.append(new vvTFPyramid(vvColor(1.0f, 1.0f, 1.0f), false, 1.0f, volinfo->center * scalar, volinfo->width * scalar, 0.0f),  vvSLNode<vvTFWidget*>::NORMAL_DELETE);
        }

        // check color pin values
        if( vvTFColor *cw = dynamic_cast<vvTFColor*>(w) )
        {
            float centerValue = cw->_pos[0];
            osg::Vec3f color(0.0, 0.0, 0.0);
            cw->_col.getRGB(color.x(), color.y(), color.z());
            volinfo->defaultTransferFunc->_widgets.append(new vvTFColor(vvColor(color.x(), color.y(), color.z()), centerValue * scalar),  vvSLNode<vvTFWidget*>::NORMAL_DELETE);
        }

        vol->tf._widgets.next();
    }

    // add colormap to transfer function
	volinfo->tf->assign(*(computeColorMap(volinfo->defaultTransferFunc, volinfo->center *  scalar, volinfo->width * scalar)));

    // remove voldesc object
    if( vol )
        delete vol;
    vol = NULL;

    return volinfo;
}

osg::Image* Volume::allocateVolume(int x, int y, int z, int& sizeS, int& sizeT, int& sizeR, int numberBytesPerComponent, GLenum& dataType)
{
    switch(numberBytesPerComponent)
    {
        case 1 : dataType = GL_UNSIGNED_BYTE; break;
        case 2 : dataType = GL_UNSIGNED_SHORT; break;
        case 4 : dataType = GL_UNSIGNED_INT; break;
        default :
            osg::notify(osg::NOTICE)<<"Error: numberBytesPerComponent="<<numberBytesPerComponent<<" not supported, only 1,2 or 4 are supported."<<std::endl;
             return 0;
                         
    }

    // create volume
    int s_maximumTextureSize=2048, t_maximumTextureSize=2048, r_maximumTextureSize=2048;  
    sizeS = x;
    sizeT = y;
    sizeR = z;

    // set to power of 2
    clampToNearestValidPowerOfTwo(sizeS, sizeT, sizeR, s_maximumTextureSize, t_maximumTextureSize, r_maximumTextureSize);

    osg::Image* image = new osg::Image;
    image->allocateImage(sizeS, sizeT, sizeR, GL_LUMINANCE, dataType);
    return image;
}

float Volume::convertValue(GLenum dataType, const char* data)
{
    float value = 0.0;
    switch(dataType)
    {
        case(GL_BYTE):              value = float(*(const char*)(data)) / 128.0f; break;
        case(GL_UNSIGNED_BYTE):     value = float(*(const unsigned char*)(data)) /255.0f; break;
        case(GL_SHORT):             value = float(*(const short*)(data)) /32768.0f; break;
        case(GL_UNSIGNED_SHORT):    value = float(*(const unsigned short*)(data)) /65535.0f; break;
        case(GL_INT):               value = float(*(const int*)(data)) /2147483648.0f; break;
        case(GL_UNSIGNED_INT):      value = float(*(const unsigned int*)(data)) /4294967295.0f; break;
        case(GL_FLOAT):             value = float(*(const float*)(data)); break;
    }
    return value; 
}

void Volume::insertValue(GLenum dataType, char* data, float value)
{
    switch(dataType)
    {
        case(GL_BYTE):              *(char*)data = (char)(value * 128.0f); break;
        case(GL_UNSIGNED_BYTE):     *(unsigned char*)data = (unsigned char)(value * 255.0f); break;
        case(GL_SHORT):             *(short*)data = (short)(value * 32768.0f); break;
        case(GL_UNSIGNED_SHORT):    *(unsigned short*)data = (unsigned short)(value * 65535.0f); break;
        case(GL_INT):               *(int*)data = (int)(value * 2147483648.0f); break;
        case(GL_UNSIGNED_INT):      *(unsigned int*)data = (unsigned int)(value * 4294967295.0f); break;
        case(GL_FLOAT):             *(float*)data = value; break;
    }
}

bool Volume::loadFile(std::string filename)
{
    int numberBytesPerComponent = 0;
    float xMultiplier, yMultiplier, zMultiplier;
    xMultiplier = yMultiplier = zMultiplier = 1.0;

    // temporary parameters
    std::string name;
    int x, y ,z, sizeS, sizeT, sizeR;

    std::cerr << "Load called on " << filename << std::endl;

    volumeinfo *info = NULL;

    // find out what type of file
    if( osgDB::getFileExtension(filename) == "xvf" )
    {
        name =  osgDB::getSimpleFileName(filename);
        info = loadXVF(filename, x, y, z, xMultiplier, yMultiplier, zMultiplier, sizeS, sizeT, sizeR, numberBytesPerComponent);
    }

    // check to see if a volume was loaded
    if( !info )
        return false;

    float alphaFunc=0.02f;
    double sampleDensityWhenMoving = 0.0;
   
    // create volume info
    info->name = name;
    info->x = x;
    info->y = y;
    info->z = z;
    info->xMultiplier = xMultiplier;
    info->yMultiplier = yMultiplier;
    info->zMultiplier = zMultiplier;
    info->clipUniform = new osg::Uniform("clipPlane", false);
    
    // average values and spread over range 
    {
        // compute range of values
        osg::Vec4 minValue, maxValue;
        osg::computeMinMax(info->image.get(), minValue, maxValue); //TODO
        osg::modifyImage(info->image.get(),ScaleOperator(1.0f/maxValue.r()));
        //std::cerr << "Min xyz " << minValue.x() << " " << minValue.y() << " " << minValue.z() << std::endl; 
        //std::cerr << "Max xyz " << maxValue.x() << " " << maxValue.y() << " " << maxValue.z() << std::endl; 
    }

    // should be in the range of -0.5 to 0.5 (0.0, 0.0, 0.0 is the bottom left hand corner of the volume)
    info->clipposUniform = new osg::Uniform("position", osg::Vec3(0.5, 0.5, 0.7));
    info->clipnormUniform = new osg::Uniform("normal", osg::Vec3(0.0, 0.5, 0.0));
    info->isosurfaceValueMin = new osg::Uniform("IsoSurfaceValueMin", 0.6f);
   
    osg::ref_ptr<osgVolume::Volume> volume = new osgVolume::Volume;
    info->tile = new osgVolume::VolumeTile;
    volume->addChild(info->tile);

    osg::StateSet* state = info->tile->getOrCreateStateSet();
    state->addUniform(info->clipUniform);
    state->addUniform(info->clipposUniform);
    state->addUniform(info->clipnormUniform);
    state->addUniform(info->isosurfaceValueMin);

    osg::ref_ptr<osgVolume::ImageLayer> layer = new osgVolume::ImageLayer(info->image);
    layer->rescaleToZeroToOneRange();   //TODO
    layer->setTexelScale(osg::Vec4(1.0, 1.0, 1.0, 1.0));

    osg::ref_ptr<osg::RefMatrix> matrix = new osg::RefMatrix(sizeS, 0.0,   0.0,   0.0,
                                    0.0,   sizeT, 0.0,   0.0,
                                    0.0,   0.0,   sizeR, 0.0,
                                    0.0,   0.0,   0.0,   1.0);

    if (info->xMultiplier!=1.0 || info->yMultiplier!=1.0 || info->zMultiplier!=1.0)
    {
        matrix->postMultScale(osg::Vec3d(fabs(info->xMultiplier), fabs(info->yMultiplier), fabs(info->zMultiplier)));
    }

    layer->setLocator(new osgVolume::Locator(*matrix));
    info->tile->setLocator(new osgVolume::Locator(*matrix));
    info->tile->setLayer(layer.get());

    // compute bounds
    if( x > sizeS )
        x = sizeS;
    if( y > sizeT )
        y = sizeT;
    if( z > sizeR )
        z = sizeR;

    //create a frame around the volume
    osg::MatrixTransform* mattrans = new osg::MatrixTransform;
    mattrans->setMatrix(*matrix);

    osg::Geode * geode = new osg::Geode();
    osg::Geometry* boxgeom = new osg::Geometry();
    osg::Vec3Array * vertices = new osg::Vec3Array(24);

    (*vertices)[0].set(0.0, 0.0, 0.0);
    (*vertices)[1].set(0.0, 1.0, 0.0);
    (*vertices)[2].set(1.0, 1.0, 0.0);
    (*vertices)[3].set(1.0, 0.0, 0.0);

    (*vertices)[4].set(0.0, 0.0, 1.0);
    (*vertices)[5].set(0.0, 1.0, 1.0);
    (*vertices)[6].set(1.0, 1.0, 1.0);
    (*vertices)[7].set(1.0, 0.0, 1.0);

    (*vertices)[8].set(0.0, 0.0, 0.0);
    (*vertices)[9].set(0.0, 0.0, 1.0);
    (*vertices)[10].set(1.0, 0.0, 1.0);
    (*vertices)[11].set(1.0, 0.0, 0.0);

    (*vertices)[12].set(0.0, 1.0, 0.0);
    (*vertices)[13].set(0.0, 1.0, 1.0);
    (*vertices)[14].set(0.0, 0.0, 1.0);
    (*vertices)[15].set(0.0, 0.0, 0.0);

    (*vertices)[16].set(1.0, 1.0, 0.0);
    (*vertices)[17].set(1.0, 1.0, 1.0);
    (*vertices)[18].set(0.0, 1.0, 1.0);
    (*vertices)[19].set(0.0, 1.0, 0.0);

    (*vertices)[20].set(1.0, 0.0, 0.0);
    (*vertices)[21].set(1.0, 0.0, 1.0);
    (*vertices)[22].set(1.0, 1.0, 1.0);
    (*vertices)[23].set(1.0, 1.0, 0.0);

    boxgeom->setVertexArray(vertices);

    osg::Vec4Array* colors = new osg::Vec4Array(1);
    (*colors)[0].set(1.0, 0.0, 0.0, 1.0);

    boxgeom->setColorArray(colors);
    boxgeom->setColorBinding(osg::Geometry::BIND_OVERALL);

    boxgeom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUADS,0,24));
    boxgeom->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    
    osg::PolygonMode* polyModeObj = new osg::PolygonMode();
    polyModeObj->setMode(  osg::PolygonMode::FRONT_AND_BACK, osg::PolygonMode::LINE );
    boxgeom->getOrCreateStateSet()->setAttribute( polyModeObj );

    geode->setNodeMask(geode->getNodeMask() & ~INTERSECT_MASK);
    geode->addDrawable(boxgeom);
    mattrans->addChild(geode);

    info->sp = new osgVolume::SwitchProperty;
    info->sp->setActiveProperty(0);
    
    // create scene object
    SceneObject* so = new SceneObject(name, false, false, false, true, false);
    PluginHelper::registerSceneObject(so,"Volume");
  
    osgVolume::AlphaFuncProperty* ap = new osgVolume::AlphaFuncProperty(alphaFunc);
    osgVolume::SampleDensityProperty* sd = new osgVolume::SampleDensityProperty(0.005);
    osgVolume::SampleDensityWhenMovingProperty* sdwm = sampleDensityWhenMoving!=0.0 ? new osgVolume::SampleDensityWhenMovingProperty(sampleDensityWhenMoving) : 0;
   
    // temp holds before elements are placed in maps 
    MenuCheckbox* mcb = NULL;

    MenuRangeValue* mrvt = new MenuRangeValue("TransparencyValue", 0.0, 1.0, 1.0);
    mrvt->setCallback(this);
    info->tp = new osgVolume::TransparencyProperty(mrvt->getValue());
    info->tfp = (info->tf.valid()) ? new osgVolume::TransferFunctionProperty(info->tf) : 0;

    MenuRangeValue* mrvi = new MenuRangeValue("IsoValue", 0.0, 1.0, 0.5);
    mrvi->setCallback(this);

    MenuRangeValue* mrvib = new MenuRangeValue("IsoBandValue", 0.0, 1.0, 0.4);
    mrvib->setCallback(this);

    {
        // Standard
        osgVolume::CompositeProperty* cp = new osgVolume::CompositeProperty;
        cp->addProperty(ap);
        cp->addProperty(sd);
        cp->addProperty(info->tp);
        if (sdwm) cp->addProperty(sdwm);
        //if (info->tfp) cp->addProperty(info->tfp);

        info->sp->addProperty(cp);

        mcb = new MenuCheckbox("Standard", true);
        mcb->setCallback(this);
        so->addMenuItem(mcb);
        _standardMap[so] = mcb;
    }

    {
        // Light
        osgVolume::CompositeProperty* cp = new osgVolume::CompositeProperty;
        cp->addProperty(ap);
        cp->addProperty(sd);
        cp->addProperty(info->tp);
        cp->addProperty(new osgVolume::LightingProperty);
        if (sdwm) cp->addProperty(sdwm);
        //if (info->tfp) cp->addProperty(info->tfp);

        info->sp->addProperty(cp);

        mcb = new MenuCheckbox("Light", false);
        mcb->setCallback(this);
        so->addMenuItem(mcb);
        _lightMap[so] = mcb;
    }

    {
        // Isosurface
        osgVolume::CompositeProperty* cp = new osgVolume::CompositeProperty;
        cp->addProperty(sd);
        cp->addProperty(info->tp);
        info->isp = new osgVolume::IsoSurfaceProperty(mrvi->getValue());
        cp->addProperty(info->isp);
        if (sdwm) cp->addProperty(sdwm);
        //if (info->tfp) cp->addProperty(info->tfp);

        info->sp->addProperty(cp);

        mcb = new MenuCheckbox("IsoSurface", false);
        mcb->setCallback(this);
        so->addMenuItem(mcb);
        _isosurfaceMap[so] = mcb;

    }

    {
        // MaximumIntensityProjection
        osgVolume::CompositeProperty* cp = new osgVolume::CompositeProperty;
        cp->addProperty(ap);
        cp->addProperty(sd);
        cp->addProperty(info->tp);
        cp->addProperty(new osgVolume::MaximumIntensityProjectionProperty);
        if (sdwm) cp->addProperty(sdwm);
        //if (info->tfp) cp->addProperty(info->tfp);

        info->sp->addProperty(cp);

        mcb = new MenuCheckbox("MaxIntensity", false);
        mcb->setCallback(this);
        so->addMenuItem(mcb);
        _maxintensityMap[so] = mcb;
    }

    // add isovalue to menu
    so->addMenuItem(mrvi);
    _isosurfaceValueMap[so] = mrvi;
    
    // add transparency
    so->addMenuItem(mrvt);
    _transparencyValueMap[so] = mrvt;

    // create submenu for adjusting transfer function
    SubMenu* transfer = new SubMenu("Transfer Functions");
    so->addMenuItem(transfer);

    MenuCheckbox* mtfcb = new MenuCheckbox("Enable", false);
    mtfcb->setCallback(this);
    transfer->addItem(mtfcb);
    _transferFuncMap[so] = mtfcb;
    
    MenuCheckbox* mtdcb = new MenuCheckbox("Default", true);
    mtdcb->setCallback(this);
    transfer->addItem(mtdcb);
    _transferDefaultMap[so] = mtdcb;

    MenuCheckbox* mctbb = new MenuCheckbox("Bright", false);
    mctbb->setCallback(this);
    transfer->addItem(mctbb);
    _transferBrightMap[so] =  mctbb;

    MenuCheckbox* mcthb = new MenuCheckbox("Hue", false);
    mcthb->setCallback(this);
    transfer->addItem(mcthb);
    _transferHueMap[so] = mcthb;

    MenuCheckbox* mctgb = new MenuCheckbox("Gray", false);
    mctgb->setCallback(this);
    transfer->addItem(mctgb);
    _transferGrayMap[so] = mctgb;

    MenuRangeValue* mrvp = new MenuRangeValue("Position", 0.0, 1.0, info->center);
    mrvp->setCallback(this);
    transfer->addItem(mrvp);
    _transferPositionMap[so] = mrvp;

    MenuRangeValue* mrvbw = new MenuRangeValue("Base Width", 0.0, 2.0, info->width);
    mrvbw->setCallback(this);
    transfer->addItem(mrvbw);
    _transferBaseWidthMap[so] = mrvbw;

    // add animation menu if applicable
    if( info->numberOfFrames > 1 )
    {
        SubMenu* animation = new SubMenu("Animation");
        so->addMenuItem(animation);

        MenuCheckbox* mcpb = new MenuCheckbox("Play", false);
        mcpb->setCallback(this);
        animation->addItem(mcpb);
        _playMap[so] = mcpb;

        MenuRangeValue* mrvsb = new MenuRangeValue("Speed", 0.0, 10.0, 0.4);
        mrvsb->setCallback(this);
        animation->addItem(mrvsb);
        _speedMap[so] = mrvsb;
        
        MenuRangeValue* mrvfb = new MenuRangeValue("Frame", 0.0, info->numberOfFrames, 0.0);
        mrvfb->setCallback(this);
        animation->addItem(mrvfb);
        _frameMap[so] = mrvfb;
    }

    // add delete button
    MenuButton* mb = new MenuButton("Delete");
    mb->setCallback(this);
    so->addMenuItem(mb);
    _deleteMap[so] = mb;

    layer->addProperty(info->sp);

    info->tile->setVolumeTechnique(new osgVolume::RayTracedTechnique);

    so->addChild(mattrans);
    so->addChild(volume);
    so->attachToScene();
    so->setNavigationOn(true);
    so->addMoveMenuItem();
    so->addNavigationMenuItem();
    so->addScaleMenuItem("Scale", 0.1, 10.0, 1.0);

    _volumeMap[so] = info;

    return true;
}

void Volume::menuCallback(MenuItem* menuItem)
{
    //check default button menus
    for(std::map<SceneObject*,MenuCheckbox*>::iterator it = _standardMap.begin(); it != _standardMap.end(); it++)
    {
        if(menuItem == it->second)
        {
            // only used when clicked
            if(it->second->getValue())
            {
                volumeinfo* info = _volumeMap[it->first];
                info->sp->setActiveProperty(0);
                info->tile->setDirty(true);
                info->tile->init();

                // disable other buttons
                cvr::MenuCheckbox* mcb = _lightMap[it->first];
                mcb->setValue(false);
                mcb = _isosurfaceMap[it->first];
                mcb->setValue(false);
                mcb = _maxintensityMap[it->first];
                mcb->setValue(false);
            }
            else
            {
                it->second->setValue(true);
            }
            return;
        }
    }

    for(std::map<SceneObject*,MenuCheckbox*>::iterator it = _lightMap.begin(); it != _lightMap.end(); it++)
    {
        if(menuItem == it->second)
        {
            // only used when clicked
            if(it->second->getValue())
            {
                volumeinfo* info = _volumeMap[it->first];
                info->sp->setActiveProperty(1);
                info->tile->setDirty(true);
                info->tile->init();

                // disable other buttons
                cvr::MenuCheckbox* mcb = _standardMap[it->first];
                mcb->setValue(false);
                mcb = _isosurfaceMap[it->first];
                mcb->setValue(false);
                mcb = _maxintensityMap[it->first];
                mcb->setValue(false);
            }
            else
            {
                it->second->setValue(true);
            }
            return;
        }
    }

    for(std::map<SceneObject*,MenuCheckbox*>::iterator it = _isosurfaceMap.begin(); it != _isosurfaceMap.end(); it++)
    {
        if(menuItem == it->second)
        {
            // only used when clicked
            if(it->second->getValue())
            {
                volumeinfo* info = _volumeMap[it->first];
                info->sp->setActiveProperty(2);
                info->tile->setDirty(true);
                info->tile->init();

                // disable other buttons
                cvr::MenuCheckbox* mcb = _standardMap[it->first];
                mcb->setValue(false);
                mcb = _lightMap[it->first];
                mcb->setValue(false);
                mcb = _maxintensityMap[it->first];
                mcb->setValue(false);
            }
            else
            {
                it->second->setValue(true);
            }
            return;
        } 
    }

    for(std::map<SceneObject*,MenuCheckbox*>::iterator it = _maxintensityMap.begin(); it != _maxintensityMap.end(); it++)
    {
        if(menuItem == it->second)
        {
            // only used when clicked
            if(it->second->getValue())
            {
                volumeinfo* info = _volumeMap[it->first];
                info->sp->setActiveProperty(3);
                info->tile->setDirty(true);
                info->tile->init();

                // disable other buttons
                cvr::MenuCheckbox* mcb = _standardMap[it->first];
                mcb->setValue(false);
                mcb = _lightMap[it->first];
                mcb->setValue(false);
                mcb = _isosurfaceMap[it->first];
                mcb->setValue(false);
            }
            else
            {
                it->second->setValue(true);
            }
            return;
        }
    }

    for(std::map<SceneObject*,MenuRangeValue*>::iterator it = _isosurfaceValueMap.begin(); it != _isosurfaceValueMap.end(); it++)
    {
        if(menuItem == it->second)
        {
            volumeinfo* info = _volumeMap[it->first];
            info->isp->setValue(it->second->getValue());
            return;
        }
    }

    for(std::map<SceneObject*,MenuRangeValue*>::iterator it = _transparencyValueMap.begin(); it != _transparencyValueMap.end(); it++)
    {
        if(menuItem == it->second)
        {
            volumeinfo* info = _volumeMap[it->first];
            info->tp->setValue(it->second->getValue());
            return;
        }
    }

    for(std::map<SceneObject*,MenuCheckbox*>::iterator it = _transferFuncMap.begin(); it != _transferFuncMap.end(); it++)
    {
        if(menuItem == it->second)
        {
            volumeinfo* info = _volumeMap[it->first];

            // check if a transferFunc exists
            if( !info->tfp.valid() )
                return;

            for(int i = 0; i < info->sp->getNumProperties(); i++)
            {
                osgVolume::CompositeProperty* cp = dynamic_cast<osgVolume::CompositeProperty*> (info->sp->getProperty(i));
                if( cp )
                {
                    if( it->second->getValue() )
                    {
                        // check if transfer property is attached
                        bool test = true;
                        for(int j = 0; j < cp->getNumProperties(); j++)
                        {
                            if( cp->getProperty(j) == info->tfp.get() )
                                test = false;
                        }

                        // add transfer func
                        if( test )
                        {
                            cp->addProperty(info->tfp);
                            info->tile->setDirty(true);
                            info->tile->init();
                        }
                    }
                    else
                    {
                        for(int j = 0; j < cp->getNumProperties(); j++)
                        {
                            if( cp->getProperty(j) == info->tfp.get() )
                            {
                                cp->removeProperty(j);
                                info->tile->setDirty(true);
                                info->tile->init();
                                break;
                            }
                        }
                    }
                }   
            } 
            return;
        }
    }

    //compute and set transfer function
    for(std::map<SceneObject*,MenuCheckbox*>::iterator it = _transferDefaultMap.begin(); it != _transferDefaultMap.end(); it++)
    {
        if(menuItem == it->second)
        {
            volumeinfo* info = _volumeMap[it->first];
            osg::TransferFunction1D::ColorMap* colorMap = NULL;

            // only used when clicked
            if(it->second->getValue())
            {
                // first value is color palete, second position, third base width
                colorMap = computeColorMap(info->defaultTransferFunc, _transferPositionMap[it->first]->getValue(), _transferBaseWidthMap[it->first]->getValue());

                // disable other buttons
                cvr::MenuCheckbox* mcb = _transferBrightMap[it->first];
                mcb->setValue(false);
                mcb = _transferHueMap[it->first];
                mcb->setValue(false);
                mcb = _transferGrayMap[it->first];
                mcb->setValue(false);
            
                // set the transferFunction
                setTransferFunction(info, colorMap);
                info->tile->setDirty(true);
                info->tile->init();
            }
            else
            {
                it->second->setValue(true);
            }
        }
    }


    for(std::map<SceneObject*,MenuCheckbox*>::iterator it = _transferGrayMap.begin(); it != _transferGrayMap.end(); it++)
    {
        if(menuItem == it->second)
        {
            volumeinfo* info = _volumeMap[it->first];
            osg::TransferFunction1D::ColorMap* colorMap = NULL;

            // only used when clicked
            if(it->second->getValue())
            {
                // first value is color palete, second position, third base width
                colorMap = computeColorMap(2, _transferPositionMap[it->first]->getValue(), _transferBaseWidthMap[it->first]->getValue());

                // disable other buttons
                cvr::MenuCheckbox* mcb = _transferBrightMap[it->first];
                mcb->setValue(false);
                mcb = _transferHueMap[it->first];
                mcb->setValue(false);
                mcb = _transferDefaultMap[it->first];
                mcb->setValue(false);
            
                // set the transferFunction
                setTransferFunction(info, colorMap);
                info->tile->setDirty(true);
                info->tile->init();
            }
            else
            {
                it->second->setValue(true);
            }
        }
    }

    for(std::map<SceneObject*,MenuCheckbox*>::iterator it = _transferBrightMap.begin(); it != _transferBrightMap.end(); it++)
    {
        if(menuItem == it->second)
        {
            volumeinfo* info = _volumeMap[it->first];
            osg::TransferFunction1D::ColorMap* colorMap = NULL;

            // only used when clicked
            if(it->second->getValue())
            {
                // first value is color palete, second position, third base width
                colorMap = computeColorMap(0, _transferPositionMap[it->first]->getValue(), _transferBaseWidthMap[it->first]->getValue());

                // disable other buttons
                cvr::MenuCheckbox* mcb = _transferGrayMap[it->first];
                mcb->setValue(false);
                mcb = _transferHueMap[it->first];
                mcb->setValue(false);
                mcb = _transferDefaultMap[it->first];
                mcb->setValue(false);
            
                // set the transferFunction
                setTransferFunction(info, colorMap);
                info->tile->setDirty(true);
                info->tile->init();
            }
            else
            {
                it->second->setValue(true);
            }
        }
    }

    for(std::map<SceneObject*,MenuCheckbox*>::iterator it = _transferHueMap.begin(); it != _transferHueMap.end(); it++)
    {
        if(menuItem == it->second)
        {
            volumeinfo* info = _volumeMap[it->first];
            osg::TransferFunction1D::ColorMap* colorMap = NULL;

            // only used when clicked
            if(it->second->getValue())
            {
                // first value is color palete, second position, third base width
                colorMap = computeColorMap(1, _transferPositionMap[it->first]->getValue(), _transferBaseWidthMap[it->first]->getValue());

                // disable other buttons
                cvr::MenuCheckbox* mcb = _transferGrayMap[it->first];
                mcb->setValue(false);
                mcb = _transferBrightMap[it->first];
                mcb->setValue(false);
                mcb = _transferDefaultMap[it->first];
                mcb->setValue(false);
            
                // set the transferFunction
                setTransferFunction(info, colorMap);
                info->tile->setDirty(true);
                info->tile->init();
            }
            else
            {
                it->second->setValue(true);
            }
        }
    }

    // check slider position movement for updates
    for(std::map<SceneObject*,MenuRangeValue*>::iterator it = _transferPositionMap.begin(); it != _transferPositionMap.end(); it++)
    {
        // make sure one of the modes is set
        if(menuItem == it->second && (_transferDefaultMap[it->first]->getValue() || _transferGrayMap[it->first]->getValue() || _transferBrightMap[it->first]->getValue() || _transferHueMap[it->first]->getValue()))
        {
           volumeinfo* info = _volumeMap[it->first];
           
           // find out what color index to use // TODO use default colormap
           int colorIndex = -1;
           if( _transferBrightMap[it->first]->getValue() )
               colorIndex = 0;
           if( _transferHueMap[it->first]->getValue() )
               colorIndex = 1;
           else if( _transferGrayMap[it->first]->getValue() )
               colorIndex = 2;

           // first value is color palete, second position, third base width
           osg::TransferFunction1D::ColorMap* colorMap = NULL;
           if( colorIndex == -1 )
                colorMap = computeColorMap(info->defaultTransferFunc, _transferPositionMap[it->first]->getValue(), _transferBaseWidthMap[it->first]->getValue());
           else
                colorMap = computeColorMap(colorIndex, _transferPositionMap[it->first]->getValue(), _transferBaseWidthMap[it->first]->getValue());
           
           // set the transferFunction
           setTransferFunction(info, colorMap);
           info->tile->setDirty(true);
           info->tile->init();
        }
    }

    for(std::map<SceneObject*,MenuRangeValue*>::iterator it = _transferBaseWidthMap.begin(); it != _transferBaseWidthMap.end(); it++)
    {
        // make sure one of the modes is set
        if(menuItem == it->second && (_transferDefaultMap[it->first]->getValue() || _transferGrayMap[it->first]->getValue() || _transferBrightMap[it->first]->getValue() || _transferHueMap[it->first]->getValue()))
        {
           volumeinfo* info = _volumeMap[it->first];

           // find out what color index to use //TODO use default colormap
           int colorIndex = -1;
           if( _transferBrightMap[it->first]->getValue() )
               colorIndex = 0;
           if( _transferHueMap[it->first]->getValue() )
               colorIndex = 1;
           else if( _transferGrayMap[it->first]->getValue() )
               colorIndex = 2;

           // first value is color palete, second position, third base width
           osg::TransferFunction1D::ColorMap* colorMap = NULL;
           if( colorIndex == -1 )
                colorMap = computeColorMap(info->defaultTransferFunc, _transferPositionMap[it->first]->getValue(), _transferBaseWidthMap[it->first]->getValue());
           else
                colorMap = computeColorMap(colorIndex, _transferPositionMap[it->first]->getValue(), _transferBaseWidthMap[it->first]->getValue());
           
           // set the transferFunction
           setTransferFunction(info, colorMap);
           info->tile->setDirty(true);
           info->tile->init();
        }
    }

    // animation playback 
    for(std::map<SceneObject*,MenuCheckbox*>::iterator it = _playMap.begin(); it != _playMap.end(); it++)
    {
        if(menuItem == it->second)
        {
            if(cvr::ComController::instance()->isMaster())
            {
                volumeinfo* info = _volumeMap[it->first];
                osg::ImageSequence* sequence = dynamic_cast<osg::ImageSequence*> (info->image.get());
                if( sequence )
                {
                    if(it->second->getValue())
                        sequence->play();
                    else
                        sequence->pause();
                }
            }
            return;
        }
    }

    for(std::map<SceneObject*,MenuRangeValue*>::iterator it = _speedMap.begin(); it != _speedMap.end(); it++)
    {
        if(menuItem == it->second)
        {
            if(cvr::ComController::instance()->isMaster())
            {
                volumeinfo* info = _volumeMap[it->first];
                osg::ImageSequence* sequence = dynamic_cast<osg::ImageSequence*> (info->image.get());
                if( sequence )
                {
                    sequence->setLength(it->second->getValue() * info->numberOfFrames);
                }
            }
            return;
        }
    }

    for(std::map<SceneObject*,MenuButton*>::iterator it = _deleteMap.begin(); it != _deleteMap.end(); it++)
    {
        if(menuItem == it->second)
        {
            if(_standardMap.find(it->first) != _standardMap.end())
            {
                delete _standardMap[it->first];
                _standardMap.erase(it->first);
            }

            if(_lightMap.find(it->first) != _lightMap.end())
            {
                delete _lightMap[it->first];
                _lightMap.erase(it->first);
            }

            if(_isosurfaceMap.find(it->first) != _isosurfaceMap.end())
            {
                delete _isosurfaceMap[it->first];
                _isosurfaceMap.erase(it->first);
            }

            if(_maxintensityMap.find(it->first) != _maxintensityMap.end())
            {
                delete _maxintensityMap[it->first];
                _maxintensityMap.erase(it->first);
            }

            if(_isosurfaceValueMap.find(it->first) != _isosurfaceValueMap.end())
            {
                delete _isosurfaceValueMap[it->first];
                _isosurfaceValueMap.erase(it->first);
            }

            if(_transparencyValueMap.find(it->first) != _transparencyValueMap.end())
            {
                delete _transparencyValueMap[it->first];
                _transparencyValueMap.erase(it->first);
            }

            if(_transferFuncMap.find(it->first) != _transferFuncMap.end())
            {
                delete _transferFuncMap[it->first];
                _transferFuncMap.erase(it->first);
            }

            if(_transferPositionMap.find(it->first) != _transferPositionMap.end())
            {
                delete _transferPositionMap[it->first];
                _transferPositionMap.erase(it->first);
            }

            if(_transferPositionMap.find(it->first) != _transferPositionMap.end())
            {
                delete _transferPositionMap[it->first];
                _transferPositionMap.erase(it->first);
            }

            if(_transferBaseWidthMap.find(it->first) != _transferBaseWidthMap.end())
            {
                delete _transferBaseWidthMap[it->first];
                _transferBaseWidthMap.erase(it->first);
            }

            if(_transferBrightMap.find(it->first) != _transferBrightMap.end())
            {
                delete _transferBrightMap[it->first];
                _transferBrightMap.erase(it->first);
            }

            if(_transferHueMap.find(it->first) != _transferHueMap.end())
            {
                delete _transferHueMap[it->first];
                _transferHueMap.erase(it->first);
            }

            if(_transferGrayMap.find(it->first) != _transferGrayMap.end())
            {
                delete _transferGrayMap[it->first];
                _transferGrayMap.erase(it->first);
            }

            //TODO need to remove player controls

            if(_volumeMap.find(it->first) != _volumeMap.end())
            {
                delete _volumeMap[it->first];
                _volumeMap.erase(it->first);
            }

            delete it->first;
            delete it->second;
            _deleteMap.erase(it);

            return;
        }
    }

}

osg::TransferFunction1D::ColorMap* Volume::computeColorMap(vvTransFunc* transfunc, float center, float width)
{
    // if min and max are not 0.0 then add new pyramid widget
    if( center != 0.0 && width != 0.0)
    {
        // remove old pyramidWidgets
        transfunc->deleteWidgets(vvTFWidget::TF_PYRAMID);
        transfunc->_widgets.append(new vvTFPyramid(vvColor(1.0f, 1.0f, 1.0f), false, 1.0f, center, width, 0.0f), vvSLNode<vvTFWidget*>::NORMAL_DELETE);
    }

    //check if transfer function and then convert
    int size = 4096;
    float* rgba = new float[size * 4];
    transfunc->computeTFTexture(size, 1, 1, rgba, 0.0, 1.0);

    osg::TransferFunction1D::ColorMap* colorMap = new osg::TransferFunction1D::ColorMap();

    for(int i = 0; i < size; i++)
    {
        float* base = &(rgba[i * 4]);
        colorMap->insert(std::pair<float, osg::Vec4> ((float)i/size, osg::Vec4(base[0], base[1], base[2], base[3])));
    }

    // clean up temporary array
    delete[] rgba;
    rgba = NULL;

    return colorMap;
}

osg::TransferFunction1D::ColorMap* Volume::computeColorMap(int colorTable, float position, float baseWidth)
{
    if( colorTable < 0 || colorTable > 2)
        return NULL;

    vvTransFunc transfunc;
    transfunc.setDefaultColors(colorTable, 0.0, 1.0);
    transfunc._widgets.append(new vvTFPyramid(vvColor(1.0f, 1.0f, 1.0f), false, 1.0f, position, baseWidth, 0.0f), vvSLNode<vvTFWidget*>::NORMAL_DELETE);
   
    osg::TransferFunction1D::ColorMap* colorMap = computeColorMap(&transfunc, position - (baseWidth * 0.5), position + (baseWidth * 0.5));
    
    // clean up transfunc memory
    transfunc.deleteWidgets(vvTFWidget::TF_PYRAMID);
    transfunc.deleteWidgets(vvTFWidget::TF_COLOR);
    
    return colorMap;
}

void Volume::setTransferFunction(volumeinfo* info, osg::TransferFunction1D::ColorMap* colorMap)
{
    // if no colorMap attach set deafult if exists
    if( info->tfp.valid() )
    {
        osg::TransferFunction1D* trans1d = dynamic_cast<osg::TransferFunction1D*> (info->tfp->getTransferFunction());
        if( trans1d )
        {
            // check if adding a colorMap
            if( colorMap )
            {
                trans1d->assign(*colorMap);

                // remove colorMap after assignment
                delete colorMap;
            }
        }
    }
    info->tile->setDirty(true);
    info->tile->init();
}

/*
bool Volume::processEvent(InteractionEvent * ie)
{
    TrackedButtonInteractionEvent * tie = ie->asTrackedButtonEvent();

    if(tie)
    {
        if(_eventActive && _activeHand != tie->getHand())
        {
            return false;
        }

        if(_clipBoxMenuItem->getValue() && tie->getButton() == _moveButton)
        {
            if(tie->getInteraction() == BUTTON_DOWN)
            {
                _lastHandInv = osg::Matrix::inverse(tie->getTransform());
                _lastHandMat = tie->getTransform();
                _lastobj2world = getObjectToWorldMatrix();
                _eventActive = true;
                _moving = true;
                _activeHand = tie->getHand();
                return true;
            }
            else if(_moving
                    && (tie->getInteraction() == BUTTON_DRAG
                            || tie->getInteraction() == BUTTON_UP))
            {
                processMove(tie->getTransform());
                if(tie->getInteraction() == BUTTON_UP)
                {
                    _eventActive = false;
                    _moving = false;
                   isoactiveHand = -2;
                }
                return true;
            }
        }
    }
    return false;
}
*/

void Volume::preFrame()
{
/*
    if( _root )
    {

	//check if test for intersection with frame
	if(_clipBoxMenuItem->getValue())
	{
		osg::Vec3 pStart(0,0,0);
            	osg::Vec3 pEnd(0,100000,0);
            	pStart = pStart * TrackingManager::instance()->getHandMat(0);
                pEnd = pEnd * TrackingManager::instance()->getHandMat(0);
                std::vector<IsectInfo> results = getObjectIntersection(_clipBox, pStart, pEnd);
		if( results.size() )
		{
			osg::Vec3 point= results[0].point;	
			printf("Point intersection %f %f %f\n", point.x(), point.y(), point.z());
		}
	}

    }
*/

    // exit preframe if no volumes
    if( ! _volumeMap.size() )
        return;

    // create and id lookup map
    std::map<int, animationinfo > timeLookup;

    // send all animation data
    std::map<cvr::SceneObject*,volumeinfo*>::iterator it = _volumeMap.begin();
    for(; it != _volumeMap.end(); ++it)
    {
        animationinfo dataPacket;

        cvrImageSequence* sequence = dynamic_cast<cvrImageSequence*> (it->second->image.get());
        if( sequence )
        {
            if(cvr::ComController::instance()->isMaster())
            {
                dataPacket.id = it->second->id;
                dataPacket.time = sequence->getCurrentTime();
                dataPacket.frame = sequence->getFrame(dataPacket.time); 
                //std::cerr << "Current time " << dataPacket.time << "  and frame " << dataPacket.frame << std::endl;
                cvr::ComController::instance()->sendSlaves(&dataPacket,sizeof(dataPacket));
            }
            else
            {
                cvr::ComController::instance()->readMaster(&dataPacket,sizeof(dataPacket));
            }
            timeLookup[dataPacket.id] = dataPacket;
        }
    }

    // update slaves
    if(!cvr::ComController::instance()->isMaster())
    {
        // update all the seek locations for the image sequences
        for(it = _volumeMap.begin(); it != _volumeMap.end(); ++it)
        {
            osg::ImageSequence* sequence = dynamic_cast<osg::ImageSequence*> (it->second->image.get());
            if( sequence )
            {
                if(timeLookup.find(it->second->id) != timeLookup.end())
                {
                    animationinfo animinfo = timeLookup[it->second->id];
                    sequence->seek(animinfo.time);
                }

            }
        }
    }

    // update slider
    for(it = _volumeMap.begin(); it != _volumeMap.end(); ++it)
    {
        // update slider
        if(_frameMap.find(it->first) != _frameMap.end())
        {
            animationinfo animinfo = timeLookup[it->second->id];
            _frameMap[it->first]->setValue(animinfo.frame);
        }
    }
}

bool Volume::init()
{
    std::cerr << "Volume init\n";
    //osg::setNotifyLevel( osg::INFO );
    return true;
}


Volume::~Volume()
{
   printf("Called Volume destructor\n");
}
