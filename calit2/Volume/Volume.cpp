#include "Volume.h"

#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/SceneManager.h>
#include <cvrKernel/PluginManager.h>
#include <cvrKernel/PluginHelper.h>
#include <cvrKernel/NodeMask.h>
#include <cvrMenu/MenuSystem.h>
#include <cvrKernel/ComController.h>
#include <cvrKernel/ScreenConfig.h>
#include <cvrKernel/InteractionEvent.h>
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

typedef itk::Image<float, 3> ImageType;
typedef itk::GradientAnisotropicDiffusionImageFilter<ImageType, ImageType>  DiffusionFilterType;

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
    //osg::ref_ptr<osg::ImageSequence> imageSequence = new osg::ImageSequence;
    osg::ref_ptr<cvrImageSequence> imageSequence = new cvrImageSequence;
    imageSequence->setLength(vol->getStoredFrames());
    imageSequence->setLoopingMode( osg::ImageStream::LOOPING );
    imageSequence->setLength(vol->getStoredFrames() * 0.4f);
    imageSequence->pause();

    //std::cerr << "Number of frames found " << vol->getStoredFrames() << std::endl;

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
    
    //float minValue, maxValue;
    //vol->findMinMax(0, minValue, maxValue);
    //minValue = convertValue(volinfo->datatype, (char*)&minValue); 
    //maxValue = convertValue(volinfo->datatype, (char*)&maxValue); 

    //std::cerr << "Osg Max " << maxValue[0] << " min " << minValue[0] << std::endl;
    //std::cerr << "Osg Max " << max[0] << " min " << min[0] << std::endl;

    float scale = 0.99f/(maxValue[0]-minValue[0]);
    float offset = -minValue[0] * scale;

    //std::cerr << "Scale " << scale << " offset " << offset << std::endl;

    // create a transferfunction
	volinfo->tf = new osg::TransferFunction1D();

    // create colorMap (access widget and get center and min max ranges)
    // Looks only for PyramidWidgets
    int size = vol->tf._widgets.count();
    vol->tf._widgets.first();

    // temporary transferfunction table
    volinfo->defaultTransferFunc = new vvTransFunc();

    for(int i = 0; i < size; i++)
    {
        vvTFWidget* w = vol->tf._widgets.getData();

        if( vvTFPyramid *pw = dynamic_cast<vvTFPyramid*>(w) )
        {
            volinfo->center = (pw->_pos[0] / scale) - offset;
            volinfo->width = pw->_bottom[0] / scale;
            volinfo->defaultTransferFunc->_widgets.append(new vvTFPyramid(vvColor(1.0f, 1.0f, 1.0f), false, 1.0f, volinfo->center, volinfo->width, 0.0f),  vvSLNode<vvTFWidget*>::NORMAL_DELETE);
        }

        // check color pin values
        if( vvTFColor *cw = dynamic_cast<vvTFColor*>(w) )
        {
            float centerValue = (cw->_pos[0] / scale) - offset;
            //std::cerr << "Color position " << centerValue << " initial " << cw->_pos[0] << std::endl;
            osg::Vec3f color(0.0, 0.0, 0.0);
            cw->_col.getRGB(color.x(), color.y(), color.z());
            volinfo->defaultTransferFunc->_widgets.append(new vvTFColor(vvColor(color.x(), color.y(), color.z()), centerValue),  vvSLNode<vvTFWidget*>::NORMAL_DELETE);
        }

        vol->tf._widgets.next();
    }

    //std::cerr << "Center " << volinfo->center << " width " << volinfo->width << std::endl;
        
    // add colormap to transfer function
	volinfo->tf->assign(*(computeColorMap(volinfo->defaultTransferFunc, volinfo->center, volinfo->width)));

    // remove voldesc object
    if( vol )
        delete vol;
    vol = NULL;

    return volinfo;
}

struct Volume::volumeinfo* Volume::loadVol(std::string filename, int &x, int &y, int &z, float &xMultiplier, float &yMultiplier, float &zMultiplier, 
                                            int& sizeS, int& sizeT, int& sizeR, int& numberBytesPerComponent, std::string& name)
{
    struct volumeinfo* volinfo = NULL;

    // parse vol file for information
    ifstream cfile;
    cfile.open(filename.c_str(), ios::in);

    if(!cfile.fail())
    {
        string line;
        while(!cfile.eof())
        {
            cfile >> name;
            if(cfile.eof())
            {
                break;
            }
	        cfile >> x;
	        cfile >> y;
	        cfile >> z;
            cfile >> numberBytesPerComponent;
            cfile >> xMultiplier;
            cfile >> yMultiplier;
            cfile >> zMultiplier;
        }
        cfile.close();

	    volinfo = new struct volumeinfo;

	    // allocate volume
	    GLenum dataType;
	    osg::Image* image = allocateVolume(x, y, z, sizeS, sizeT, sizeR, numberBytesPerComponent, dataType);
 
        // read in volume data
        std::string path = osgDB::getFilePath(filename);
        std::string rawFile = osgDB::concatPaths(path, name);

        // if vol read in texture data
        FILE* fin = fopen( rawFile.c_str() , "rb" );
        if (!fin)
        {
            std::cerr << "Failed reading rawFile\n";
            delete volinfo;
            return NULL;
        }
    
        unsigned int r_offset = (z < sizeR) ? sizeR/2 - z/2 : 0;

        // read in data
        for(int r=0;r<z;++r)
        {
            for(int t=0;t<y;++t)
            {
                char* data = (char*) image->data(0,t,r+r_offset);
                fread(data,numberBytesPerComponent,x,fin);
            }
        }
        fclose( fin );

        // create an image sequence
        //osg::ref_ptr<osg::ImageSequence> imageSequence = new osg::ImageSequence;
        osg::ref_ptr<cvrImageSequence> imageSequence = new cvrImageSequence;
        imageSequence->setLength(1);
        imageSequence->setLoopingMode( osg::ImageStream::LOOPING );
        imageSequence->setLength(0.1f);
        imageSequence->addImage(image);

        volinfo->image = imageSequence.get();
        volinfo->id = id++;
        volinfo->numberOfFrames = 1;
        //volinfo->timer = NULL;
    }
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

void Volume::createPlyPoints(volumeinfo * info, osg::Vec3 position, float lowerBound, float upperBound, float voxelSmoothScalar)
{
    ofstream ofs("surfaceTest.vtk");
    ofs << "# vtk DataFile Version 2.0\n";
    ofs << "Volume example\n";
    ofs << "ASCII\n";
    ofs << "DATASET STRUCTURED_POINTS\n";
    ofs << "DIMENSIONS " << info->x << " " << info->y << " " << info->z << "\n";
    ofs << "ORIGIN 0 0 0\n";
    ofs << "ASPECT_RATIO " << info->xMultiplier << " " << info->yMultiplier << " " << info->zMultiplier << "\n";
    ofs << "POINT_DATA " << (info->x * info->y * info->z) << "\n";
    ofs << "SCALARS volume_scalars float\n";
    ofs << "LOOKUP_TABLE default\n";

    osg::ref_ptr<osg::Vec3Array> vertices = new osg::Vec3Array();
    osg::ref_ptr<osg::Vec4Array> colors = new osg::Vec4Array();
    createPoints(info, vertices, colors, position, lowerBound, upperBound);
    
    std::cerr << "Initial number of points " << vertices->size() << std::endl;

    // temporary map to use to data look up
    std::map<osg::Vec3, float> boundPoints;

    // load vertices into a map
    for(int i = 0; i < vertices->size(); i++)
    {
        boundPoints[vertices->at(i)] = colors->at(i)[0];
    }
    
    // fill in voxels
    fillInVolume(info, boundPoints);

    // average surrounding voxels (currently just over all averaging)
    std::map<osg::Vec3, float>::iterator it = boundPoints.begin();
    for(; it != boundPoints.end(); ++it)
    {
       it->second = averageSurroundingVoxels(info, boundPoints, it->first, 0.0, 1, 1, 1, true); 
    } 

    // call recursize voxel smoothing function
    //smoothVoxelData(info, boundPoints, voxelSmoothScalar);

    // write out the rest of the ply file
    for(int i = 0; i < info->z; i++)
    {
	    for(int j = 0; j < info->y; j++)
	    {
	        for(int k =0; k < info->x; k++)
	        {
                osg::Vec3 point(k, j, i);

		        if( boundPoints.find(point) != boundPoints.end() )
                {
		            ofs << boundPoints.find(point)->second;
		            ofs << "\n";
                }
		        else
                {
                    ofs << "0.0\n";

                    // try averaging surround voxels (smoothing) then add to map
                    //float computedValue = averageSurroundingVoxels(info, boundPoints, point, 0.35, 1, 1, 1);
                    //if( computedValue != 0.0 )
                    //    boundPoints[point] = computedValue;

		            //ofs << computedValue;
		            //ofs << "\n";
                }
 	        }
	    }
    }
    ofs.close();

    std::cerr << "Overall number of Points after averaging " << boundPoints.size() << std::endl;
}

void Volume::diffuseVolume(volumeinfo* info)
{
    ImageType::Pointer image = ImageType::New();
    ImageType::SizeType size;
    size[0] = info->x; // size along X
    size[1] = info->y; // size along Y
    size[2] = info->z; // size along Z

    ImageType::IndexType start;
    start[0] = 0; // first index on X
    start[1] = 0; // first index on Y
    start[2] = 0; // first index on Z

    ImageType::RegionType region;
    region.SetSize(size);
    region.SetIndex(start);
    image->SetRegions( region );

    float spacing[ ImageType::ImageDimension ];
    spacing[0] = info->xMultiplier; // spacing in mm along X
    spacing[1] = info->yMultiplier; // spacing in mm along Y
    spacing[2] = info->zMultiplier; // spacing in mm along Z
    image->SetSpacing( spacing );
    image->Allocate();
    image->FillBuffer(0);

    // accessing pixels
    ImageType::PixelType pixelValue;

    itk::ImageRegionIterator<ImageType> inputImageIterator(image, region);

    for(int i = 0; i < info->z; i++)
    {
        for(int j = 0; j < info->y; j++)
        {
            for(int k = 0; k < info->x; k++)
            {
                char* data = (char*) info->image->data(k,j,i + info->r_offset);
                pixelValue = convertValue(info->datatype, (char *)data); 
                inputImageIterator.Set(pixelValue);
                ++inputImageIterator;
            }
        }
    }

    DiffusionFilterType::Pointer diffusion = DiffusionFilterType::New();
    diffusion->SetNumberOfIterations( 10 );
    diffusion->SetConductanceParameter( 2.0 );
    diffusion->SetTimeStep(0.045);
    diffusion->SetInput(image);
    diffusion->Update();
    
    itk::ImageRegionIterator<ImageType> outputImageIterator(diffusion->GetOutput(), region);

    // replace image values in original data
    for(int i = 0; i < info->z; i++)
    {
        for(int j = 0; j < info->y; j++)
        {
            for(int k = 0; k < info->x; k++)
            {
                pixelValue = outputImageIterator.Get();
                char* data = (char*) info->image->data(k,j,i + info->r_offset);
                insertValue(info->datatype, data, pixelValue);
                ++outputImageIterator;
            }
        }
    }
}

void Volume::smoothVoxelData(volumeinfo* info, std::map<osg::Vec3, float> & boundPoints, float percentageSurrounding)
{
    std::queue<osg::Vec3> testVoxels;

    // do initial pass and add data to modifiedVoxels
	for(int i = 0; i < info->z; i++)
    {
	    for(int j = 0; j < info->y; j++)
	    {
	        for(int k =0; k < info->x; k++)
	        {
                osg::Vec3 point(k, j, i);

		        if( boundPoints.find(point) == boundPoints.end() )
                {
                    // try averaging surround voxels (smoothing) then add to map
                    float computedValue = averageSurroundingVoxels(info, boundPoints, point, percentageSurrounding, 1, 1, 1);
                    if( computedValue != 0.0 )
					{
                        boundPoints[point] = computedValue;
						findSurroundingEmptyVoxels(info, boundPoints, point, testVoxels);
		            }
                }
 	        }
	    }
    }

	std::cerr << "Total initial num of test voxels: " << testVoxels.size() << std::endl;
   
    int counter = 0;
    
    // loop through modified Voxels checking average and add if required
	while( !testVoxels.empty() )
	{
		osg::Vec3 point = testVoxels.front();
		testVoxels.pop();

		float computedValue = averageSurroundingVoxels(info, boundPoints, point, percentageSurrounding, 1, 1, 1);
        if( computedValue != 0.0 )
		{
        	boundPoints[point] = computedValue;
			findSurroundingEmptyVoxels(info, boundPoints, point, testVoxels);
            counter++;
		}
	}

    std::cerr << "Finished smoothing, number of times through loop: " << counter << std::endl;
}

void Volume::findSurroundingEmptyVoxels(volumeinfo* info, std::map<osg::Vec3, float> & boundPoints, osg::Vec3 point, std::queue<osg::Vec3> & emptyVoxels)
{
   // location params
    int r, s, t;

    // create boundary vectors
    std::vector<int> zValues;
    std::vector<int> yValues;
    std::vector<int> xValues;

	r = point.z();
	t = point.y();
	s = point.x();

	if((r > 0) > 0)
	    zValues.push_back(-1);
	if(t > 0)
	    yValues.push_back(-1);
	if(s > 0)
	    xValues.push_back(-1);

	zValues.push_back(0);
	yValues.push_back(0);
	xValues.push_back(0);
		
	if(r < info->z - 1)
	    zValues.push_back(1);
	if(t < info->y - 1)
	    yValues.push_back(1);
	if(s < info->x - 1)
	    xValues.push_back(1);
		
	int x, y, z;
    
    // temporary place to hold look up values
	std::vector<float> resultValues;

	for(int i = 0; i < zValues.size(); i++)
	{
	    for(int j = 0; j < yValues.size(); j++)
	    {
			for(int k = 0; k < xValues.size(); k++)
			{
		    	x = s + xValues[k];
		    	y = t + yValues[j];
		    	z = r + zValues[i];

				if( boundPoints.find(osg::Vec3(x, y, z)) == boundPoints.end())
				{
					emptyVoxels.push(osg::Vec3(x, y, z));
				}
			}
		}
	}
}

float Volume::averageSurroundingVoxels(volumeinfo* info, std::map<osg::Vec3, float> & boundPoints, osg::Vec3 point, float percentageSurrounding, int xwidth, int ywidth, int zwidth, bool validVoxels)
{
    // location params
    int r, s, t;

    // create boundary vectors
    std::vector<int> zValues;
    std::vector<int> yValues;
    std::vector<int> xValues;

	r = point.z();
	t = point.y();
	s = point.x();

	if((r > 0) > 0)
	    zValues.push_back(-1);
	if(t > 0)
	    yValues.push_back(-1);
	if(s > 0)
	    xValues.push_back(-1);

	zValues.push_back(0);
	yValues.push_back(0);
	xValues.push_back(0);
		
	if(r < info->z - 1)
	    zValues.push_back(1);
	if(t < info->y - 1)
	    yValues.push_back(1);
	if(s < info->x - 1)
	    xValues.push_back(1);
		
	int x, y, z;
    
    // temporary place to hold look up values
	std::vector<float> resultValues;

	for(int i = 0; i < zValues.size(); i++)
	{
	    for(int j = 0; j < yValues.size(); j++)
	    {
			for(int k = 0; k < xValues.size(); k++)
			{
		    	x = s + xValues[k];
		    	y = t + yValues[j];
		    	z = r + zValues[i];

				if( boundPoints.find(osg::Vec3(x, y, z)) != boundPoints.end())
				{
					resultValues.push_back(boundPoints.find(osg::Vec3(x, y, z))->second);
				}
			}
		}
	}

    // compute average if above/equal to percentage requirement
	if( ((((float)resultValues.size()) / 26) >= percentageSurrounding ) || validVoxels )
	{
		float result = 0.0;
		for(int i = 0; i < resultValues.size();i++)
		{
			result += resultValues.at(i);
		}
		return (result / resultValues.size());
	}

    return 0.0;
}

void Volume::fillInVolume(volumeinfo* info, std::map<osg::Vec3, float> & boundPoints)
{

    std::cerr << "Initial data size " << boundPoints.size() << std::endl;

    std::map<osg::Vec3, float> additionValues;

    // x axis filling
    for(int i = 0; i < info->z; i++)
    {
        for(int j = 0; j < info->y; j++)
        {
            int first = -1;
            int last = -1;
            for(int k = 0; k < info->x; k++)
            {
                // forwards check
                if((first == -1) && (boundPoints.find(osg::Vec3(k, j, i)) != boundPoints.end()))
                {
                    first = k; 
                }

                // backwards check
                if((last == -1) && (boundPoints.find(osg::Vec3((info->x - 1 - k), j, i)) != boundPoints.end()))
                {
                    last = k;
                }

                if( first != -1 && last != -1 )
                    break;
            }

            // loop through and add values to additionsMap
            for(int index = first; index < last; index++)
            {
                char* data = (char*) info->image->data(index,j,i + info->r_offset);
                float value = convertValue(info->datatype, (char *)data);
                additionValues[osg::Vec3(index, j, i)] = value; 
            }
        }
    }

    // y axis filling
    for(int i = 0; i < info->z; i++)
    {
        for(int k = 0; k < info->x; k++)
        {
            int first = -1;
            int last = -1;
            for(int j = 0; j < info->y; j++)
            {
                // forwards check
                if((first == -1) && (boundPoints.find(osg::Vec3(k, j, i)) != boundPoints.end()))
                {
                    first = j; 
                }

                // backwards check
                if((last == -1) && (boundPoints.find(osg::Vec3(k, (info->y - 1 - j), i)) != boundPoints.end()))
                {
                    last = j;
                }

                if( first != -1 && last != -1 )
                    break;
            }

            // loop through and add values to addtionsMap
            for(int index = first; index < last; index++)
            {
                char* data = (char*) info->image->data(k,index,i + info->r_offset);
                float value = convertValue(info->datatype, (char *)data);
                additionValues[osg::Vec3(k, index, i)] = value; 
            }
        }
    }


    //z axis filling
    for(int j = 0; j < info->y; j++)
    {
        for(int k = 0; k < info->x; k++)
        {
            int first = -1;
            int last = -1;
            for(int i = 0; i < info->z; i++)
            {
                // forwards check
                if((first == -1) && (boundPoints.find(osg::Vec3(k, j, i)) != boundPoints.end()))
                {
                    first = i; 
                }

                // backwards check
                if((last == -1) && (boundPoints.find(osg::Vec3(k, j, (info->z - 1 - i))) != boundPoints.end()))
                {
                    last = i;
                }

                if( first != -1 && last != -1 )
                    break;
            }

            // loop through and add values to addtionsMap
            for(int index = first; index < last; index++)
            {
                char* data = (char*) info->image->data(k,j,index + info->r_offset);
                float value = convertValue(info->datatype, (char *)data);
                additionValues[osg::Vec3(k, j, index)] = value; 
            }
        }
    }

    std::cerr << "Adding " << additionValues.size() << " values\n";

    // add new data to map
    std::map<osg::Vec3, float>::iterator it = additionValues.begin();
    for(; it != additionValues.end(); ++it)
    {
        boundPoints[it->first] = it->second;
    }
}


osg::Geode* Volume::createSurface(volumeinfo * info, osg::Vec3 position, float lowerBound, float upperBound, int steps)
{
    osg::ref_ptr<osg::Vec3Array> vertices = new osg::Vec3Array();
    osg::ref_ptr<osg::Vec4Array> colors = new osg::Vec4Array();

    createPoints(info, vertices, colors, position, lowerBound, upperBound, steps);
 
    if (vertices->size() == 0)
	return NULL;

    // add vertices to map
    std::map<osg::Vec3, float> data;
    for(int i = 0; i < vertices->size(); i++)
    {
        data[vertices->at(i)] = colors->at(i)[0];
    }
    
    // fill in voxels
    fillInVolume(info, data);
  
    // create vtk grid for marching cubes 
    vtkStructuredPoints *grid = vtkStructuredPoints::New();
    grid->SetDimensions(info->x, info->y, info->z);
    grid->SetSpacing(info->xMultiplier, info->yMultiplier, info->zMultiplier);
    grid->SetNumberOfScalarComponents(1);
    grid->SetScalarTypeToFloat();

    vtkFloatArray* scalararray = vtkFloatArray::New();
    for(int i = 0; i < (info->z); i++)
    {
        for(int j = 0; j < (info->y); j++)
        {
            for(int k = 0; k < (info->x); k++)
            {
                float value = 0.0;
                if( data.find(osg::Vec3(k, j, i)) != data.end() )
                    value = data.find(osg::Vec3(k, j, i))->second;

                scalararray->InsertNextValue(value); 
            }
        }
    }
   
    grid->GetPointData()->SetScalars(scalararray);

    // run marching cubes against data using (lower bound value)
    vtkMarchingContourFilter* surface = vtkMarchingContourFilter::New();
    surface->SetInput(grid);
    surface->SetValue(0, lowerBound);
    surface->ComputeNormalsOn();
    surface->ComputeGradientsOff();
    surface->Update();
   
    // extract largest region
    vtkPolyDataConnectivityFilter* confilter = vtkPolyDataConnectivityFilter::New();
    confilter->SetInputConnection(surface->GetOutputPort());
    confilter->SetExtractionModeToLargestRegion();
    
    // surface filter
    vtkWindowedSincPolyDataFilter* smoother = vtkWindowedSincPolyDataFilter::New();
    smoother->SetInput(confilter->GetOutput());
    smoother->SetNumberOfIterations(15);
    smoother->BoundarySmoothingOff();
    smoother->FeatureEdgeSmoothingOff();
    smoother->SetFeatureAngle(120.0);
    smoother->SetPassBand(.001);
    smoother->NonManifoldSmoothingOn();
    smoother->NormalizeCoordinatesOn();
    smoother->Update();
    
    vtkPolyDataMapper* surfaceMapper = vtkPolyDataMapper::New();
    surfaceMapper->SetInputConnection(smoother->GetOutputPort());
    surfaceMapper->ScalarVisibilityOff();
    surfaceMapper->Update();

    vtkActor* actor = vtkActor::New();
    actor->SetMapper (surfaceMapper);

    osg::Geode* geode = vtkActorToOSG(actor);

    osg::StateSet* stateset = geode->getOrCreateStateSet();
    osg::Material* material = new osg::Material();
    material->setDiffuse(Material::FRONT_AND_BACK, Vec4(0.8, 0.8, 0.8, 1.0));
    material->setAlpha(Material::FRONT_AND_BACK, 1.0f);
    stateset->setAttributeAndModes(material, StateAttribute::OVERRIDE | StateAttribute::ON);
  
    grid->Delete();
    surface->Delete();
    confilter->Delete();
    scalararray->Delete();
    smoother->Delete();
    surfaceMapper->Delete();
    actor->Delete();

    // clean up resulting mesh
    osgUtil::Simplifier simplifier;
    simplifier.setDoTriStrip(false);
    simplifier.setSmoothing(true);
    simplifier.apply(*geode);

    return geode;
}


osg::Geode* Volume::createPointSet(volumeinfo * info, osg::Vec3 position, float lowerBound, float upperBound, int steps)
{
    osg::Vec3Array* vertices = new osg::Vec3Array();
    osg::Vec4Array* colors = new osg::Vec4Array();

    createPoints(info, vertices, colors, position, lowerBound, upperBound, steps);
    
    // modify vertices to map to volume position
    for(int i =0; i < vertices->size(); i++)
    {
      vertices->at(i)[0] = vertices->at(i)[0] * info->xMultiplier;  
      vertices->at(i)[1] = vertices->at(i)[1] * info->yMultiplier;  
      vertices->at(i)[2] = (vertices->at(i)[2] + info->r_offset) * info->zMultiplier;  
    }   
  
    osg::Geode* geode = new osg::Geode();
    geode->setCullingActive(false);
    osg::Geometry* nodeGeom = new osg::Geometry();
    nodeGeom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::POINTS,0, vertices->size()));
    osg::VertexBufferObject* vboP = nodeGeom->getOrCreateVertexBufferObject();
    vboP->setUsage (GL_STREAM_DRAW);

    osg::StateSet *stateset = nodeGeom->getOrCreateStateSet();
    stateset->setAttribute(pgm1);
    stateset->addUniform(new osg::Uniform("pointScale", 0.5f));
    stateset->addUniform(new osg::Uniform("globalAlpha",1.0f));

    //Material *material = new Material;
    //material->setDiffuse(Material::FRONT_AND_BACK, Vec4(1.0, 0.0, 0.0, 1.0));
    //material->setAlpha(Material::FRONT_AND_BACK, 1.0f);
    //stateset->setAttributeAndModes(material, StateAttribute::OVERRIDE | StateAttribute::ON);
    //stateset->setMode(GL_BLEND, StateAttribute::OVERRIDE | StateAttribute::ON );
    //stateset->setRenderingHint(StateSet::TRANSPARENT_BIN);

    nodeGeom->setUseDisplayList (false);
    nodeGeom->setUseVertexBufferObjects(true);
    nodeGeom->setVertexArray(vertices);
    nodeGeom->setColorArray(colors);
    nodeGeom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);

    geode->addDrawable(nodeGeom);
    geode->dirtyBound();

    return geode;                         
}

void Volume::createXYZPoints(volumeinfo * info, osg::Vec3 position, float lowerBound, float upperBound)
{
    osg::ref_ptr<osg::Vec3Array> vertices = new osg::Vec3Array();
    
    createPoints(info, vertices, NULL, position, lowerBound, upperBound);

    ofstream ofs("test.xyz");
    for(int i = 0; i < (int) vertices->size(); i++)
    {
        osg::Vec3 point = vertices->at(i);
        ofs << point.x();
        ofs << " ";
        ofs << point.y();
        ofs << " ";
        ofs << point.z();
        ofs << "\n";
    }
    ofs.close();
}

void Volume::createPoints(volumeinfo * info, osg::Vec3Array * boundPoints, osg::Vec4Array* boundScalars, osg::Vec3 startPoint, float lowerBound, float upperBound, int maxSteps)
{
    // mark visited voxels
    bool* visited = new bool[info->x * info->y *info->z];
    for(int i = 0; i < (info->x * info->y * info->z); i++)
        visited[i] = false;

    // set initial point as visited (so not to include as an edge)
    visited[(int)startPoint.x() + (int)(startPoint.y() * info->x) + (int)(startPoint.z() * info->x  * info->y)] = true;

    // create queue for values
    std::queue< std::pair<osg::Vec3, int> > activeValues;
    activeValues.push(std::pair<osg::Vec3, int> (startPoint, 0));

    // location params
    int r, s, t;

    // create boundary vectors
    std::vector<int> zValues;
    std::vector<int> yValues;
    std::vector<int> xValues;

    while( !activeValues.empty() )
    {
        std::pair<osg::Vec3, int> step = activeValues.front();
        activeValues.pop();

        osg::Vec3 point = step.first; 
        int currentStep = step.second; // number of steps taken in breathfirst traversal
        r = point.z();
        t = point.y();
        s = point.x();

        if(r > 0)
            zValues.push_back(-1);
        if(t > 0)
            yValues.push_back(-1);
        if(s > 0)
            xValues.push_back(-1);

        zValues.push_back(0);
        yValues.push_back(0);
        xValues.push_back(0);
                
        if(r < info->z - 1)
            zValues.push_back(1);
        if(t < info->y - 1)
            yValues.push_back(1);
        if(s < info->x - 1)
            xValues.push_back(1);
                
        float tempvalue = 0.0;
        int x, y, z, key;

        for(int i = 0; i < zValues.size(); i++)
        {
            for(int j = 0; j < yValues.size(); j++)
            {
                for(int k = 0; k < xValues.size(); k++)
                {
                    x = s + xValues[k];
                    y = t + yValues[j];
                    z = r + zValues[i];

                    char* data = (char*) info->image->data(x,y,z + info->r_offset);
                    tempvalue = convertValue(info->datatype, (char *)data);
  
                    key = x + (y * info->x) + (z * info->x  * info->y);

                    if( !visited[key] && (tempvalue > lowerBound) && (tempvalue < upperBound) )
                    {
                        osg::Vec3 actualPoint(x,y,z);

                        // used to limit traversal length through volume (maxSteps zero means continue until volume completed)
                        if( (maxSteps == 0)  || ( currentStep < maxSteps) )
                            activeValues.push(std::pair<osg::Vec3, int> (actualPoint, currentStep + 1));
                       
                        visited[key] = true;

                        boundPoints->push_back(actualPoint);
                        
                        // add scalar value to red component
                        if( boundScalars != NULL )
                        {
                            boundScalars->push_back(osg::Vec4(tempvalue, 0.0, 0.0, 1.0));
                        }
                    } 
                }
            }
        }

        //clear values
        xValues.clear();
        yValues.clear();
        zValues.clear();
    }

    delete[] visited;
    visited = NULL;
}

void Volume::createPly(volumeinfo * info)
{
    // look at volume and output vertices based on image values
    osg::Geode* geode = new osg::Geode();
    osg::Geometry* geom = new osg::Geometry();
    osg::Vec3Array* vertices = new osg::Vec3Array();
    
    std::vector<int> zValues;
    std::vector<int> yValues;
    std::vector<int> xValues;
    std::vector<float> values;
                
    float band = 0.02;
    float interestValue = 0.3;
    int addedValues = 0;
    int numPoints = 0;

    for(int r = 0; r < info->z; r++)
    {
        for(int t = 0; t < info->y; t++)
        {
            char* data = (char*) info->image->data(0,t,r + info->r_offset);
            for(int s = 0; s < info->x; s++)
            {
                float value = convertValue(info->datatype, data);

                // check surrounding values then insert
                if(r > 0)
                    zValues.push_back(-(s * t * info->numberOfBytesPerComponent));
                if(t > 0)
                    yValues.push_back(-(s * info->numberOfBytesPerComponent));
                if(s > 0)
                    xValues.push_back(-(info->numberOfBytesPerComponent));

                zValues.push_back(0);
                yValues.push_back(0);
                xValues.push_back(0);
                
                if(r < info->z - 1)
                    zValues.push_back((s * t * info->numberOfBytesPerComponent));
                if(t < info->y - 1)
                    yValues.push_back((s * info->numberOfBytesPerComponent));
                if(s < info->x - 1)
                    xValues.push_back(info->numberOfBytesPerComponent);

                for(int i = 0; i < zValues.size(); i++)
                {
                    for(int j = 0; j < yValues.size(); j++)
                    {
                        for(int k = 0; k < xValues.size(); k++)
                        {
                            float tempValue = convertValue(info->datatype, (char *)(data + zValues[i] + yValues[j] + xValues[k]));
    
                            if( (tempValue > (interestValue - band)) && (tempValue < (interestValue + band)) )
                            {
                                values.push_back(tempValue);
                            } 
                        }
                    }
                }

		        // average
                if( values.size() > 8 )
                {
                    // insert modified color back into image data
                    float newValue = 0;
                    for(int i = 0; i < values.size(); i++)
                    {
                        newValue += values[i];
                    }

                    newValue /= values.size();

                    insertValue(info->datatype, data, newValue);
                   
                    addedValues++;

                    value = newValue;
                }

		        // check value then add to vertice list
		        if( (value > (interestValue - band)) && (value < (interestValue + band)) )
                {
                	vertices->push_back(osg::Vec3(info->xMultiplier * s, info->yMultiplier * t, info->zMultiplier * (r + info->r_offset)));
			        numPoints++;
                } 
                

                // increment to next value
                data += info->numberOfBytesPerComponent;

                // reset values
                zValues.clear();
                yValues.clear();
                xValues.clear();
                values.clear();
            }
        }
    }

    std::cerr << "Added values via averaging " << addedValues << " total points " << numPoints << std::endl;

    geom->setVertexArray(vertices);
    geode->addDrawable(geom);

    // write out test ply file
    osgDB::writeNodeFile(*geode, "test.obj");

    // remove geode
    geode->unref();	
}

void Volume::createModel(volumeinfo * info)
{
    // look at volume and output vertices based on image values
    //osg::Vec3Array* vertices = new osg::Vec3Array();
    //osg::Vec4Array* colors = new osg::Vec4Array();
    
    ofstream ofs("test.vtk");
    ofs << "# vtk DataFile Version 2.0\n";
    ofs << "Volume example\n";
    ofs << "ASCII\n";
    ofs << "DATASET STRUCTURED_POINTS\n";
    ofs << "DIMENSIONS " << info->x << " " << info->y << " " << info->z << "\n";
    ofs << "ORIGIN 0 0 0\n";
    ofs << "ASPECT_RATIO " << info->xMultiplier << " " << info->yMultiplier << " " << info->zMultiplier << "\n";
    ofs << "POINT_DATA " << (info->x * info->y * info->z) << "\n";
    ofs << "SCALARS volume_scalars float\n";
    ofs << "LOOKUP_TABLE default\n";

    std::cerr << "X Y Z Multipliers: " << info->xMultiplier << " " << info->yMultiplier << " " << info->zMultiplier << std::endl;

    std::vector<int> zValues;
    std::vector<int> yValues;
    std::vector<int> xValues;
    std::vector<float> values;
                
    float band = 0.02;
    float interestValue = 0.3;
    int addedValues = 0;

    for(int r = 0; r < info->z; r++)
    {
        for(int t = 0; t < info->y; t++)
        {
            char* data = (char*) info->image->data(0,t,r + info->r_offset);
            for(int s = 0; s < info->x; s++)
            {
                float value = convertValue(info->datatype, data);

                // check surrounding values then insert
                if(r > 0)
                    zValues.push_back(-(s * t * info->numberOfBytesPerComponent));
                if(t > 0)
                    yValues.push_back(-(s * info->numberOfBytesPerComponent));
                if(s > 0)
                    xValues.push_back(-(info->numberOfBytesPerComponent));

                zValues.push_back(0);
                yValues.push_back(0);
                xValues.push_back(0);
                
                if(r < info->z - 1)
                    zValues.push_back((s * t * info->numberOfBytesPerComponent));
                if(t < info->y - 1)
                    yValues.push_back((s * info->numberOfBytesPerComponent));
                if(s < info->x - 1)
                    xValues.push_back(info->numberOfBytesPerComponent);

                for(int i = 0; i < zValues.size(); i++)
                {
                    for(int j = 0; j < yValues.size(); j++)
                    {
                        for(int k = 0; k < xValues.size(); k++)
                        {
                            float tempValue = convertValue(info->datatype, (char *)(data + zValues[i] + yValues[j] + xValues[k]));
    
                            if( (tempValue > (interestValue - band)) && (tempValue < (interestValue + band)) )
                            {
                                values.push_back(tempValue);
                            } 
                        }
                    }
                }

                if( values.size() > 8 )
                {
                    // insert modified color back into image data
                    float newValue = 0;
                    for(int i = 0; i < values.size(); i++)
                    {
                        newValue += values[i];
                    }

                    newValue /= values.size();

                    insertValue(info->datatype, data, newValue);
                   
                    addedValues++;

                    value = newValue;
                }

                ofs << value << "\n";

                // increment to next value
                data += info->numberOfBytesPerComponent;

                // reset values
                zValues.clear();
                yValues.clear();
                xValues.clear();
                values.clear();
            }
        }
    }

    std::cerr << "Total added values " << addedValues << std::endl;

    ofs.close();
}

bool Volume::loadFile(std::string filename)
{
    int numberBytesPerComponent = 0;
    float xMultiplier, yMultiplier, zMultiplier;
    xMultiplier = yMultiplier = zMultiplier = 1.0;

    // temporary parameters
    std::string name;
    int x, y ,z, sizeS, sizeT, sizeR;

    //std::cerr << "Load called on " << filename << std::endl;

    volumeinfo *info = NULL;

    // find out what type of file
    if( osgDB::getFileExtension(filename) == "xvf" )
    {
        name =  osgDB::getSimpleFileName(filename);
        info = loadXVF(filename, x, y, z, xMultiplier, yMultiplier, zMultiplier, sizeS, sizeT, sizeR, numberBytesPerComponent);
    }

    if( osgDB::getFileExtension(filename) == "vol" )
    {
        info = loadVol(filename, x, y, z, xMultiplier, yMultiplier, zMultiplier, sizeS, sizeT, sizeR, numberBytesPerComponent, name);
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
        //osg::Vec4 minValue, maxValue;
        //osg::computeMinMax(info->image.get(), minValue, maxValue); //TODO
        //osg::modifyImage(info->image.get(),ScaleOperator(1.0f/maxValue.r()));
        //std::cerr << "Min xyz " << minValue.x() << " " << minValue.y() << " " << minValue.z() << std::endl; 
        //std::cerr << "Max xyz " << maxValue.x() << " " << maxValue.y() << " " << maxValue.z() << std::endl; 
    }

    // try smoothing the image data
    //diffuseVolume(info);
    
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
    osg::Vec3Array * vertices = new osg::Vec3Array(16);

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
    (*vertices)[10].set(0.0, 1.0, 0.0);
    (*vertices)[11].set(0.0, 1.0, 1.0);
    (*vertices)[12].set(1.0, 1.0, 0.0);
    (*vertices)[13].set(1.0, 1.0, 1.0);
    (*vertices)[14].set(1.0, 0.0, 0.0);
    (*vertices)[15].set(1.0, 0.0, 1.0);

    boxgeom->setVertexArray(vertices);

    osg::Vec4Array* colors = new osg::Vec4Array(1);
    (*colors)[0].set(1.0, 0.0, 0.0, 1.0);

    boxgeom->setColorArray(colors);
    boxgeom->setColorBinding(osg::Geometry::BIND_OVERALL);

    boxgeom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::LINE_LOOP,0,4));
    boxgeom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::LINE_LOOP,4,4));
    boxgeom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::LINES,8,8));
    boxgeom->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

    geode->addDrawable(boxgeom);
    mattrans->addChild(geode);
//
/*
    // add test normal
    osg::Geode* normGeode = new osg::Geode();
    osg::Geometry* normgeom = new osg::Geometry();
    osg::Vec3Array* normvertices = new osg::Vec3Array(2);
   
    osg::Vec3 point, normal;
    info->clipposUniform->get(point);
    info->clipnormUniform->get(normal);
    (*normvertices)[0].set(point);
    (*normvertices)[1].set(normal + point);

    normgeom->setVertexArray(normvertices);

    osg::Vec4Array* normcolors = new osg::Vec4Array(1);
    (*normcolors)[0].set(0.0, 1.0, 0.0, 1.0);

    normgeom->setColorArray(normcolors);
    normgeom->setColorBinding(osg::Geometry::BIND_OVERALL);

    normgeom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::LINE_STRIP,0,2));
    normgeom->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

    normGeode->addDrawable(normgeom);
    mattrans->addChild(normGeode);
*/
//

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

    //MenuCheckbox* mtfcb = new MenuCheckbox("Default Function", false);
    //mtfcb->setCallback(this);
    //_transferFuncMap[so] = mtfcb;

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
    
    // add isobandvalue to menu
    //so->addMenuItem(mrvib);
    //_isosurfaceBandValueMap[so] = mrvib;

    // add transfer func
    //so->addMenuItem(mtfcb);
    //_transferFuncMap[so] = mtfcb;

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

    // add point cluster controls
    {
        info->seedPoint = new SceneObject("SeedPoint", false, true, false, false, false);
        PluginHelper::registerSceneObject(info->seedPoint,"VolumeSegment");

        // create seed point to move around
        osg::Vec3 seedPoint(info->x * 0.5, info->y * 0.5, info->z * 0.5);
        osg::Vec3 locationPoint(seedPoint.x() * info->xMultiplier, seedPoint.y() * info->yMultiplier, (seedPoint.z() + info->r_offset) * info->zMultiplier );

        osg::Geode* spoint = new osg::Geode();
        osg::ShapeDrawable* shape = new osg::ShapeDrawable(new osg::Sphere(osg::Vec3(0.0, 0.0, 0.0), 4.0f));
        spoint->addDrawable(shape);

	osg::StateSet* stateset = spoint->getOrCreateStateSet();
	Material *material = new Material;
    	material->setDiffuse(Material::FRONT_AND_BACK, Vec4(0.8, 0.0, 0.8, 0.5));
    	material->setAlpha(Material::FRONT_AND_BACK, 0.5f);
    	stateset->setAttributeAndModes(material, StateAttribute::OVERRIDE | StateAttribute::ON);
    	stateset->setMode(GL_BLEND, StateAttribute::OVERRIDE | StateAttribute::ON );
    	stateset->setRenderingHint(StateSet::TRANSPARENT_BIN);

        info->seedPoint->addChild(spoint);
        info->seedPoint->setPosition(locationPoint);
        so->addChild(info->seedPoint);
        //info->seedPoint->getChildNode(0)->setNodeMask(~0);
        //info->seedPoint->getChildNode(0)->setNodeMask(0);

        // initialize surface transform
        info->surface = new osg::MatrixTransform();
        osg::Matrix mat;
        mat.setTrans(osg::Vec3(0.0, 0.0, (info->zMultiplier * info->r_offset)));
        info->surface->setMatrix(mat);
        so->addChild(info->surface);
        
        // create menus for controlling the point
        SubMenu* point = new SubMenu("Point");
        so->addMenuItem(point);
        
        MenuCheckbox* mecb = new MenuCheckbox("Enable Seed", true);
        mecb->setCallback(this);
        point->addItem(mecb);
        _enableSeedMap[so] = mecb;

        MenuRangeValue* mrvlb = new MenuRangeValue("Lower Bound", 0.0, 1.0, 0.3);
        point->addItem(mrvlb);
        _lowerBoundMap[so] = mrvlb;
        
        MenuRangeValue* mrvub = new MenuRangeValue("Upper Bound", 0.0, 1.0, 0.7);
        point->addItem(mrvub);
        _upperBoundMap[so] = mrvub;
        
        MenuRangeValue* mrvsb = new MenuRangeValue("Max Number of steps", 0.0, 500.0, 50.0);
        point->addItem(mrvsb);
        _stepMap[so] = mrvsb;
        
        MenuRangeValue* mrvvb = new MenuRangeValue("Voxel % Surround", 0.0, 1.0, 0.5);
        //point->addItem(mrvvb);
        _voxelSmoothMap[so] = mrvvb;
        
        MenuText* mtv = new MenuText("Seed Value");
        point->addItem(mtv);
        _currentValueMap[so] = mtv;
        
        MenuButton* mcpb = new MenuButton("Compute Points");
        mcpb->setCallback(this);
        point->addItem(mcpb);
        _createPointsMap[so] = mcpb;
        
        MenuButton* mcsb = new MenuButton("Compute Surface");
        mcsb->setCallback(this);
        point->addItem(mcsb);
        _createSurfaceMap[so] = mcsb;
        
        MenuButton* mwsb = new MenuButton("Write Surface");
        mwsb->setCallback(this);
        point->addItem(mwsb);
        _writeSurfaceMap[so] = mwsb;
        
        MenuButton* mrpb = new MenuButton("Remove Points");
        mrpb->setCallback(this);
        point->addItem(mrpb);
        _removePointsMap[so] = mrpb;
        
        MenuButton* mrsb = new MenuButton("Remove Surface");
        mrsb->setCallback(this);
        point->addItem(mrsb);
        _removeSurfaceMap[so] = mrsb;
    }

    // add save position button
    MenuButton* mb = new MenuButton("Save position");
    mb->setCallback(this);
    so->addMenuItem(mb);
    _saveMap[so] = mb;

    // add delete button
    mb = new MenuButton("Delete");
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

    // check if there exits a preset configuration
    bool nav;
    nav = so->getNavigationOn();
    so->setNavigationOn(false);

    if(_locInit.find(name) != _locInit.end())
    {
         so->setTransform(_locInit[name].second);
         so->setScale(so->getScale());
    }
    so->setNavigationOn(nav);

    _volumeMap[so] = info;

    return true;
}

void Volume::menuCallback(MenuItem* menuItem)
{

	//check for main menu selections
    std::map< cvr::MenuItem* , std::string>::iterator it = _menuFileMap.find(menuItem);
    if( it != _menuFileMap.end() )
    {
        loadFile(it->second);
        return;
    }

    // check for remove all button
    if( menuItem == _removeButton )
    {
        removeAll();
		return;
    }

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
/*
    for(std::map<SceneObject*,MenuRangeValue*>::iterator it = _isosurfaceBandValueMap.begin(); it != _isosurfaceBandValueMap.end(); it++)
    {
        if(menuItem == it->second)
        {
            float basevalue =  _isosurfaceValueMap.find(it->first)->second->getValue();
            volumeinfo* info = _volumeMap[it->first];
            osg::StateSet* state = info->tile->getOrCreateStateSet();
            osg::Uniform* uniform = state->getUniform("IsoSurfaceValueMin");
            if( uniform )
            {
                float value = ((basevalue + it->second->getValue()) < 1.0) ? (basevalue + it->second->getValue()) : 1.0f;
                uniform->set(value);
                std::cerr << "Value band set " << value << std::endl;
            }
            return;
        }
    }
*/
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
    
    for(std::map<SceneObject*,MenuButton*>::iterator it = _saveMap.begin(); it != _saveMap.end(); it++)
    {
        if(menuItem == it->second)
        {
            bool nav;
            nav = it->first->getNavigationOn();
            it->first->setNavigationOn(false);

            _locInit[it->first->getName()] = std::pair<float, osg::Matrix>(1.0,it->first->getTransform());

            it->first->setNavigationOn(nav);

            writeConfigFile();
            return;
        }
    }

    // check if seed has been turned on
    for(std::map<SceneObject*,MenuCheckbox*>::iterator it = _enableSeedMap.begin(); it != _enableSeedMap.end(); it++)
    {
         if(menuItem == it->second)
         {
            volumeinfo* info = _volumeMap[it->first];

            if( it->second->getValue() )
            {
                std::cerr << "enable\n";
            	PluginHelper::registerSceneObject(info->seedPoint,"VolumeSegment");
            	//info->seedPoint->attachToScene();
            }
            else
            {
                std::cerr << "disable\n";
                PluginHelper::unregisterSceneObject(info->seedPoint);
            }
            return;
         }
    }

    // check for point placement
    for(std::map<SceneObject*,MenuButton*>::iterator it = _createPointsMap.begin(); it != _createPointsMap.end(); it++)
    {
         if(menuItem == it->second)
         {
            //remove old data
            volumeinfo* info = _volumeMap[it->first];

            // remove old data if it exists
            it->first->removeChild(info->points);

            // construct new data
            info->points = createPointSet(info, info->seedLocation, _lowerBoundMap[it->first]->getValue(), _upperBoundMap[it->first]->getValue(), _stepMap[it->first]->getValue());

            // add new data to scene object
            it->first->addChild(info->points);

            // temporary write a point file
            //createXYZPoints(info, seedPoint, _lowerBoundMap[it->first]->getValue(), _upperBoundMap[it->first]->getValue());
            //createPlyPoints(info, seedPoint, _lowerBoundMap[it->first]->getValue(), _upperBoundMap[it->first]->getValue(), _voxelSmoothMap[it->first]->getValue());
            return;
         }
    }

    // remove points
    for(std::map<SceneObject*,MenuButton*>::iterator it = _removePointsMap.begin(); it != _removePointsMap.end(); it++)
    {
         if(menuItem == it->second)
         {
            //remove old data
            volumeinfo* info = _volumeMap[it->first];
            
            // remove old data if it exists
            it->first->removeChild(info->points);

            return;
         }
    }

    // check for surface creation
    for(std::map<SceneObject*,MenuButton*>::iterator it = _createSurfaceMap.begin(); it != _createSurfaceMap.end(); it++)
    {
         if(menuItem == it->second)
         {
            //remove old data
            volumeinfo* info = _volumeMap[it->first];

            // remove old data if it exists
            while( info->surface->getNumChildren() )
                info->surface->removeChild(info->surface->getChild(0));

	    osg::Geode* surface = createSurface(info, info->seedLocation, _lowerBoundMap[it->first]->getValue(), _upperBoundMap[it->first]->getValue(), _stepMap[it->first]->getValue());
	    if (surface)
            	info->surface->addChild(surface);

            return;
         }
    }

    // write surface model if it exists
    for(std::map<SceneObject*,MenuButton*>::iterator it = _writeSurfaceMap.begin(); it != _writeSurfaceMap.end(); it++)
    {
         if(menuItem == it->second)
         {
            if( cvr::ComController::instance()->isMaster() )
            {
                volumeinfo* info = _volumeMap[it->first];
                
                std::string fileName = osgDB::getNameLessExtension(info->name);
                fileName.append(".obj");
                std::cerr << "Using: " << fileName << std::endl;

                if( info->surface->getNumChildren() )
                {
                    osgDB::writeNodeFile(*(info->surface->getChild(0)), fileName);
                    std::cerr << "Wrote out " << fileName << std::endl;
                }
            }
            return;
         }
    }


    // remove points
    for(std::map<SceneObject*,MenuButton*>::iterator it = _removeSurfaceMap.begin(); it != _removeSurfaceMap.end(); it++)
    {
         if(menuItem == it->second)
         {
            //remove old data
            volumeinfo* info = _volumeMap[it->first];
            
            // remove old data if it exists
            while( info->surface->getNumChildren() )
                info->surface->removeChild(info->surface->getChild(0));

            return;
         }
    }



    for(std::map<SceneObject*,MenuButton*>::iterator it = _deleteMap.begin(); it != _deleteMap.end(); it++)
    {
        if(menuItem == it->second)
        {
            deleteVolume(it->first);
            return;
        }
    }

}

void Volume::deleteVolume(cvr::SceneObject* vol)
{
    if(_standardMap.find(vol) != _standardMap.end())
    {
        delete _standardMap[vol];
        _standardMap.erase(vol);
    }

    if(_isosurfaceValueMap.find(vol) != _isosurfaceValueMap.end())
    {
        delete _isosurfaceValueMap[vol];
        _isosurfaceValueMap.erase(vol);
    }

    if(_transparencyValueMap.find(vol) != _transparencyValueMap.end())
    {
        delete _transparencyValueMap[vol];
        _transparencyValueMap.erase(vol);
    }

    if(_lightMap.find(vol) != _lightMap.end())
    {
        delete _lightMap[vol];
        _lightMap.erase(vol);
    }

    if(_isosurfaceMap.find(vol) != _isosurfaceMap.end())
    {
        delete _isosurfaceMap[vol];
        _isosurfaceMap.erase(vol);
    }

    if(_maxintensityMap.find(vol) != _maxintensityMap.end())
    {
        delete _maxintensityMap[vol];
        _maxintensityMap.erase(vol);
    }

    if(_isosurfaceValueMap.find(vol) != _isosurfaceValueMap.end())
    {
        delete _isosurfaceValueMap[vol];
        _isosurfaceValueMap.erase(vol);
    }

    if(_transparencyValueMap.find(vol) != _transparencyValueMap.end())
    {
        delete _transparencyValueMap[vol];
        _transparencyValueMap.erase(vol);
    }

    if(_transferFuncMap.find(vol) != _transferFuncMap.end())
    {
        delete _transferFuncMap[vol];
        _transferFuncMap.erase(vol);
    }

    if(_transferPositionMap.find(vol) != _transferPositionMap.end())
    {
        delete _transferPositionMap[vol];
        _transferPositionMap.erase(vol);
    }

    if(_transferPositionMap.find(vol) != _transferPositionMap.end())
    {
        delete _transferPositionMap[vol];
        _transferPositionMap.erase(vol);
    }

    if(_transferBaseWidthMap.find(vol) != _transferBaseWidthMap.end())
    {
        delete _transferBaseWidthMap[vol];
        _transferBaseWidthMap.erase(vol);
    }

    if(_transferBrightMap.find(vol) != _transferBrightMap.end())
    {
        delete _transferBrightMap[vol];
        _transferBrightMap.erase(vol);
    }

    if(_transferHueMap.find(vol) != _transferHueMap.end())
    {
        delete _transferHueMap[vol];
        _transferHueMap.erase(vol);
    }

    if(_transferGrayMap.find(vol) != _transferGrayMap.end())
    {
        delete _transferGrayMap[vol];
        _transferGrayMap.erase(vol);
    }
    
    if(_saveMap.find(vol) != _saveMap.end())
    {
        delete _saveMap[vol];
        _saveMap.erase(vol);
    }

    if(_deleteMap.find(vol) != _deleteMap.end())
    {
        delete _deleteMap[vol];
        _deleteMap.erase(vol);
    }

    //TODO need to remove player controls

    if(_volumeMap.find(vol) != _volumeMap.end())
    {
        delete _volumeMap[vol];
        _volumeMap.erase(vol);

        delete vol;
        vol = NULL;
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
            //else
            //{
            //    // make sure there is a default colorMap
            //    if( info->defaultColorMap  )
            //    {
            //        trans1d->assign(*info->defaultColorMap);
            //    }
            //}
        }
    }
    info->tile->setDirty(true);
    info->tile->init();
}

bool Volume::processEvent(InteractionEvent* event)
{
  cerr << "pe" << endl;
  if (event->getEventType() == KEYBOARD_INTER_EVENT)
  {
    KeyboardInteractionEvent *keyEvent = event->asKeyboardEvent();
    int key = keyEvent->getKey();
    if (keyEvent->getInteraction() == KEY_DOWN)
    {
      cerr << "switching eyes" << endl;
      //osgseteyeseparationmultiplier(-1)
    }
  }
  return false;
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
                return true;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      /
            }
        }
    }
    return false;
}
*/

void Volume::preFrame()
{
    // exit preframe if no volumes
    if( ! _volumeMap.size() )
        return;

    // check if point is enabled if so show value in volume
    for(std::map<SceneObject*,MenuCheckbox*>::iterator it = _enableSeedMap.begin(); it != _enableSeedMap.end(); it++)
    {
        volumeinfo* info = _volumeMap[it->first];

        if( it->second->getValue() )
        {
            SceneObject* pointSeed = it->first->getChildObject(0);
            if( pointSeed )
            {
                osg::Vec3 position = pointSeed->getPosition();
                info->seedLocation.set(position.x() / info->xMultiplier, position.y() / info->yMultiplier, (position.z() / info->zMultiplier) - info->r_offset );

                // make sure point is inside volume
                if( (info->seedLocation.x() >= 0) && (info->seedLocation.x() <= info->x) && 
                    (info->seedLocation.y() >= 0) && (info->seedLocation.y() <= info->y) && 
                    (info->seedLocation.z() >= 0) && (info->seedLocation.z() <= info->z) )
                {
                    char* data = (char*) info->image->data(info->seedLocation.x(), info->seedLocation.y(), info->seedLocation.z());
                    std::stringstream ss;
                    ss << "Value: ";
                    ss << convertValue(info->datatype, (char *)data);
                    _currentValueMap[it->first]->setText(ss.str());
                }
            }
        }
    }

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
    std::cerr << "Volume::init()" << endl;
    //osg::setNotifyLevel( osg::INFO );
    
    pgm1 = new osg::Program;
    pgm1->setName( "Sphere" );
    std::string shaderPath = ConfigManager::getEntry("Plugin.Points.ShaderPath");
    pgm1->addShader(osg::Shader::readShaderFile(osg::Shader::VERTEX, osgDB::findDataFile(shaderPath + "/Sphere.vert")));
    pgm1->addShader(osg::Shader::readShaderFile(osg::Shader::FRAGMENT, osgDB::findDataFile(shaderPath + "/Sphere.frag")));
    pgm1->addShader(osg::Shader::readShaderFile(osg::Shader::GEOMETRY, osgDB::findDataFile(shaderPath + "/Sphere.geom")));
    pgm1->setParameter( GL_GEOMETRY_VERTICES_OUT_EXT, 4 );
    pgm1->setParameter( GL_GEOMETRY_INPUT_TYPE_EXT, GL_POINTS );
    pgm1->setParameter( GL_GEOMETRY_OUTPUT_TYPE_EXT, GL_TRIANGLE_STRIP );

	// create default menu
    _volumeMenu = new SubMenu("Volume", "Volume");
    _volumeMenu->setCallback(this);

    _filesMenu = new SubMenu("Files","Files");
    _filesMenu->setCallback(this);
    _volumeMenu->addItem(_filesMenu);
    
    _removeButton = new MenuButton("Remove All");
    _removeButton->setCallback(this);
    _volumeMenu->addItem(_removeButton);

    // read in configurations
    _configPath = ConfigManager::getEntry("Plugin.Volume.ConfigDir");

    ifstream cfile;
    cfile.open((_configPath + "/Init.cfg").c_str(), ios::in);

    if(!cfile.fail())
    {
        string line;
        while(!cfile.eof())
        {
            osg::Matrix m;
            float scale;
            char name[150];
            cfile >> name;
            if(cfile.eof())
            {
                break;
            }
			cfile >> scale;
			for(int i = 0; i < 4; i++)
        	{
        		for(int j = 0; j < 4; j++)
        		{
            		cfile >> m(i, j);
        		}
        	}
            _locInit[string(name)] = pair<float, osg::Matrix>(scale, m);
        }
    }
    cfile.close();

    // read in configuartion files
    vector<string> list;

    string configBase = "Plugin.Volume.Files";

    ConfigManager::getChildren(configBase,list);

    for(int i = 0; i < list.size(); i++)
    {
        MenuButton * button = new MenuButton(list[i]);
        button->setCallback(this);

        // add mapping
        _menuFileMap[button] = ConfigManager::getEntry("value",configBase + "." + list[i],"");

        // add button
        _filesMenu->addItem(button);
    }

    // add menu
    cvr:MenuSystem::instance()->addMenuItem(_volumeMenu);

    return true;
}

void Volume::removeAll()
{
    std::map<cvr::SceneObject*,volumeinfo*>::iterator it;

    while( (it = _volumeMap.begin())  != _volumeMap.end() )
    {
        deleteVolume(it->first);
    }
}

void Volume::writeConfigFile()
{
    // only write on head node
    if(cvr::ComController::instance()->isMaster())
    {

        ofstream cfile;
        cfile.open((_configPath + "/Init.cfg").c_str(), ios::trunc);

        if(!cfile.fail())
        {
    	    for(map<std::string, std::pair<float, osg::Matrix> >::iterator it = _locInit.begin();
        	    it != _locInit.end(); it++)
    	    {
        	    //cerr << "Writing entry for " << it->first << endl;
        	    cfile << it->first << " " << it->second.first << " ";
        	    for(int i = 0; i < 4; i++)
        	    {
        		    for(int j = 0; j < 4; j++)
        		    {
            		    cfile << it->second.second(i, j) << " ";
        		    }
        	    }
        	    cfile << endl;
    	    }
        }
        cfile.close();
    }
}



Volume::~Volume()
{
   printf("Called Volume destructor\n");
}
