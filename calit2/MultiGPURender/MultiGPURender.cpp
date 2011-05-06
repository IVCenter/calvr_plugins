#include "MultiGPURender.h"
#include "Timing.h"

#include <kernel/SceneManager.h>
#include <config/ConfigManager.h>

#include <osg/Material>

#include <iostream>
#include <sstream>
#include <cstdio>
#include <algorithm>

CVRPLUGIN(MultiGPURender)

using namespace cvr;
using namespace osg;

MultiGPURender::MultiGPURender()
{
    _animation = NULL;
}

MultiGPURender::~MultiGPURender()
{
    if(_animation)
    {
	//delete _animation;
	_animation->done();
    }
}

bool MultiGPURender::init()
{
    std::cerr << "MultiGPURender init()" << std::endl;

    //_basePath = ConfigManager::getEntry("Plugin.MultiGPURender.BasePath");
    _basePath = ConfigManager::getEntry("basePath","Plugin.MultiGPURender.Animation","");
    _baseName = ConfigManager::getEntry("baseName","Plugin.MultiGPURender.Animation","");
    _frames = ConfigManager::getInt("frames","Plugin.MultiGPURender.Animation",1);
    _colors = ConfigManager::getInt("colors","Plugin.MultiGPURender.Animation",1);
    _numGPUs = ConfigManager::getInt("Plugin.MultiGPURender.NumberOfGPUs");

    int depth = ConfigManager::getInt("Plugin.MultiGPURender.DepthBuffer",24);

    std::stringstream file;
    if(_numGPUs == 2)
    {
	file << "combine2";
    }
    else if(_numGPUs == 3)
    {
	file << "combine3";
    }
    else if(_numGPUs == 4)
    {
	file << "combine4";
    }
    else
    {
	file << "combine";
    }

    if(depth == 32)
    {
	file << "-32";
    }
    else if(depth == 24)
    {
	file << "-24";
    }
    else if(depth == 16)
    {
	file << "-16";
    }

    file << ".frag";

    _geode = new Geode();
    _drawable = new MultiGPUDrawable("combine.vert",file.str());
    
    _geode->addDrawable(_drawable);

    osg::StateSet * stateset = _geode->getOrCreateStateSet();
    osg::Material * mat = new osg::Material();
    mat->setColorMode(osg::Material::DIFFUSE);

    stateset->setAttributeAndModes(mat, StateAttribute::ON);
    //_geode->setCullingActive(false);

    _geode->setDataVariance(osg::Object::STATIC);
    SceneManager::instance()->getObjectsRoot()->addChild(_geode);

    _animation = new AnimationManager(_basePath,_baseName,_frames,_colors,LOAD_ALL,_drawable);
    _animation->start();

    //loadColorSplitData(_basePath, 0);

    return true;
}

void MultiGPURender::preFrame()
{
#ifdef PRINT_TIMING
    static struct timeval lastframe;
    static struct timeval thisframe;

    getTime(thisframe);
    if(_animation->isCacheDone())
    {
	printDiff("FrameTime: ",lastframe,thisframe);
    }
    lastframe = thisframe;
#endif
    _animation->update();
    //std::cerr << "Frame." << std::endl;
}

void MultiGPURender::loadColorSplitData(std::string basePath, int frame)
{
    std::cerr << "Load started." << std::endl;
    for(int i = 0; i < 29; i++)
    {
	//std::cerr << "Loading color " << i << std::endl;
	char num[10];
	sprintf(num,"%.4d",frame);
	std::stringstream ss;
	ss << basePath << "/" << _baseName << num << "Color" << i << ".bvbo";

	FILE * file; 
	file = fopen(ss.str().c_str(),"rb");
	if(!file)
	{
	    std::cerr << "Unable to open file " << ss.str() << std::endl;
	    return;
	}
	int numPrim;
	fread(&numPrim,1,sizeof(int),file);

	//std::cerr << "numPrim " << numPrim << std::endl;

	if(!numPrim)
	{
	    std::cerr << "No data in file " << ss.str() << std::endl;
	    _lineData.push_back(NULL);
	    _lineSize.push_back(0);
	    _quadData.push_back(std::pair<float*,float*>(NULL,NULL));
	    _quadSize.push_back(0);
	    _colorList.push_back(osg::Vec4());
	    fclose(file);
	    continue;
	}

	float r,g,b;
	fread(&r,1,sizeof(float),file);
	fread(&g,1,sizeof(float),file);
	fread(&b,1,sizeof(float),file);
	_colorList.push_back(osg::Vec4(r,g,b,1.0));

	GLenum type;
	fread(&type,1,sizeof(GLenum),file);
	if(numPrim < 2)
	{
	    if(type == GL_LINES)
	    {
		_quadData.push_back(std::pair<float*,float*>(NULL,NULL));
		_quadSize.push_back(0);
	    }
	    else
	    {
		_lineData.push_back(NULL);
		_lineData.push_back(0);
	    }
	}
	unsigned int totalNumPoints = 0;
	for(int j = 0; j < numPrim; j++)
	{
	    unsigned int size;
	    fread(&size,1,sizeof(unsigned int),file);
	    std::cerr << "size: " << size << std::endl;
	    if(type == GL_LINES)
	    {
		//std::cerr << "type is LINE" << std::endl;
		float * data = new float[size * 3 * 2];
		fread(data,1,sizeof(float)*3*2*size,file);
		_lineData.push_back(data);
		_lineSize.push_back(size);
		totalNumPoints += size * 2;
	    }
	    else if(type == GL_QUADS)
	    {
		//std::cerr << "type is quad" << std::endl;
		float * data = new float[size * 3 * 4];
		fread(data,1,sizeof(float)*3*4*size,file);
		float * ndata = new float[size*3 * 4];
		fread(ndata,1,sizeof(float)*3*4*size,file);
		_quadData.push_back(std::pair<float*,float*>(data,ndata));
		_quadSize.push_back(size);
		totalNumPoints += size * 8;
		int numTri;
		fread(&numTri,1,sizeof(int),file);
		//std::cerr << "Found " << numTri << " triangles." << std::endl;
		if(numTri)
		{
		    std::cerr << "Triangles " << numTri << std::endl;
		    _triSize.push_back(numTri);
		    data = new float[numTri * 3 * 3];
		    fread(data,1,sizeof(float)*3*3*numTri,file);
		    ndata = new float[numTri * 3 * 3];
		    fread(ndata,1,sizeof(float)*3*3*numTri,file);
		    _triData.push_back(std::pair<float*,float*>(data,ndata));
		    totalNumPoints += numTri * 6;
		}
		else
		{
		    _triSize.push_back(0);
		    _triData.push_back(std::pair<float*,float*>(NULL,NULL));

		}
	    }
	    if(j+1 < numPrim)
	    {
		fread(&type,1,sizeof(GLenum),file);
	    }
	}

	_partSizes.push_back(std::pair<unsigned int,int>(totalNumPoints,i));
	fclose(file);
    }
    std::cerr << "Load finished." << std::endl;

    std::sort(_partSizes.begin(),_partSizes.end(),SizeSort());
    std::vector<unsigned int> currentSize;
    for(int i = 0; i < _numGPUs; i++)
    {
	currentSize.push_back(0);
	_gpuPartsMap[i] = std::vector<int>();
    }

    for(int i = 0; i < _partSizes.size(); i++)
    {
	int gpu = 0;
	unsigned int size = currentSize[0];
	for(int j = 1; j < _numGPUs; j++)
	{
	    if(currentSize[j] < size)
	    {
		gpu = j;
		size = currentSize[j];
	    }
	}
	_gpuPartsMap[gpu].push_back(_partSizes[i].second);
	currentSize[gpu] += _partSizes[i].first;
    }

    for(int i = 0; i < _numGPUs; i++)
    {
	std::cerr << "GPU " << i << std::endl;
	for(int j = 0; j < _gpuPartsMap[i].size(); j++)
	{
	    std::cerr << _gpuPartsMap[i][j] << std::endl;
	}
	std::cerr << "currentSize " << currentSize[i] << std::endl;
    }

    for(int i = 0; i < _numGPUs; i++)
    {
	for(int j = 0; j < _gpuPartsMap[i].size(); j++)
	{
	    /*if(_gpuPartsMap[i][j] != 20)
	    {
		continue;
	    }*/
	    if(_lineData[_gpuPartsMap[i][j]])
	    {
		_drawable->addArray(i,GL_LINES,_lineData[_gpuPartsMap[i][j]],NULL,_lineSize[_gpuPartsMap[i][j]], _colorList[_gpuPartsMap[i][j]]);
	    }

	    if(_quadData[_gpuPartsMap[i][j]].first && _quadData[_gpuPartsMap[i][j]].second)
	    {
		_drawable->addArray(i,GL_QUADS,_quadData[_gpuPartsMap[i][j]].first,_quadData[_gpuPartsMap[i][j]].second,_quadSize[_gpuPartsMap[i][j]], _colorList[_gpuPartsMap[i][j]]);
	    }

	    if(_triData[_gpuPartsMap[i][j]].first && _triData[_gpuPartsMap[i][j]].second)
	    {
		_drawable->addArray(i,GL_TRIANGLES,_triData[_gpuPartsMap[i][j]].first,_triData[_gpuPartsMap[i][j]].second,_triSize[_gpuPartsMap[i][j]], _colorList[_gpuPartsMap[i][j]]);
	    }
	}
    }
}
