#include "AnimationManager.h"
#include "Timing.h"

#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/ComController.h>
#include <sstream>
#include <algorithm>
#include <cstdio>

#include <sys/syscall.h>
#include <sys/stat.h>

using namespace cvr;
using namespace std;

AnimationManager::AnimationManager(std::string basepath, std::string basename,int frames,int colors, LoadType lt, MultiGPUDrawable * drawable)
{
    _basePath = basepath;
    _baseName = basename;
    _frames = frames;
    _colors = colors;
    _lt = lt;
    _drawable = drawable;
    _cacheDone = false;

    _currentFrame = -1;
    _loadRatio = ConfigManager::getFloat("loadRatio","Plugin.CrashDemo.Animation",1.0);

    _numGPUs = ConfigManager::getInt("Plugin.CrashDemo.NumberOfGPUs");
    _timingStack = new CircularStack(10);

    loadOrCreateInfoFile();
}

AnimationManager::~AnimationManager()
{
}

bool AnimationManager::isCacheDone()
{
    return _cacheDone;
}

void AnimationManager::update()
{
    static bool frameLoading = false;
    /*if(!_cacheDone)
    {
	return;
    }*/

    if(_currentFrame < 0)
    {
	bool cDone;
        if(cvr::ComController::instance()->isMaster())
        {
          cDone = _cacheDone;
          int numSlaves = cvr::ComController::instance()->getNumSlaves();
          bool sDone[numSlaves];
          cvr::ComController::instance()->readSlaves(sDone,sizeof(bool));
          for(int i = 0; i < numSlaves; i++)
          {
            cDone = cDone && sDone[i];
          }
          cvr::ComController::instance()->sendSlaves(&cDone,sizeof(bool));
        }
        else
        {
          cDone = _cacheDone;
          cvr::ComController::instance()->sendMaster(&cDone,sizeof(bool));
          cvr::ComController::instance()->readMaster(&cDone,sizeof(bool));
        }

	if(!cDone)
        {
          //std::cerr << "Waiting for load to finish." << std::endl;
          return;
        }

        cvr::ComController::instance()->sync();
	_drawable->setFrame(_frameMap[0]);
	_currentFrame = 0;
	return;
    }

    struct timeval now;
    getTime(now);
    _drawable->setPreFrameTime(now);

    //return;
    if(!frameLoading)
    {
	_currentFrame = (_currentFrame + 1) % _frames;
	_drawable->setNextFrame(_frameMap[_currentFrame], (unsigned int)(_loadRatio * _maxBytes) + 1);
	frameLoading = true;
	return;
    }

    if(_drawable->nextFrameLoadDone())
    {
	//std::cerr << "Load Finished." << std::endl;
	_drawable->swapFrames();
	_currentFrame = (_currentFrame + 1) % _frames;
	_drawable->setNextFrame(_frameMap[_currentFrame], (unsigned int)(_loadRatio * _maxBytes) + 1);
    }
    else
    {
	//std::cerr << "Waiting for GPU load finish." << std::endl;
    }

}

void AnimationManager::run()
{
    if(_lt == LOAD_ALL)
    {
	for(int i = 0; i < _frames; i++)
	{
	    loadColorSplitData(i);
	}

	/*std::map<int,int> lineMap;
	std::map<int,int> quadMap;
	std::map<int,int> triMap;
	for(map<int,AFrame*>::iterator it = _frameMap.begin(); it != _frameMap.end(); it++)
	{
	    for(int i = 0; i < _colors; i++)
	    {
		if(it->second->lineSize[i] > lineMap[i])
		{
		    lineMap[i] = it->second->lineSize[i];
		}
		if(it->second->quadSize[i] > quadMap[i])
		{
		    quadMap[i] = it->second->quadSize[i];
		}
		if(it->second->triSize[i] > triMap[i])
		{
		    triMap[i] = it->second->triSize[i];
		}
	    }
	}
	for(map<int,AFrame*>::iterator it = _frameMap.begin(); it != _frameMap.end(); it++)
	{
	    it->second->maxLineSize = lineMap;
	    it->second->maxQuadSize = quadMap;
	    it->second->maxTriSize = triMap;
	}*/
	
	_cacheDone = true;

	std::cerr << "All frames loaded." << std::endl;

	struct timespec stime;
	stime.tv_sec = 0;
	stime.tv_nsec = 1000000;
	while(1)
	{
	    nanosleep(&stime,NULL);
	}
    }
}

void AnimationManager::loadColorSplitData(int frame)
{
    int dataLoaded = 0;
    struct timeval start, end;
    getTime(start);

    std::cerr << "Load started." << std::endl;
    struct AFrame * frameStruct = new AFrame;
    if(_frameMap[frame])
    {
	//delete old frame prehapse
    }
    _frameMap[frame] = frameStruct;
    frameStruct->frameNum = frame;
    frameStruct->maxLineSize = _maxLineSize;
    frameStruct->maxQuadSize = _maxQuadSize;
    frameStruct->maxTriSize = _maxTriSize;
    for(int i = 0; i < _colors; i++)
    {
	//std::cerr << "Loading color " << i << std::endl;
	char num[10];
	sprintf(num,"%.4d",frame);
	std::stringstream ss;
	ss << _basePath << "/" << _baseName << num << "Color" << i << ".bvbo";

	FILE * file; 
	file = fopen(ss.str().c_str(),"rb");
	if(!file)
	{
	    std::cerr << "Unable to open file " << ss.str() << std::endl;
	    return;
	}
	int numPrim;
	fread(&numPrim,1,sizeof(int),file);
	dataLoaded += sizeof(int);

	//std::cerr << "numPrim " << numPrim << std::endl;

	if(!numPrim)
	{
	    std::cerr << "No data in file " << ss.str() << std::endl;
	    frameStruct->lineData.push_back(NULL);
	    frameStruct->lineSize.push_back(0);
	    frameStruct->quadData.push_back(std::pair<float*,float*>(NULL,NULL));
	    frameStruct->quadSize.push_back(0);
	    frameStruct->triData.push_back(std::pair<float*,float*>(NULL,NULL));
	    frameStruct->triSize.push_back(0);
	    frameStruct->colorList.push_back(osg::Vec4());
	    fclose(file);
	    continue;
	}

	float r,g,b;
	fread(&r,1,sizeof(float),file);
	fread(&g,1,sizeof(float),file);
	fread(&b,1,sizeof(float),file);
	frameStruct->colorList.push_back(osg::Vec4(r,g,b,1.0));
	dataLoaded += 3 * sizeof(float);


	GLenum type;
	fread(&type,1,sizeof(GLenum),file);
	dataLoaded += sizeof(GLenum);

	if(numPrim < 2)
	{
	    if(type == GL_LINES)
	    {
		frameStruct->quadData.push_back(std::pair<float*,float*>(NULL,NULL));
		frameStruct->quadSize.push_back(0);
	    }
	    else
	    {
		frameStruct->lineData.push_back(NULL);
		frameStruct->lineData.push_back(0);
	    }
	}
	unsigned int totalNumPoints = 0;
	for(int j = 0; j < numPrim; j++)
	{
	    unsigned int size;
	    fread(&size,1,sizeof(unsigned int),file);
	    dataLoaded += sizeof(unsigned int);
	    std::cerr << "size: " << size << std::endl;
	    if(type == GL_LINES)
	    {
		//std::cerr << "type is LINE" << std::endl;
		float * data = new float[size * 3 * 2];
			
		fread(data,1,sizeof(float)*3*2*size,file);
		// PHILIP simple hack to not drawlines
		data[0] = 0.0;

		dataLoaded += sizeof(float)*3*2*size;
		frameStruct->lineData.push_back(data);
		frameStruct->lineSize.push_back(size);
		totalNumPoints += size * 2;
	    }
	    else if(type == GL_QUADS)
	    {
		//std::cerr << "type is quad" << std::endl;
		float * data = new float[size * 3 * 4];
		fread(data,1,sizeof(float)*3*4*size,file);
		dataLoaded += sizeof(float)*3*4*size;
		float * ndata = new float[size*3 * 4];
		fread(ndata,1,sizeof(float)*3*4*size,file);
		dataLoaded += sizeof(float)*3*4*size;
		frameStruct->quadData.push_back(std::pair<float*,float*>(data,ndata));
		frameStruct->quadSize.push_back(size);
		totalNumPoints += size * 8;
		int numTri;
		fread(&numTri,1,sizeof(int),file);
		dataLoaded += sizeof(int);
		//std::cerr << "Found " << numTri << " triangles." << std::endl;
		if(numTri)
		{
		    std::cerr << "Triangles " << numTri << std::endl;
		    frameStruct->triSize.push_back(numTri);
		    data = new float[numTri * 3 * 3];
		    fread(data,1,sizeof(float)*3*3*numTri,file);
		    dataLoaded += sizeof(float)*3*3*numTri;
		    ndata = new float[numTri * 3 * 3];
		    fread(ndata,1,sizeof(float)*3*3*numTri,file);
		    dataLoaded += sizeof(float)*3*3*numTri;
		    frameStruct->triData.push_back(std::pair<float*,float*>(data,ndata));
		    totalNumPoints += numTri * 6;
		}
		else
		{
		    frameStruct->triSize.push_back(0);
		    frameStruct->triData.push_back(std::pair<float*,float*>(NULL,NULL));

		}
	    }
	    if(j+1 < numPrim)
	    {
		fread(&type,1,sizeof(GLenum),file);
		dataLoaded += sizeof(GLenum);
	    }
	}

	frameStruct->partSizes.push_back(std::pair<unsigned int,int>(totalNumPoints,i));
	fclose(file);
    }
    std::cerr << "Load finished." << std::endl;

    getTime(end);
    _timingStack->push(dataLoaded,getDiff(start,end));

    std::cerr << "Time per byte: " << _timingStack->getTimePerByte() << std::endl;

    if(_partsMap.size())
    {
	frameStruct->gpuPartsMap = _partsMap;
	return;
    }

    std::sort(frameStruct->partSizes.begin(),frameStruct->partSizes.end(),SizeSortA());
    std::vector<unsigned int> currentSize;
    for(int i = 0; i < _numGPUs; i++)
    {
	currentSize.push_back(0);
	frameStruct->gpuPartsMap[i] = std::vector<int>();
    }

    for(int i = 0; i < frameStruct->partSizes.size(); i++)
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
	frameStruct->gpuPartsMap[gpu].push_back(frameStruct->partSizes[i].second);
	currentSize[gpu] += frameStruct->partSizes[i].first;
    }

    for(int i = 0; i < _numGPUs; i++)
    {
	std::cerr << "GPU " << i << std::endl;
	for(int j = 0; j < frameStruct->gpuPartsMap[i].size(); j++)
	{
	    std::cerr << frameStruct->gpuPartsMap[i][j] << std::endl;
	}
	std::cerr << "currentSize " << currentSize[i] << std::endl;
    }

    _partsMap = frameStruct->gpuPartsMap;

    /*for(int i = 0; i < _numGPUs; i++)
    {
	for(int j = 0; j < _gpuPartsMap[i].size(); j++)
	{
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
    }*/
}

void AnimationManager::loadOrCreateInfoFile()
{
    std::string fileName = ConfigManager::getEntry("infoFile","Plugin.CrashDemo.Animation","default.ani");

    std::stringstream ss;
    ss << _basePath << "/" << fileName;

    FILE * file; 
    file = fopen(ss.str().c_str(),"rb");
    if(!file)
    {
	std::cerr << "Unable to open file " << ss.str() << " : attempting to create." << std::endl;
	file = fopen(ss.str().c_str(),"wb");
	if(!file)
	{
	    std::cerr << "Unable to open file for writing." << std::endl;
	    return;
	}

	FILE * frameFile;
	int frameCount = 0;
	std::stringstream ssFile;
	char num[10];
	sprintf(num,"%.4d",frameCount);
	ssFile << _basePath << "/" << _baseName << num << "Color" << 0 << ".bvbo";
	frameFile = fopen(ssFile.str().c_str(),"rb");
	while(frameFile)
	{
	    unsigned int frameSize = 0;
	    _frameLineSizeMap[frameCount] = std::vector<unsigned int>();
	    _frameQuadSizeMap[frameCount] = std::vector<unsigned int>();
	    _frameTriSizeMap[frameCount] = std::vector<unsigned int>();
	    for(int i = 0; i < _colors; i++)
	    {
		struct stat st;
		if(stat(ssFile.str().c_str(),&st) != 0)
		{
		    std::cerr << "Error stating file: " << ssFile.str() << std::endl;
		}
		else
		{
		    frameSize += st.st_size;
		}

		int numPrim;
		fread(&numPrim,1,sizeof(int),frameFile);

		if(!numPrim)
		{
		    _frameLineSizeMap[frameCount].push_back(0);
		    _frameQuadSizeMap[frameCount].push_back(0);
		    _frameTriSizeMap[frameCount].push_back(0);
		}
		else
		{
		    fseek(frameFile,3*sizeof(float),SEEK_CUR);
		    GLenum type;
		    fread(&type,1,sizeof(GLenum),frameFile);
		    if(numPrim < 2)
		    {
			if(type == GL_LINES)
			{
			    _frameQuadSizeMap[frameCount].push_back(0);
			    _frameTriSizeMap[frameCount].push_back(0);
			}
			else
			{
			    _frameLineSizeMap[frameCount].push_back(0);
			}
		    }

		    for(int j = 0; j < numPrim; j++)
		    {
			unsigned int size;
			fread(&size,1,sizeof(unsigned int),frameFile);
			if(type == GL_LINES)
			{
			    fseek(frameFile,sizeof(float)*3*2*size,SEEK_CUR);
			    _frameLineSizeMap[frameCount].push_back(size);
			}
			else if(type == GL_QUADS)
			{
			    fseek(frameFile,sizeof(float)*3*4*size,SEEK_CUR);
			    fseek(frameFile,sizeof(float)*3*4*size,SEEK_CUR);
			    _frameQuadSizeMap[frameCount].push_back(size);
			    int numTri;
			    fread(&numTri,1,sizeof(int),frameFile);
			    if(numTri)
			    {
				_frameTriSizeMap[frameCount].push_back(numTri);
				fseek(frameFile,sizeof(float)*3*3*numTri,SEEK_CUR);
				fseek(frameFile,sizeof(float)*3*3*numTri,SEEK_CUR);
			    }
			    else
			    {
				_frameTriSizeMap[frameCount].push_back(0);

			    }
			}
			if(j+1 < numPrim)
			{
			    fread(&type,1,sizeof(GLenum),frameFile);
			}
		    }
		}

		fclose(frameFile);
		if(i+1 < _colors)
		{
		    std::stringstream tss;
		    sprintf(num,"%.4d",frameCount);
		    tss << _basePath << "/" << _baseName << num << "Color" << i+1 << ".bvbo";
		    frameFile = fopen(tss.str().c_str(),"rb");
		}
	    }

	    _frameSizeMap[frameCount] = frameSize;

	    frameCount++;
	    std::stringstream tss;
	    sprintf(num,"%.4d",frameCount);
	    tss << _basePath << "/" << _baseName << num << "Color" << 0 << ".bvbo";
	    frameFile = fopen(tss.str().c_str(),"rb");
	}

	_fullFrameCount = frameCount;
	_fullColorCount = _colors;
	fwrite(&frameCount,1,sizeof(int),file);
	std::cerr << "Frame Count: " << frameCount << std::endl;

	std::cerr << "Colors : " << _colors << std::endl;
	fwrite(&_colors,1,sizeof(int),file);

	for(int i = 0; i < frameCount; i++)
	{
	    fwrite(&_frameSizeMap[i],1,sizeof(unsigned int),file);
	    std::cerr << "Frame Size: " << _frameSizeMap[i] << std::endl;
	    for(int j = 0; j < _frameLineSizeMap[i].size(); j++)
	    {
		fwrite(&_frameLineSizeMap[i][j],1,sizeof(unsigned int),file);
		fwrite(&_frameQuadSizeMap[i][j],1,sizeof(unsigned int),file);
		fwrite(&_frameTriSizeMap[i][j],1,sizeof(unsigned int),file);
		std::cerr << "Line: " << _frameLineSizeMap[i][j] << " Quad: " << _frameQuadSizeMap[i][j] << " Tri: " << _frameTriSizeMap[i][j] << std::endl;
	    }
	}

	fclose(file);
	std::cerr << "Info File Write Finished." << std::endl;
    }
    else
    {
	fread(&_fullFrameCount,1,sizeof(int),file);
	fread(&_fullColorCount,1,sizeof(int),file);
	std::cerr << "Full Frame Count: " << _fullFrameCount << " colors: " << _fullColorCount << std::endl;
	for(int i = 0; i < _fullFrameCount; i++)
	{
	    _frameLineSizeMap[i] = std::vector<unsigned int>();
	    _frameQuadSizeMap[i] = std::vector<unsigned int>();
	    _frameTriSizeMap[i] = std::vector<unsigned int>();
	    fread(&_frameSizeMap[i],1,sizeof(unsigned int),file);
	    std::cerr << "Frame Size: " << _frameSizeMap[i] << std::endl;
	    for(int j = 0; j < _fullColorCount; j++)
	    {
		unsigned int tempint;
		fread(&tempint,1,sizeof(unsigned int),file);
		_frameLineSizeMap[i].push_back(tempint);
		fread(&tempint,1,sizeof(unsigned int),file);
		_frameQuadSizeMap[i].push_back(tempint);
		fread(&tempint,1,sizeof(unsigned int),file);
		_frameTriSizeMap[i].push_back(tempint);
		std::cerr << "Line: " << _frameLineSizeMap[i][j] << " Quad: " << _frameQuadSizeMap[i][j] << " Tri: " << _frameTriSizeMap[i][j] << std::endl;
	    }
	}
    }

    std::vector<std::pair<unsigned int,int> > partSizes;

    std::map<int,unsigned int> colorMaxMap;

    for(int i = 0; i < _colors; i++)
    {
	unsigned int maxsize = 0;
	unsigned int sizeBytes = 0;
	for(int j = 0; j < _frames; j++)
	{
	    unsigned int thisSize = 0;
	    thisSize += 2 * _frameLineSizeMap[j][i];
	    thisSize += 8 * _frameQuadSizeMap[j][i];
	    thisSize += 6 * _frameTriSizeMap[j][i];
	    if(thisSize > maxsize)
	    {
		maxsize = thisSize;
	    }

	    unsigned int thisSizeBytes = 0;
	    thisSizeBytes += 2 * 3 * _frameLineSizeMap[j][i] * sizeof(float);
	    thisSizeBytes += 8 * 3 *_frameQuadSizeMap[j][i] * sizeof(float);
	    thisSizeBytes += 6 * 3 *_frameTriSizeMap[j][i] * sizeof(float);
	    if(thisSizeBytes > sizeBytes)
	    {
		sizeBytes = thisSizeBytes;
	    }
	}
	partSizes.push_back(std::pair<unsigned int,int>(maxsize,i));
	colorMaxMap[i] = sizeBytes;
    }

    std::sort(partSizes.begin(),partSizes.end(),SizeSortA());
    std::vector<unsigned int> currentSize;
    for(int i = 0; i < _numGPUs; i++)
    {
	currentSize.push_back(0);
	_partsMap[i] = std::vector<int>();
    }

    for(int i = 0; i < partSizes.size(); i++)
    {
	/*int gpu = 0;
	unsigned int size = currentSize[0];
	for(int j = 1; j < _numGPUs; j++)
	{
	    if(currentSize[j] < size)
	    {
		gpu = j;
		size = currentSize[j];
	    }
	}
	_partsMap[gpu].push_back(partSizes[i].second);
	currentSize[gpu] += partSizes[i].first;
	_maxBytesPerGPU[gpu] += colorMaxMap[partSizes[i].second];*/
        for(int j = 0; j < _numGPUs; j++)
        {
            _partsMap[j].push_back(partSizes[i].second);
            currentSize[j] += partSizes[i].first;
            _maxBytesPerGPU[j] += colorMaxMap[partSizes[i].second];
        }
    }

    std::cerr << "From File" << std::endl;

    for(int i = 0; i < _numGPUs; i++)
    {
	std::cerr << "GPU " << i << std::endl;
	for(int j = 0; j < _partsMap[i].size(); j++)
	{
	    std::cerr << _partsMap[i][j] << std::endl;
	}
	std::cerr << "currentSize " << currentSize[i] << std::endl;
	std::cerr << "MaxSizeBytes: " << _maxBytesPerGPU[i] << std::endl;
    }

    _maxBytes = 0;
    for(int i = 0; i < _numGPUs; i++)
    {
	if(_maxBytesPerGPU[i] > _maxBytes)
	{
	    _maxBytes = _maxBytesPerGPU[i];
	}
    }

    for(int i = 0; i < _frames; i++)
    {
	for(int j = 0; j < _colors; j++)
	{
	    if(_frameLineSizeMap[i][j] > _maxLineSize[j])
	    {
		_maxLineSize[j] = _frameLineSizeMap[i][j];
	    }
	    if(_frameQuadSizeMap[i][j] > _maxQuadSize[j])
	    {
		_maxQuadSize[j] = _frameQuadSizeMap[i][j];
	    }
	    if(_frameTriSizeMap[i][j] > _maxTriSize[j])
	    {
		_maxTriSize[j] = _frameTriSizeMap[i][j];
	    }
	}
    }
}
