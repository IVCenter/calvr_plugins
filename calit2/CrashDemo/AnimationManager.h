#ifndef _ANIMATION_MANAGER_MULT
#define _ANIMATION_MANAGER_MULT

#include <iostream>
#include <string>
#include <map>
#include "MultiGPUDrawable.h"
#include "CircularStack.h"

#include <OpenThreads/Thread>

enum LoadType
{
    LOAD_ALL=0,
    DYNAMIC
};

struct SizeSortA
{
    bool operator() (std::pair<unsigned int, int> const& first, std::pair<unsigned int, int> const& second)
    {
        return first.first > second.first;
    }
};

struct AFrame
{
    int frameNum;
    std::vector<std::pair<unsigned int,int> > partSizes;
    std::vector<float *> lineData;
    std::vector<unsigned int> lineSize;
    std::vector<std::pair<float*,float*> > quadData;
    std::vector<unsigned int> quadSize;
    std::vector<std::pair<float*,float*> > triData;
    std::vector<unsigned int> triSize;

    std::vector<osg::Vec4> colorList;

    std::map<int,std::vector<int> > gpuPartsMap;
    std::map<int,int> maxLineSize;
    std::map<int,int> maxQuadSize;
    std::map<int,int> maxTriSize;
};

class AnimationManager : public OpenThreads::Thread
{
    public:
        AnimationManager(std::string basepath, std::string basename,int frames,int colors, LoadType lt, MultiGPUDrawable * drawable);
        ~AnimationManager();

        void update();

        bool isCacheDone();

        virtual void run();

    protected:
        void loadColorSplitData(int frame);
        void loadOrCreateInfoFile();

        std::string _baseName;
        std::string _basePath;
        int _colors;
        int _frames;
        LoadType _lt;
        MultiGPUDrawable * _drawable;
        bool _cacheDone;

        int _numGPUs;

        std::map<int,AFrame*> _frameMap;
        int _currentFrame;

        std::map<int,int> _maxLineSize;
        std::map<int,int> _maxQuadSize;
        std::map<int,int> _maxTriSize;

        int _fullFrameCount;
        int _fullColorCount;
        std::map<int,unsigned int> _frameSizeMap;
        std::map<int,std::vector<unsigned int> > _frameLineSizeMap;
        std::map<int,std::vector<unsigned int> > _frameQuadSizeMap;
        std::map<int,std::vector<unsigned int> > _frameTriSizeMap;

        std::map<int,unsigned int> _maxBytesPerGPU;
        unsigned int _maxBytes;
        float _loadRatio;

        std::map<int,std::vector<int> > _partsMap;
        CircularStack * _timingStack;
};

#endif
