#ifndef MULTI_GPU_RENDER
#define MULTI_GPU_RENDER

#include <kernel/CVRPlugin.h>

#include <osg/Geode>
#include <osg/Vec4>

#include <string>
#include <vector>
#include <map>

#include "MultiGPUDrawable.h"
#include "AnimationManager.h"

struct SizeSort
{
    bool operator() (std::pair<unsigned int, int> const& first, std::pair<unsigned int, int> const& second)
    {
        return first.first > second.first;
    }
};

class MultiGPURender : public cvr::CVRPlugin
{
    public:
        MultiGPURender();
        virtual ~MultiGPURender();

        bool init();
        void preFrame();

    protected:
        void loadColorSplitData(std::string basePath, int frame);
        std::vector<std::pair<unsigned int,int> > _partSizes;
        std::vector<float *> _lineData;
        std::vector<unsigned int> _lineSize;
        std::vector<std::pair<float*,float*> > _quadData;
        std::vector<unsigned int> _quadSize;
        std::vector<std::pair<float*,float*> > _triData;
        std::vector<unsigned int> _triSize;
        
        std::vector<osg::Vec4> _colorList;

        std::map<int,std::vector<int> > _gpuPartsMap;

        std::string _basePath;
        std::string _baseName;
        int _frames;
        int _colors;

        osg::Geode * _geode;
        MultiGPUDrawable * _drawable;

        AnimationManager * _animation;

        int _numGPUs;
};

#endif
