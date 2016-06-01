#ifndef PANO_DRAWABLE_LOD_H
#define PANO_DRAWABLE_LOD_H

#include <GL/glew.h>

#ifndef WIN32
#include <unistd.h>
#endif

// hack for compile on dan's mac
#ifndef GL_DOUBLE_VEC2
#undef GL_ARB_gpu_shader_fp64
#endif

#ifndef GL_DOUBLE_MAT3x2
#undef GL_ARB_gpu_shader_fp64
#endif

#include <osg/Drawable>

#include "sph-cache.hpp"
#include "sph-model.hpp"

#include <osg/Version>
#include <OpenThreads/Mutex>

#include <string>
#include <vector>
#include <map>

enum PanTransitionType
{
    NORMAL,
    ZOOM,
    MORPH
};

struct PanoDrawableInfo
{
    std::vector<std::string> leftEyeFiles;
    std::map<int,std::vector<int> > leftFileIDs;
    std::vector<std::string> rightEyeFiles;
    std::map<int,std::vector<int> > rightFileIDs;
    std::map<int,bool> updateDoneMap;

    std::map<int,int> initMap;
    OpenThreads::Mutex initLock;

    std::map<int,OpenThreads::Mutex*> updateLock;

    static std::map<int,sph_cache*> cacheMap;
    static std::map<int,std::vector<std::pair<bool,sph_model*> > > currentModels;
    static OpenThreads::Mutex staticLock;
    std::map<int,sph_model*> modelMap;
    std::map<int,sph_model*> transitionModelMap;

    osg::Matrix fromTransitionTransform;
    osg::Matrix toTransitionTransform;
    float transitionFade;
};

class PanoDrawableLOD : public osg::Drawable
{
    public:
        PanoDrawableLOD(PanoDrawableInfo * pdi, float radius, int mesh, int depth, int size, std::string vertFile = "sph-zoomer.vert", std::string fragFile = "sph-render.frag");
        PanoDrawableLOD(const PanoDrawableLOD&,const osg::CopyOp& copyop=osg::CopyOp::SHALLOW_COPY);
        virtual ~PanoDrawableLOD();
 
        void cleanup();

        void next();
        void previous();

        void transitionDone();

        void setZoom(osg::Vec3 dir, float k);
        void setRadius(float radius) { _radius = radius; }
       
        void setAlpha(float alpha) { _alpha = alpha; }
        float getAlpha() { return _alpha; }

        void setDebug(bool d);
        
        float getCurrentFadeTime() { return _currentFadeTime; }

        virtual Object* cloneType() const { return NULL; }
        virtual Object* clone(const osg::CopyOp& copyop) const { return new PanoDrawableLOD(*this,copyop); }
        virtual bool isSameKindAs(const Object* obj) const { return dynamic_cast<const PanoDrawableLOD*>(obj)!=NULL; }
        virtual const char* libraryName() const { return "PanoView"; }
        virtual const char* className() const { return "PanoDrawableLOD"; }
#if ( OSG_VERSION_LESS_THAN(3, 4, 0) )
        virtual osg::BoundingBox computeBound() const;
#else
		virtual osg::BoundingBox computeBoundingBox() const;
#endif
        virtual void updateBoundingBox();

        virtual void drawImplementation(osg::RenderInfo&) const;

        void setTransitionType(PanTransitionType transitionType)
        {
            _transitionType = transitionType;
        }

        int getSetSize();
        int getCurrentIndex()
        {
            return _currentIndex;
        }
        int getLastIndex()
        {
            return _lastIndex;
        }
        int getNextIndex()
        {
            return _nextIndex;
        }

    protected:
        struct PanoUpdate : public osg::Drawable::UpdateCallback
        {
            virtual void update(osg::NodeVisitor *, osg::Drawable *);
        };

        enum DrawEye
        {
            DRAW_LEFT = 1,
            DRAW_RIGHT = 2
        };

        PanoDrawableInfo * _pdi;

        float _radius;
        float _alpha;
        int _mesh;
        int _depth;
        int _size;
        char * _vertData;
        char * _fragData;

        int _currentIndex;
        int _lastIndex;
        int _nextIndex;
        float _totalFadeTime;
        float _currentFadeTime;
        mutable bool _badInit;

        mutable bool _transitionActive;

        mutable osg::BoundingBox _boundingBox;

        PanTransitionType _transitionType;
};

#endif
