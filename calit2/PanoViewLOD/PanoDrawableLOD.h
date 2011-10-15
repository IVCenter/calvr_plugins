#ifndef PANO_DRAWABLE_LOD_H
#define PANO_DRAWABLE_LOD_H

#include "sph-cache.hpp"
#include "sph-model.hpp"

#include <osg/Drawable>
#include <OpenThreads/Mutex>

#include <string>
#include <vector>
#include <map>

class PanoDrawableLOD : public osg::Drawable
{
    public:
        PanoDrawableLOD(std::string leftEyeFile, std::string rightEyeFile, float radius, int mesh, int depth, int size, std::string vertFile = "sph-zoomer.vert", std::string fragFile = "sph-render.frag");
        PanoDrawableLOD(const PanoDrawableLOD&,const osg::CopyOp& copyop=osg::CopyOp::SHALLOW_COPY);
        virtual ~PanoDrawableLOD();

        virtual Object* cloneType() const { return NULL; }
        virtual Object* clone(const osg::CopyOp& copyop) const { return new PanoDrawableLOD(*this,copyop); }
        virtual bool isSameKindAs(const Object* obj) const { return dynamic_cast<const PanoDrawableLOD*>(obj)!=NULL; }
        virtual const char* libraryName() const { return "PanoView"; }
        virtual const char* className() const { return "PanoDrawableLOD"; }
        virtual osg::BoundingBox computeBound() const;
        virtual void updateBoundingBox();

        virtual void drawImplementation(osg::RenderInfo&) const;

    protected:
        struct PanoUpdate : public osg::Drawable::UpdateCallback
        {
            virtual void update(osg::NodeVisitor *, osg::Drawable *);
        };

        std::vector<std::string> _leftEyeFiles;
        mutable std::vector<int> _leftFileIDs;
        std::vector<std::string> _rightEyeFiles;
        mutable std::vector<int> _rightFileIDs;

        mutable std::map<int,bool> _updateDoneMap;

        float _radius;
        int _mesh;
        int _depth;
        int _size;
        char * _vertData;
        char * _fragData;

        int _currentIndex;
        bool _badInit;

        mutable OpenThreads::Mutex _initLock;

        mutable std::map<int,sph_cache*> _cacheMap;
        mutable std::map<int,sph_model*> _modelMap;

        mutable osg::BoundingBox _boundingBox;
};

#endif
