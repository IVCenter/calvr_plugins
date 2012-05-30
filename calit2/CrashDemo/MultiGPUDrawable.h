#ifndef MULTI_GPU_DRAWABLE_H
#define MULTI_GPU_DRAWABLE_H

#include <osg/Drawable>
#include <osg/Vec4>
#include <OpenThreads/Mutex>

#include <map>
#include <string>

#include <sys/time.h>

struct AFrame;

enum FillStage
{
    LINE,
    QUAD_VERT,
    QUAD_NORM,
    TRI_VERT,
    TRI_NORM
};

class MultiGPUDrawable : public osg::Drawable
{
    public:
        MultiGPUDrawable(std::string vertFile, std::string fragFile);
        MultiGPUDrawable(const MultiGPUDrawable&,const osg::CopyOp& copyop=osg::CopyOp::SHALLOW_COPY);

        virtual void drawImplementation(osg::RenderInfo& ri) const;

        virtual Object* cloneType() const { return new MultiGPUDrawable("",""); }
        virtual Object* clone(const osg::CopyOp& copyop) const { return new MultiGPUDrawable(*this,copyop); }
        virtual osg::BoundingBox computeBound() const;
        virtual void updateBoundingBox();

        void addArray(int gpu, GLenum type, float * data, float * normals, unsigned int size, osg::Vec4 color);
        void setFrame(AFrame * frame);
        void setNextFrame(AFrame * frame, unsigned int bytes = 0);
        bool nextFrameLoadDone();
        void swapFrames();

        float getAvgLoadTime();
        float getDrawTime();

        void setPreFrameTime(struct timeval & time);
    protected:
        virtual ~MultiGPUDrawable();
        void initShaders(int context) const;
        void loadGeometry(int context) const;
        void initBuffers(int context) const;
        void drawContext(int context) const;
        void drawScreen() const;

        void initVBOs(int context) const;
        void loadVBOs(int context) const;
        void loadNextVBOs(int context) const;
        void drawVBOs(int context) const;
        void drawFrameVBOs(int context) const;

        bool _useDrawShader;

        mutable std::map<int,osg::Drawable*> _geometryMap;
        mutable std::map<int,bool> _initMap;
        mutable std::map<int,GLuint> _frameBufferMap;
        mutable std::map<int,GLuint> _colorTextureMap;
        mutable std::map<int,GLuint> _depthTextureMap;
        mutable std::map<int,GLuint> _depthRTextureMap;
        mutable std::map<int,GLuint> _depthR8TextureMap;
        mutable std::map<int,GLuint> _colorBufferMap;
        mutable std::map<int,GLuint> _depthBufferMap;
        mutable std::map<int,GLint *> _colorDataMap;
        mutable std::map<int,GLint *> _depthDataMap;
        mutable std::map<int,unsigned char *> _depthR8DataMap;

        bool * threadSyncBlock;
        GLuint * colorCopyBuffers;
        GLuint * depthCopyBuffers;
        GLuint * colorTextures;
        GLuint * depthTextures;
        GLuint * depthRTextures;
        GLuint * depthR8Textures;

        std::string _shaderDir;
        int _width, _height;
        int _gpus;

        mutable OpenThreads::Mutex _mutex;
        mutable OpenThreads::Mutex _vboMutex;
        std::string _vertFile;
        std::string _fragFile;
        std::string _drawVertFile;
        std::string _drawFragFile;
        mutable GLuint _vertShader;
        mutable GLuint _fragShader;
        mutable GLuint _shaderProgram;
        mutable GLuint _screenArray;

        mutable GLint * colorsUni;
        mutable GLint * depthUni;
        mutable GLint * depthR8Uni;
        mutable GLint texturesUni;

        mutable std::map<int,GLint *> redLookupUni;
        mutable std::map<int,GLint *> greenLookupUni;
        mutable std::map<int,GLint *> blueLookupUni;

        mutable std::map<int,GLuint> _drawVertShaderMap;
        mutable std::map<int,GLuint> _drawFragShaderMap;
        mutable std::map<int,GLuint> _drawShaderProgMap;

        mutable std::map<int,bool> _makeVBOs;
        mutable std::map<int,bool> _madeVBOs;
        mutable std::map<int,bool> _loadVBOs;

        mutable std::map<int,std::vector<GLuint> > _lineVBOs;
        mutable std::map<int,std::vector<float *> > _lineData;
        mutable std::map<int,std::vector<unsigned int> > _lineSize;
        mutable std::map<int,std::vector<std::pair<std::vector<GLuint>,std::vector<GLuint> > > > _quadVBOs;
        mutable std::map<int,std::vector<std::pair<float*,float*> > > _quadData;
        mutable std::map<int,std::vector<std::vector<unsigned int> > > _quadSize;
        mutable std::map<int,std::vector<std::pair<float*,float*> > > _triData;
        mutable std::map<int,std::vector<unsigned int> > _triSize;
        mutable std::map<int,std::vector<std::pair<GLuint,GLuint> > > _triVBOs;

        mutable std::map<int,std::vector<GLuint> > _lineNextVBOs;
        mutable std::map<int,std::vector<std::pair<std::vector<GLuint>,std::vector<GLuint> > > > _quadNextVBOs;
        mutable std::map<int,std::vector<std::pair<GLuint,GLuint> > > _triNextVBOs;
        mutable std::map<int,std::vector<unsigned int> > _lineNextSize;
        mutable std::map<int,std::vector<std::vector<unsigned int> > > _quadNextSize;
        mutable std::map<int,std::vector<unsigned int> > _triNextSize;

        mutable unsigned int _nextFrameBytes;
        mutable std::map<int,bool> _nextFrameLoadDone;
        mutable std::map<int,bool> _loadNextFrameSize;

        mutable AFrame * _currentFrame;
        mutable AFrame * _nextFrame;

        mutable std::map<int,std::vector<osg::Vec4> > _colorMap;

        mutable std::map<int,bool> _getTimings;
        mutable std::map<int,float> _dataLoadTime;
        mutable float _drawTime;

        mutable std::map<int,int> _prgParts;
        mutable std::map<int,FillStage> _prgStage;
        mutable std::map<int,int> _prgQuadNum;
        mutable std::map<int,int> _prgOffset;

        mutable struct timeval _preFrameTime;
};

#endif
