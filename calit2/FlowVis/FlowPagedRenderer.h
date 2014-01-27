#ifndef FLOW_PAGED_RENDERER_H
#define FLOW_PAGED_RENDERER_H

#include "FlowObject.h"
#include "FlowVis.h"
#include "VBOCache.h"

#include <string>
#include <map>
#include <pthread.h>

#include <GL/gl.h>

#define LIC_TEXTURE_SIZE 1024

enum UniType
{
    UNI_FLOAT,
    UNI_FLOAT3,
    UNI_MAT3,
    UNI_INT,
    UNI_UINT
};

struct UniData
{
    UniType type;
    void * data;
};

class CudaGLImage;

class FlowPagedRenderer
{
    public: 
        FlowPagedRenderer(PagedDataSet * set, int frame, FlowVisType type, std::string attribute);
        ~FlowPagedRenderer();

        void frameStart(int context);
        void preFrame();
        void preDraw(int context);
        void draw(int context);
        void postFrame();

        void setType(FlowVisType type, std::string attribute);
        FlowVisType getType();
        std::string getAttribute();

        void setNextFrame(int frame);
        bool advance();

        void setUniData(std::string key, struct UniData & data);
        bool getUniData(std::string key, struct UniData & data);

        void freeResources(int context);
        bool freeDone();

        static void setCudaInitInfo(std::map<int,std::pair<int,int> > & initInfo);

    protected:
        void initUniData();

        void checkGlewInit(int context);
        void checkCudaInit(int context);
        void checkShaderInit(int context);
        void checkColorTableInit(int context);
        void checkLICInit(int context);

        void deleteUniData(UniData & data);

        struct AttribBinding
        {
            int index;
            int size;
            GLenum type;
            GLuint buffer;
        };

        struct TextureBinding
        {
            GLuint id;
            int unit;
            GLenum type;
        };

        struct UniformBinding
        {
            GLint location;
            UniType type;
            void * data;
        };

        void loadUniform(UniformBinding & uni);

        void drawElements(GLenum mode, GLsizei count, GLenum type, GLuint indVBO, GLuint vertsVBO, std::vector<float> & color, std::vector<AttribBinding> & attribBinding, GLuint program, std::vector<TextureBinding> & textureBinding, std::vector<UniformBinding> & uniBinding);

        void drawArrays(GLenum mode, GLint first, GLsizei count, GLuint vertsVBO, std::vector<float> & color, std::vector<AttribBinding> & attribBinding, GLuint program, std::vector<TextureBinding> & textureBinding, std::vector<UniformBinding> & uniBinding);

        PagedDataSet * _set;
        int _currentFrame, _nextFrame;
        std::map<int,bool> _nextFrameReady;
        pthread_mutex_t _frameReadyLock;
        FlowVisType _type;
        std::string _attribute;
        std::map<std::string,struct UniData> _uniDataMap;
        
        VBOCache * _cache;

        static std::map<int,bool> _glewInitMap;
        static pthread_mutex_t _glewInitLock;

        static std::map<int,bool> _cudaInitMap;
        static pthread_mutex_t _cudaInitLock;
        static std::map<int,std::pair<int,int> > _cudaInitInfo;

        std::map<int,bool> _shaderInitMap;
        pthread_mutex_t _shaderInitLock;

        std::map<int,GLuint> _normalProgram;

        std::map<int,GLuint> _normalFloatProgram;
        std::map<int,GLint> _normalFloatMinUni;
        std::map<int,GLint> _normalFloatMaxUni;

        std::map<int,GLuint> _normalIntProgram;
        std::map<int,GLint> _normalIntMinUni;
        std::map<int,GLint> _normalIntMaxUni;

        std::map<int,GLuint> _normalVecProgram;
        std::map<int,GLint> _normalVecMinUni;
        std::map<int,GLint> _normalVecMaxUni;

        std::map<int,GLuint> _isoProgram;
        std::map<int,GLint> _isoMaxUni;

        std::map<int,GLuint> _isoVecProgram;
        std::map<int,GLint> _isoVecMaxUni;

        std::map<int,GLuint> _planeFloatProgram;
        std::map<int,GLint> _planeFloatMinUni;
        std::map<int,GLint> _planeFloatMaxUni;
        std::map<int,GLint> _planeFloatPointUni;
        std::map<int,GLint> _planeFloatNormalUni;
        std::map<int,GLint> _planeFloatAlphaUni;

        std::map<int,GLuint> _planeVecProgram;
        std::map<int,GLint> _planeVecMinUni;
        std::map<int,GLint> _planeVecMaxUni;
        std::map<int,GLint> _planeVecPointUni;
        std::map<int,GLint> _planeVecNormalUni;
        std::map<int,GLint> _planeVecAlphaUni;

        std::map<int,GLuint> _vecPlaneProgram;
        std::map<int,GLint> _vecPlaneMinUni;
        std::map<int,GLint> _vecPlaneMaxUni;
        std::map<int,GLint> _vecPlanePointUni;
        std::map<int,GLint> _vecPlaneNormalUni;
        std::map<int,GLint> _vecPlaneUpUni;
        std::map<int,GLint> _vecPlaneRightUni;
        std::map<int,GLint> _vecPlaneBasisInvUni;

        std::map<int,GLuint> _vortexAlphaProgram;
        std::map<int,GLint> _vortexAlphaMinUni;
        std::map<int,GLint> _vortexAlphaMaxUni;

        pthread_mutex_t _licLock;
        std::map<int,bool> _licInit;
        std::map<int,GLuint> _licNoiseTex;
        std::map<int,GLuint> _licVelTex;
        std::map<int,GLuint> _licOutputTex;
        std::map<int,GLuint> _licNextOutputTex;
        std::map<int,bool> _licNextDone;
        bool _licStarted;
        std::vector<float> _licOutputPoints;
        std::vector<float> _licOutputTexCoords;
        std::map<int,GLuint> _licRenderProgram;
        std::map<int,CudaGLImage*> _licCudaNoiseImage;
        std::map<int,CudaGLImage*> _licCudaVelImage;
        std::map<int,CudaGLImage*> _licCudaOutputImage;
        std::map<int,CudaGLImage*> _licCudaNextOutputImage;
        pthread_mutex_t _licCudaLock;

        static std::map<int,GLuint> _colorTableMap;
        static pthread_mutex_t _colorTableInitLock;
};

#endif
