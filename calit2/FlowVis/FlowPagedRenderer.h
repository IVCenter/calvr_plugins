#ifndef FLOW_PAGED_RENDERER_H
#define FLOW_PAGED_RENDERER_H

#include "FlowObject.h"
#include "FlowVis.h"
#include "VBOCache.h"
#include "VisModes/VisMode.h"

#include <string>
#include <map>
#include <queue>

#ifndef WIN32
#include <pthread.h>
#endif

#include <GL/gl.h>

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

class FlowPagedRenderer
{
    public: 
        FlowPagedRenderer(PagedDataSet * set, int frame, FlowVisType type, std::string attribute, int cacheSize);
        ~FlowPagedRenderer();

        void frameStart(int context);
        void preFrame();
        void preDraw(int context);
        void draw(int context);
        void postFrame();

        void setType(FlowVisType type, std::string attribute);
        FlowVisType getType();

        void setNextFrame(int frame);
        bool advance();
        bool canAdvance();

        void setNextFrameReady(int context, bool ready);

        void setUniData(std::string key, struct UniData & data);
        bool getUniData(std::string key, struct UniData & data);

        std::map<std::string,struct UniData> & getUniDataMap()
        {
            return _uniDataMap;
        }

        GLuint getColorTableID(int context);

        void freeResources(int context);
        bool freeDone();

        static void setCudaInitInfo(std::map<int,std::pair<int,int> > & initInfo);
        static void setContextRenderCount(std::map<int,int> & contextRenderCountMap);
        static std::map<int,int> & getContextRenderCountMap()
        {
            return _contextRenderCountMap;
        }

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

        void drawElements(GLenum mode, GLsizei count, GLenum type, GLuint indVBO, GLuint vertsVBO, std::vector<float> & color, std::vector<AttribBinding> & attribBinding, GLuint program, std::vector<TextureBinding> & textureBinding, std::vector<UniformBinding> & uniBinding);

        void drawArrays(GLenum mode, GLint first, GLsizei count, GLuint vertsVBO, std::vector<float> & color, std::vector<AttribBinding> & attribBinding, GLuint program, std::vector<TextureBinding> & textureBinding, std::vector<UniformBinding> & uniBinding);

        VBOCache * getCache()
        {
            return _cache;
        }

        PagedDataSet * getSet()
        {
            return _set;
        }

        int getCurrentFrame()
        {
            return _currentFrame;
        }

        int getNextFrame()
        {
            return _nextFrame;
        }

        std::string & getAttribute()
        {
            return _attribute;
        }

    protected:
        void initUniData();

        void checkGlewInit(int context);
        void checkCudaInit(int context);
        void checkColorTableInit(int context);

        void deleteUniData(UniData & data);

        void loadUniform(UniformBinding & uni);

        struct operation
        {
            enum opType
            {
                INIT_OP=0,
                UINIT_OP
            };
            opType op;
            VisMode * visMode;
            int context;
            void runOp();
        };

        std::map<int,std::queue<operation*> > _opQueue;

        PagedDataSet * _set;
        int _currentFrame, _nextFrame;
        std::map<int,bool> _nextFrameReady;
        pthread_mutex_t _frameReadyLock;
        FlowVisType _type;
        std::string _attribute;
        std::map<std::string,struct UniData> _uniDataMap;
        std::map<FlowVisType,VisMode*> _visModeMap;
        
        VBOCache * _cache;

        static std::map<int,bool> _glewInitMap;
        static pthread_mutex_t _glewInitLock;

        static std::map<int,bool> _cudaInitMap;
        static pthread_mutex_t _cudaInitLock;
        static std::map<int,std::pair<int,int> > _cudaInitInfo;
        static std::map<int,int> _contextRenderCountMap;

        static std::map<int,GLuint> _colorTableMap;
        static pthread_mutex_t _colorTableInitLock;
};

#endif
