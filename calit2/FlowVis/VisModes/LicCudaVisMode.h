#ifndef LIC_CUDA_VIS_MODE_H
#define LIC_CUDA_VIS_MODE_H

#include "VisMode.h"

#include <map>
#include <vector>

#ifndef WIN32
#include <pthread.h>
#else
#include "../pthread_win.h"
#endif

#include <GL/gl.h>

#define LIC_TEXTURE_SIZE 1024

class CudaGLImage;

class LicCudaVisMode : public VisMode
{
    public:
        LicCudaVisMode();
        virtual ~LicCudaVisMode();

        virtual void initContext(int context);
        virtual void uinitContext(int context);
        virtual void frameStart(int context);
        virtual void draw(int context);
        virtual void postFrame();

    protected:
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

        pthread_mutex_t _licLock;
        std::map<int,bool> _licInit;
        std::map<int,GLuint> _licNoiseTex;
        std::map<int,GLuint> _licVelTex;
        std::map<int,GLuint> _licOutputTex;
        std::map<int,GLuint> _licNextOutputTex;
        std::map<int,bool> _licNextDone;
        bool _licStarted;
        std::map<int,bool> _licFinished;
        bool _licOutputValid;
        std::vector<float> _licOutputPoints;
        std::vector<float> _licNextOutputPoints;
        std::map<int,GLuint> _licRenderProgram;
        std::map<int,GLint> _licRenderAlphaUni;
        std::map<int,CudaGLImage*> _licCudaNoiseImage;
        std::map<int,CudaGLImage*> _licCudaVelImage;
        std::map<int,CudaGLImage*> _licCudaOutputImage;
        std::map<int,CudaGLImage*> _licCudaNextOutputImage;
        std::map<int,int> _licContextRenderCount;

        // temp consts for lic kernels
        // isolated so they can be loaded in draw threads
        float ccPlanePoint[3];
        float ccPlaneNormal[3];
        float ccPlaneRight[3];
        float ccPlaneUp[3];
        float ccPlaneRightNorm[3];
        float ccPlaneUpNorm[3];
        float ccPlaneBasisLength;
        float ccTexXMin;
        float ccTexXMax;
        float ccTexYMin;
        float ccTexYMax;
};

#endif
