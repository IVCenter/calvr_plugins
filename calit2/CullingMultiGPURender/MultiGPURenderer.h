/**
 * @file MultiGPURenderer.h
 * Class that handles the parallel draw over 
 *              multiple video cards and combines them into a single image
 *
 * @author Andrew Prudhomme (aprudhomme@ucsd.edu)
 *
 * @date 09/15/2010
 */

#ifndef MULTI_GPU_RENDERER_H
#define MULTI_GPU_RENDERER_H

#include <GL/gl.h>

#include <map>
#include <string>
#include <vector>

#include <osg/Camera>
#include <OpenThreads/Mutex>

#include "CudaHelper.h"

/**
 * how many depth bits to use for the depth texture
 */
enum DepthBits
{
    D16 = 0,        ///< Single 16bit depth texture
    D24,            ///< A 24bit depth texture made of one 16bit and one 8bit texture
    D32             ///< Single 32bit depth texture from the frame buffer
};

/**
 * what method used to copy color and depth textures between graphics cards
 */
enum TextureCopyType
{
    READ_PIX = 0,       ///< use simple gl commands to move textures (i.e. readpixels)
    PBOS,               ///< use pixel buffer objects to move textures
    CUDA_COPY           ///< move textures with cuda dma copy
};

/**
 * non osg color representation
 */
struct ColorVal
{
    float r;        ///< normalized red color value
    float g;        ///< normalized green color value
    float b;        ///< normalized blue color value
};


/**
 * This class provides the interface used when rendering a part.  The class
 * is implemented by the part geometry in the chc algorithm.  
 */
class PartInfo
{
    public:
        PartInfo() {}
        virtual ~PartInfo() {}

        virtual inline bool isOn() { return false; }                    ///< Returns true if the part should be drawn in the Multi-Draw
        virtual inline int getPartNumber() { return 0; }                ///< Returns part number from d3Plot
        virtual inline GLuint getVertBuffer(int context) { return 0; }  ///< Returns vertex buffer id
        virtual inline GLuint getIndBuffer(int context) { return 0; }   ///< Returns indices buffer id
        virtual inline GLuint getNormalBuffer() { return 0; }           ///< Returns normals buffer id (not implemented) 
        virtual inline GLuint getColorBuffer() { return 0; }            ///< Returns color buffer id (not implemented, needed for per vertex color)
        virtual inline int getNumPoints() { return 0; }                 ///< Returns number of indices to push into opengl pipeline
        virtual inline int getVBOOffset(int context) { return 0; }      ///< Returns offset of first point in vertex buffer
};


/**
 * This is the main rendering class.  It initializes and manages the opengl 
 * objects and shaders needed for the parallel draw.
 */
class MultiGPURenderer
{
    public:
        MultiGPURenderer(int width, int height, bool geoShader = false, TextureCopyType copyType = READ_PIX, DepthBits dbits = D24);
        ~MultiGPURenderer();

        /// draw call per gpu, Camera used to collect stats
        void draw(int gpu, osg::Camera * cam = NULL);

        /// returns the number of gpus being used for rendering
        int getNumGPUs();

        /// sets the master map of part objects indexed by its number
        void setPartMap(std::map<int,PartInfo*> * pmap);

        /// sets the list of part numbers rendered by the given gpu
        void setPartList(int gpu, std::vector<int> plist);

        /// sets a list of colors to use for color lookup in the shader
        void setColorMapping(std::vector<struct ColorVal> & colors);
    protected:
        
        /// setup opengl buffers
        bool initBuffers(int gpu);

        /// setup opengl shaders
        bool initShaders(int gpu);

        /// raw draw call for gpu
        void drawGPU(int gpu);

        // draw combined image to screen
        void drawScreen();

        // load default colors to shader
        void loadDefaultColors();

        // load color mapping to shader
        void updateColors();

        /// window width in pixels
        int _width;
        /// window height in pixels
        int _height;

        bool _useGeoShader; ///< use geometry shader for normals
        bool _usePBOs; ///< use pixel buffer objects
        bool _cudaCopy; ///< use cuda texture copy
        DepthBits _depth; ///< how many depth bits to use
        int _numGPUs; ///< number of gpu to use for rendering

        /// flags for draw synchronization
        bool * threadSyncBlock;

        /// base directory for shader files
        std::string _shaderDir;

        /// update color mapping next frame
        bool _updateColors;

        /// color mapping
        std::vector<ColorVal> _colorMap;

        /// signals bad init
        bool _errorState;

        /// lock to serialize buffer initialization
        OpenThreads::Mutex _initMutex;

        /// flag for shader/buffer init done
        std::map<int,bool> _init;

        /// frame buffer objects
        std::map<int,GLuint> _frameBufferMap;

        std::map<int,GLuint> _colorTextureMap;          ///< frame buffer color textures
        std::map<int,GLuint> _depthTextureMap;          ///< frame buffer depth textures
        std::map<int,GLuint> _depth16TextureMap;        ///< frame buffer 16bit depth textures
        std::map<int,GLuint> _depth8TextureMap;         ///< frame buffer 8bit depth textures

        std::map<int,GLuint> _colorBufferMap;           ///< pixel buffers to store color textures in memory
        std::map<int,GLuint> _depthBufferMap;           ///< pixel buffers to store depth textures in memory
        std::map<int,GLuint> _depth8BufferMap;          ///< pixel buffers to store 8bit depth textures in memory

        std::map<int,GLuint> _colorCopyBufferMap;       ///< pixel buffers to copy color textures to primary gpu
        std::map<int,GLuint> _depthCopyBufferMap;       ///< pixel buffers to copy depth textures to primary gpu
        std::map<int,GLuint> _depth8CopyBufferMap;      ///< pixel buffers to copy 8bit depth textures to primary gpu

        /// textures on primary gpu
        std::map<int,GLuint> _colorCopyTextureMap;      ///< color textures copied to primary gpu
        std::map<int,GLuint> _depthCopyTextureMap;      ///< depth textures copied to primary gpu
        std::map<int,GLuint> _depth8CopyTextureMap;     ///< 8bit depth textures copied to primary gpu

        std::map<int,char*> _colorDataMap;          ///< ram buffers for color texture copy
        std::map<int,char*> _depthDataMap;          ///< ram buffers for depth texture copy
        std::map<int,char*> _depth8DataMap;         ///< ram buffers for 8bit depth texture copy

        /// buffer to draw full screen image
        GLuint _screenArray;

        GLint * _redLookupUni;      ///< address list of color lookup values in shader
        GLint * _greenLookupUni;    ///< address list of color lookup values in shader
        GLint * _blueLookupUni;     ///< address list of color lookup values in shader

        GLint * _colorsUni;     ///< address list of color texture index in shader
        GLint * _depthUni;      ///< address list of depth texture index in shader
        GLint * _depth8Uni;     ///< address list of 8bit depth texture index in shader

        /// address of number of color textures in shader
        GLint _texturesUni;

        /// address of number of colors in color list in shader
        GLint _numColorsUni;

        GLuint _comVert;    ///< recombination vertex shader
        GLuint _comFrag;    ///< recombination fragment shader
        GLuint _comProg;    ///< recombination shader program

        /// draw shaders
        std::map<int,GLuint> _drawVert;     ///< draw vertex shaders
        std::map<int,GLuint> _drawGeo;      ///< draw geometry shaders
        std::map<int,GLuint> _drawFrag;     ///< draw fragment shaders
        std::map<int,GLuint> _drawProg;     ///< draw shader programs

        /// map of all parts
        std::map<int, PartInfo*> * _partMap;

        /// map of parts list per gpu
        std::map<int, std::vector<int> > _partList;

        std::map<int,CudaGLImage *> _cudaColorImage;        ///< used for cuda copy of color texture to ram
        std::map<int,CudaGLImage *> _cudaDepth16Image;      ///< used for cuda copy of 16bit depth texture to ram
        std::map<int,CudaGLImage *> _cudaDepth8Image;       ///< used for cuda copy of 8bit depth texture to ram

        std::map<int,CudaGLImage *> _cudaCBColorImage;      ///< used for cuda copy of color texture from ram to gpu 0
        std::map<int,CudaGLImage *> _cudaCBDepth16Image;    ///< used for cuda copy of 16bit depth texture from ram to gpu 0
        std::map<int,CudaGLImage *> _cudaCBDepth8Image;     ///< used for cuda copy of 8bit depth texture from ram to gpu 0
};

#endif
