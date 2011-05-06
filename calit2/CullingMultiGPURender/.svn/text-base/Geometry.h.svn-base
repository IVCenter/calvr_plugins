#ifndef GEOMETRY_H
#define GEOMETRY_H

#include "MultiGPURenderer.h"

extern "C" {
    #include "MathStuff.h"
}

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <OpenThreads/Mutex>

#include "CudaHelper.h"
#include "Timing.h"

#include "BufferManager.h"

enum BufferLoadStatus
{
    NOT_LOADING = 0,
    LOADING,
    LOADED
};

/** 
	Represents a drawable geometry. It can draw simple objects (teapot,
	torus, ...). It also contains the object transformation and 
	the material of the objects. An AAB is generated as bounding volume of the object.
*/
class Geometry : public PartInfo 
{
public:
	Geometry(int partNum, std::string& filename, bool cudaCopy);
	Geometry(int partNum, std::string& filename, int numIndex, unsigned int * indices, bool cudaCopy);

        ~Geometry();

        void InitalizeBuffers(int context);

	//! renders this geometry
	void Render();
	
	// --- material settings
	void SetDiffuseColor(float diffuseR,   float diffuseG,  float diffuseB);

	//! returns boudning box of this geometry
	const AABox& GetBoundingVolume();
	//const osg::BoundingBox& GetBoundingVolume();
		
	//! set frame when geometry was last rendered. important for rendering each geometry only once.
	void SetLastRendered(int lastRendered);
	//! returns frame when geometry was last rendered
	int GetLastRendered();
	void SetShared(bool);
	bool IsShared();
	void SetFrame(int context, bool predraw = false); // will copy data from local memory to video card
        void SetFrame(int context, cudaStream_t & stream, bool predraw = false);
	void SetBuffer(int, char * buffer = NULL, bool updateStatus = true);
        void SetVisible(bool visible);
	//void SetNumber(int num) { number = num; };
	int GetNumber() { return number; };
        void SetFrameToCopy(int);
        void SetVertices(int num, int offset);
        int getLastRendered() { return mLastRendered; };	
        // partsinfo 
        bool isOn() { return isVisible; }
        int getPartNumber() { return GetNumber(); }
        void setDrawn(bool b) { _drawn = b; };
        bool isDrawn() { return _drawn; };
        float getCopyTime() { return copyTime; };
        float getLoadTime() { return loadTime; };
        float getDataSize() { return dataSize; };
        GLuint getVertBuffer(int context) 
        {
            if(context)
            {
                return vboId[0]; 
            }
            else
            {
                return vbo0Id[0];
            }
        }
        void setVertBuffer(int context, GLuint buffer);
        GLuint getIndBuffer(int context) 
        {
            if(context)
            {
                return vboId[1]; 
            }
            else
            {
                return vbo0Id[1];
            }
        }
        int getVBOOffset(int context) 
        {
            if(context)
            {
                return _vboOffset;
            }
            return _vbo0Offset; 
        }
        void setVBOOffset(int context, int offset) 
        {
            if(context)
            {
                _vboOffset = offset;
            }
            else
            {
                _vbo0Offset = offset;
            }
        }
        GLuint getNormalBuffer() { return 0; }
        GLuint getColorBuffer() { return 0; }
        int getNumPoints() { return numIndices; }
        struct ColorVal makeColor(float);

        void startNextFetch();

        void setBufferManager0(BufferManager * bufferManager);
        void setBufferManager(BufferManager * bufferManager);

        void setFrameNumbers(int context, int thisFrame, int nextFrame);

        void processPostDrawn(bool paused = false);

        int getNumVerts() { return numVertexs; }
        void setNextVertBuffer(GLuint buffer) { _nextVertVBO = buffer; }
        void setNextVBOOffset(int os) { _nextVBOOffset = os; }

        // stats
        static void ResetStats();
	static int GetTotalLoadedBytes();
	static int GetTotalPreFetched();
	static int GetTotalLoadedOnDemand();
	static int GetTotalNumGeometry();
	static double GetTotalCopyTime();

protected:

        Geometry();

        void setFrameNumber(int context, int fnum);
        void setNextFrameNumber(int fnum);

        void swapFrames(int context);

	// material
	float mAmbientColor[3];
	float mDiffuseColor[3];
	float mSpecularColor[3];
	//! the bounding box of the geometry
	AABox mBoundingBox;
	//osg::BoundingBox mBoundingBox;
	
	bool shared;
	
        bool _postDrawn;

	//! last rendered frame
	int mLastRendered;

	// static members
	//static bool sIsInitialised;

	// geode contains the primitives that make up a part
	GLuint vboId[2];
        GLuint vbo0Id[2];
        GLuint _nextVertVBO;
        int _nextVBOOffset;
        BufferLoadStatus _frameStatus;
        BufferLoadStatus _nextFrameStatus;

	float* vertices;
        unsigned int* indices;
        int offset;
        int numVertexs;
        int numIndices;
	int number; // identifier in array

	// for reading in data from file
	int _frame; // current frame data
        int _nextFrame;
	int buffernum; // what frame has been prefetched from disk into memory
        int newframe; // the frame that should be copied to video card memory from main memory
	std::string basename;
	OpenThreads::Mutex _mutex;
        OpenThreads::Mutex _statusMutex;
        ColorVal color;

        // is visible flag
        bool isVisible;
        bool drawVisible;
        bool _drawn;

        float copyTime;
        float loadTime;
        float dataSize;

        // check if buffers have been initalized
        bool initalizedBuffers;
        bool initalized0Buffers;

	static int totalNumberOfBytes;
	static int totalPreFetch;
	static int totalLoadOnDemand;
	static int totalNumGeometry;
	static double totalCopyTime;

        bool _cudaCopy;

        BufferManager * _bufferMan0;
        BufferManager * _bufferMan;

        int _vboOffset;
        int _vbo0Offset;
};


#endif // GEOMETRY_H
