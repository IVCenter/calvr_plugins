/**
 * @file BufferManager.h
 * Contains management class for large ram buffers and vbos to hold vertex data 
 *
 * @author Andrew Prudhomme (aprudhomme@ucsd.edu)
 *
 * @date 09/15/2010
 */

#ifndef BUFFER_MANAGER_H
#define BUFFER_MANAGER_H

#include <map>
#include <vector>

#include <GL/gl.h>

class Geometry;

/**
 * Class that manages ram buffers and vbo copies for vertex data.
 *
 * This is used to combine many small copies to the gpu with a few large ones by distributing space in large buffers to geometry as needed. 
 */
class BufferManager
{
    public:
        BufferManager(std::map<int,Geometry *> & partMap, std::vector<int> & partList, int context, bool cudaCopy = false);
        ~BufferManager();

        char * requestBuffer(Geometry * geo);
        char * requestNextBuffer(Geometry * geo);

        char * getBuffer0Pointer(Geometry * geo);

        void loadFrameData(bool predraw = true);

        void setFrame(int frame);
        void setNextFrame(int frame);

    protected:
        void resetFrame();
        void resetNextFrame();

        void swapFrame();

        void addMemoryBuffer();

        bool _cudaCopy;                             ///< should copies to the gpu be done with cuda functions
        int _context;                               ///< context(gpu) id for this object

        int _frame;                                 ///< current frame number
        int _nextFrame;                             ///< next frame number

        std::map<int,Geometry*> _partMap;           ///< map of parts indexed by part number
        std::vector<int> _partList;                 ///< list of parts assigned to this gpu

        std::vector<char *> _frameMemList;          ///< list of pointers to large ram buffers for current frame
        std::vector<char *> _nextFrameMemList;      ///< list of pointers to large ram buffers for next frame

        std::vector<int> _frameRemList;             ///< list of space remaining in buffers for current frame
        std::vector<int> _nextFrameRemList;         ///< list of space remaining in buffers for next frame

        std::vector<int> _frameCopiedList;          ///< list of amount of buffer already copied to the gpu for this frame
        std::vector<int> _nextFrameCopiedList;      ///< list of amount of buffer already copied to the gpu for next frame

        std::vector<int> _frameOffsetList;          ///< list of offset of next availble space in buffers for current frame
        std::vector<int> _nextFrameOffsetList;      ///< list of offset of next availble space in buffers for next frame

        std::vector<GLuint> _vboList;               ///< list of opengl buffer ids for large VBOs
};

#endif
