#ifndef VBO_CACHE_H
#define VBO_CACHE_H

#include "DataLoader.h"

#include <string>
#include <queue>
#include <vector>
#include <list>
#include <map>
#include <pthread.h>

struct BufferJob
{
    std::string fileName;
    int file;
    int offset;
    void * mappedPtr;
    void * dataPtr;
    unsigned int vbo;
    unsigned int bufferType;
    int size;
    int context;
    bool processing;
};

struct BufferInfo
{
    int offset;
    int size;
    unsigned int vbo;
    unsigned int timestamp;
    unsigned int bufferType;
    bool cudaReg;
};

struct LoadVBOParams
{
    bool quit;
    pthread_mutex_t * quitLock;
    std::map<int,std::list<BufferJob*> > * loadQueue;
    std::map<int,std::list<BufferJob*> > * doneQueue;
    pthread_mutex_t * queueLock;
};

class VBOCache
{
    public:
        VBOCache(int size);
        ~VBOCache();

        unsigned int getOrRequestBuffer(int context, int file, int offset, int size, unsigned int bufferType, bool cudaReg = false);
        int getFileID(std::string file);
        void update(int context);
        void advanceTime();

        void freeResources(int context);
        bool freeDone();

    protected:
        void getOrCreateBuffer(BufferJob * job);

        int _maxSize;
        int _currentSize;

        std::map<int,std::list<BufferJob*> > _fetchDataQueue;
        std::map<int,std::list<BufferJob*> > _loadVBOQueue;
        std::map<int,std::list<BufferJob*> > _doneQueue;
        std::map<int,std::map<int,std::list<BufferInfo*> > > _loadedBuffers;
        pthread_mutex_t _queueLock;

        std::map<std::string,int> _fileIDMap;
        std::map<int,std::string> _fileNameMap;
        pthread_mutex_t _fileNameLock;

        std::vector<pthread_t> _threadList;
        std::vector<pthread_mutex_t> _quitMutexList;
        std::vector<LoadVBOParams> _paramList;

        unsigned int _currentTimestamp;

        DataLoader * _dataLoader;
};

#endif
