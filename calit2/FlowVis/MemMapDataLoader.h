#ifndef MEM_MAP_DATA_LOADER_H
#define MEM_MAP_DATA_LOADER_H

#include "DataLoader.h"

#include <vector>

void setUseAllCores(pthread_t & thread);

struct MappedFileInfo
{
    int fileID;
    int fd;
    void * ptr;
    size_t fileSize;
    unsigned int timestamp;
};

class MemMapDataLoader : public DataLoader
{
    public:
        MemMapDataLoader();
        virtual ~MemMapDataLoader();

        void init(std::map<int,std::list<BufferJob*> > * fetchQueue, std::map<int,std::list<BufferJob*> > * vboQueue, pthread_mutex_t * queueLock);

        void run();

    protected:
        void mapFileForJob(BufferJob * job);

        std::map<int,std::list<BufferJob*> > * _fetchQueue;
        std::map<int,std::list<BufferJob*> > * _vboQueue;
        pthread_mutex_t * _queueLock;

        pthread_mutex_t _quitLock;
        bool _quit;

        std::vector<pthread_t> _threadList;
        std::map<int,MappedFileInfo> _mMap;
        pthread_mutex_t _mapLock;

        unsigned int _counter;
        int _maxMappedFiles;
};

#endif
