#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <pthread.h>
#include <map>
#include <list>

struct BufferJob;

class DataLoader
{
    public:
        virtual ~DataLoader()
        {
        }
        virtual void init(std::map<int,std::list<BufferJob*> > * fetchQueue, std::map<int,std::list<BufferJob*> > * vboQueue, pthread_mutex_t * queueLock) = 0;
};

#endif
