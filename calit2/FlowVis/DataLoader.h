#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#ifndef WIN32
#include <pthread.h>
#else
#include "pthread_win.h"
#endif

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
