#include <GL/glew.h>
#include "VBOCache.h"
#include "MemMapDataLoader.h"

#include <time.h>
#include <cstring>
#include <climits>
#include <iostream>

#ifndef WIN32
#include <sys/time.h>
#else
#include <cvrUtil/TimeOfDay.h>
#endif

#ifdef WITH_CUDA_LIB
#include "CudaHelper.h"
#endif

#ifdef WIN32
bool wsaInit = false;

int usleep(long usec)
{
    struct timeval tv;
    fd_set dummy;
    SOCKET s = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);
    FD_ZERO(&dummy);
    FD_SET(s, &dummy);
    tv.tv_sec = usec/1000000L;
    tv.tv_usec = usec%1000000L;
    return select(0, 0, 0, &dummy, &tv);
}

#endif

void * loadVBOThread(void * arg)
{
    LoadVBOParams * params = (LoadVBOParams*)arg;

    std::list<std::pair<BufferJob*,struct timeval> > waitList;
    float waitInterval = 0.02f;

    while(1)
    {
	pthread_mutex_lock(params->quitLock);
	if(params->quit)
	{
	    pthread_mutex_unlock(params->quitLock);
	    break;
	}
	pthread_mutex_unlock(params->quitLock);

	struct timeval now;
	gettimeofday(&now,NULL);

	for(std::list<std::pair<BufferJob*,struct timeval> >::iterator it = waitList.begin(); it != waitList.end(); )
	{
	    float interval = (now.tv_sec - it->second.tv_sec) + ((now.tv_usec - it->second.tv_usec)/1000000.0f);
	    if(interval > waitInterval)
	    {
		pthread_mutex_lock(params->queueLock);
		for(std::list<BufferJob*>::iterator listIt = (*params->loadQueue)[it->first->context].begin(); listIt != (*params->loadQueue)[it->first->context].end(); ++listIt)
		{
		    if((*listIt) == it->first)
		    {
			(*params->doneQueue)[it->first->context].push_back(*listIt);
			(*params->loadQueue)[it->first->context].erase(listIt);
			break;
		    }
		}
		pthread_mutex_unlock(params->queueLock);
		it = waitList.erase(it);
	    }
	    else
	    {
		++it;
	    }
	}

	BufferJob * myJob = NULL;

	pthread_mutex_lock(params->queueLock);
    
	for(std::map<int,std::list<BufferJob*> >::iterator it = params->loadQueue->begin(); it != params->loadQueue->end(); ++it)
	{
	    for(std::list<BufferJob*>::iterator listIt = it->second.begin(); listIt != it->second.end(); ++listIt)
	    {
		if(!(*listIt)->processing)
		{
		    (*listIt)->processing = true;
		    myJob = (*listIt);
		    break;
		}
	    }
	    if(myJob)
	    {
		break;
	    }
	}

	pthread_mutex_unlock(params->queueLock);

	if(myJob)
	{
	    memcpy(myJob->mappedPtr,myJob->dataPtr,myJob->size);
	    struct timeval tv;
	    gettimeofday(&tv,NULL);
	    waitList.push_back(std::pair<BufferJob*,struct timeval>(myJob,tv));
	}
	else
	{
#ifndef WIN32
	    struct timespec ts;
	    ts.tv_sec = 0;
	    ts.tv_nsec = 50000;
	    nanosleep(&ts,NULL);
#else
		usleep(50);
#endif
	}
    }

    return NULL;
}

VBOCache::VBOCache(int size)
{
    _maxSize = size;
    _currentSize = 0;
    _currentTimestamp = 0;

#ifdef WIN32
	if(!wsaInit)
	{
		WORD wVersionRequested = MAKEWORD(1,0);
		WSADATA wsaData;
		WSAStartup(wVersionRequested, &wsaData);
		wsaInit = true;
	}
#endif

    //TODO: read from config
    int numThreads = 2;
    
    pthread_mutex_init(&_queueLock,NULL);
    pthread_mutex_init(&_fileNameLock,NULL);

    for(int i = 0; i < numThreads; ++i)
    {
	pthread_mutex_t mutex;
	_quitMutexList.push_back(mutex);
    }

    for(int i = 0; i < numThreads; ++i)
    {
	pthread_mutex_init(&_quitMutexList[i],NULL);
	_paramList.push_back(LoadVBOParams());
	_paramList[i].quit = false;
	_paramList[i].quitLock = &_quitMutexList[i];
	_paramList[i].loadQueue = &_loadVBOQueue;
	_paramList[i].doneQueue = &_doneQueue;
	_paramList[i].queueLock = &_queueLock;
    }

    for(int i = 0; i < numThreads; ++i)
    {
	pthread_t thread;

	pthread_create(&thread,NULL,loadVBOThread,(void*)&_paramList[i]);
	setUseAllCores(thread);

	_threadList.push_back(thread);
    }

    _dataLoader = new MemMapDataLoader();
    _dataLoader->init(&_fetchDataQueue,&_loadVBOQueue,&_queueLock);
}

VBOCache::~VBOCache()
{
    for(int i = 0; i < _threadList.size(); ++i)
    {
	pthread_mutex_lock(_paramList[i].quitLock);
	_paramList[i].quit = true;
	pthread_mutex_unlock(_paramList[i].quitLock);
    }

    for(int i = 0; i < _threadList.size(); ++i)
    {
	pthread_join(_threadList[i],NULL);
    }

    delete _dataLoader;
}

unsigned int VBOCache::getOrRequestBuffer(int context, int file, int offset, int size, unsigned int bufferType, bool cudaReg)
{
    pthread_mutex_lock(&_queueLock);

    for(std::list<BufferInfo*>::iterator it = _loadedBuffers[context][file].begin(); it != _loadedBuffers[context][file].end(); ++it)
    {
	if((*it)->offset == offset)
	{
	    (*it)->timestamp = _currentTimestamp;
	    unsigned int vbo = (*it)->vbo;
#ifdef WITH_CUDA_LIB
	    if(cudaReg && !(*it)->cudaReg)
	    {
		checkRegBufferObj(vbo);
		(*it)->cudaReg = true;
	    }
#endif
	    pthread_mutex_unlock(&_queueLock);
	    return vbo;
	}
    }

    for(std::list<BufferJob*>::iterator it = _fetchDataQueue[context].begin(); it != _fetchDataQueue[context].end(); ++it)
    {
	if((*it)->file == file && (*it)->offset == offset)
	{
	    pthread_mutex_unlock(&_queueLock);
	    return 0;
	}
    }

    for(std::list<BufferJob*>::iterator it = _loadVBOQueue[context].begin(); it != _loadVBOQueue[context].end(); ++it)
    {
	if((*it)->file == file && (*it)->offset == offset)
	{
	    pthread_mutex_unlock(&_queueLock);
	    return 0;
	}
    }

    for(std::list<BufferJob*>::iterator it = _doneQueue[context].begin(); it != _doneQueue[context].end(); ++it)
    {
	if((*it)->file == file && (*it)->offset == offset)
	{
	    pthread_mutex_unlock(&_queueLock);
	    return 0;
	}
    }

    // not found

    BufferJob * job = new BufferJob;
    pthread_mutex_lock(&_fileNameLock);
    job->fileName = _fileNameMap[file];
    pthread_mutex_unlock(&_fileNameLock);
    job->file = file;
    job->offset = offset;
    job->mappedPtr = NULL;
    job->dataPtr = NULL;
    job->vbo = 0;
    job->bufferType = bufferType;
    job->size = size;
    job->context = context;
    job->processing = false;

    getOrCreateBuffer(job,context);

    if(job->vbo)
    {
	_fetchDataQueue[context].push_back(job);
    }
    else
    {
	delete job;
    }

    pthread_mutex_unlock(&_queueLock);

	return 0;
}

int VBOCache::getFileID(std::string file)
{
    pthread_mutex_lock(&_fileNameLock);
    if(_fileIDMap.find(file) == _fileIDMap.end())
    {
	_fileIDMap[file] = (int)_fileIDMap.size();
	_fileNameMap[_fileIDMap[file]] = file;
    }
    int fileid = _fileIDMap[file];
    pthread_mutex_unlock(&_fileNameLock);
    return fileid;
}

void VBOCache::update(int context)
{
    pthread_mutex_lock(&_queueLock);

    for(std::list<BufferJob*>::iterator it = _doneQueue[context].begin(); it != _doneQueue[context].end(); ++it)
    {
	BufferInfo * bi = new BufferInfo;
	bi->offset = (*it)->offset;
	bi->size = (*it)->size;
	bi->vbo = (*it)->vbo;
	bi->timestamp = _currentTimestamp;
	bi->bufferType = (*it)->bufferType;
	bi->cudaReg = false;
	_loadedBuffers[(*it)->context][(*it)->file].push_back(bi);
    
	glBindBuffer(bi->bufferType,bi->vbo);
	glUnmapBuffer(bi->bufferType);
	glBindBuffer(bi->bufferType,0);

	delete (*it);
    }
    _doneQueue[context].clear();

    pthread_mutex_unlock(&_queueLock);
}

void VBOCache::advanceTime()
{
    _currentTimestamp++;
}

void VBOCache::freeResources(int context)
{
    pthread_mutex_lock(&_queueLock);

    for(std::map<int,std::list<BufferInfo*> >::iterator fileIt = _loadedBuffers[context].begin(); fileIt != _loadedBuffers[context].end(); ++fileIt)
    {
	for(std::list<BufferInfo*>::iterator it = fileIt->second.begin(); it != fileIt->second.end();)
	{
#ifdef WITH_CUDA_LIB
	    if((*it)->cudaReg)
	    {
		checkUnregBufferObj((*it)->vbo);
	    }
#endif
	    glDeleteBuffers(1,&(*it)->vbo);
	    it = fileIt->second.erase(it);
	}
    }

    pthread_mutex_unlock(&_queueLock);
}

bool VBOCache::freeDone()
{
    bool done = true;

    pthread_mutex_lock(&_queueLock);

    for(std::map<int,std::list<BufferJob*> >::iterator it = _fetchDataQueue.begin(); it != _fetchDataQueue.end(); ++it)
    {
	if(it->second.size())
	{
	    done = false;
	    break;
	}
    }

    for(std::map<int,std::list<BufferJob*> >::iterator it = _loadVBOQueue.begin(); it != _loadVBOQueue.end(); ++it)
    {
	if(it->second.size())
	{
	    done = false;
	    break;
	}
    }

    for(std::map<int,std::list<BufferJob*> >::iterator it = _doneQueue.begin(); it != _doneQueue.end(); ++it)
    {
	if(it->second.size())
	{
	    done = false;
	    break;
	}
    }

    for(std::map<int,std::map<int,std::list<BufferInfo*> > >::iterator it = _loadedBuffers.begin(); it != _loadedBuffers.end(); ++it)
    {
	for(std::map<int,std::list<BufferInfo*> >::iterator it2 = it->second.begin(); it2 != it->second.end(); ++it2)
	{
	    if(it2->second.size())
	    {
		done = false;
		break;
	    }
	}
    }

    pthread_mutex_unlock(&_queueLock);

    return done;
}

void VBOCache::getOrCreateBuffer(BufferJob * job, int context)
{
    int sizek = job->size / 1024;
    while(sizek + _currentSize > _maxSize)
    {
	// find buffer to evict
	unsigned int oldestTime = UINT_MAX;
	std::list<BufferInfo*>::iterator oldestIt;
	int oldestContext, oldestFile;
	for(std::map<int,std::list<BufferInfo*> >::iterator fit = _loadedBuffers[context].begin(); fit != _loadedBuffers[context].end(); ++fit)
	{
	    for(std::list<BufferInfo*>::iterator lit = fit->second.begin(); lit != fit->second.end(); ++lit)
	    {
		if((*lit)->timestamp < oldestTime)
		{
		    oldestIt = lit;
		    oldestContext = context;
		    oldestFile = fit->first;
		    oldestTime = (*lit)->timestamp;
		}
	    }
	}

	if(oldestTime != UINT_MAX)
	{
#ifdef WITH_CUDA_LIB
	    if((*oldestIt)->cudaReg)
	    {
		checkUnregBufferObj((*oldestIt)->vbo);
	    }
#endif
	    //glBindBuffer((*oldestIt)->bufferType,(*oldestIt)->vbo);
	    //glUnmapBuffer((*oldestIt)->bufferType);
	    //glBindBuffer((*oldestIt)->bufferType,0);
	    glDeleteBuffers(1,&(*oldestIt)->vbo);
	    _currentSize -= (*oldestIt)->size / 1024;
	    delete (*oldestIt);
	    _loadedBuffers[oldestContext][oldestFile].erase(oldestIt);
	}
	else
	{
	    std::cerr << "No Buffer found to evict!" << std::endl;
	    return;
	}
    }

    _currentSize += sizek;

    glGenBuffers(1,&job->vbo);

    glBindBuffer(job->bufferType,job->vbo);

    glBufferData(job->bufferType,job->size,NULL,GL_STATIC_DRAW);
    job->mappedPtr = glMapBufferRange(job->bufferType,0,job->size,GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);

    //glBufferStorage(job->bufferType,job->size,NULL,GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT);
    //job->mappedPtr = glMapBufferRange(job->bufferType,0,job->size,GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT);

    glBindBuffer(job->bufferType,0);
}
