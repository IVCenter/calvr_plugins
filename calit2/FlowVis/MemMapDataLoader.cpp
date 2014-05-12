#include "MemMapDataLoader.h"
#include "VBOCache.h"

#include <iostream>
#include <climits>
#include <cstdio>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#ifndef WIN32
#include <sys/mman.h>
#else
#include <cvrUtil/TimeOfDay.h>
#include "mman.h"
#include "VBOCache.h"
#include <io.h>
#define close _close
#define open _open
#endif

void setUseAllCores(pthread_t & thread)
{
#ifndef WIN32
    int numCores = sysconf(_SC_NPROCESSORS_ONLN);

    if(numCores <= 0)
    {
	std::cerr << "Error getting number of cores." << std::endl;
	return;
    }

    cpu_set_t cpuset;

    CPU_ZERO(&cpuset);
    for(int i = 0; i < numCores; ++i)
    {
	CPU_SET(i,&cpuset);
    }

    int status = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
    if(status)
    {
	std::cerr << "Error setting cpu affinity." << std::endl;
    }
#else
	HANDLE process = GetCurrentProcess();

	DWORD_PTR processAM;
	DWORD_PTR systemAM;

	if(!GetProcessAffinityMask(process,&processAM,&systemAM))
	{
		std::cerr << "Error getting Affinity Mask." << std::endl;
		return;
	}

	if(!SetThreadAffinityMask(thread,processAM))
	{
		std::cerr << "Error setting cpu affinity." << std::endl;
	}
#endif
}

void * runLoader(void * arg)
{
    MemMapDataLoader * loader = (MemMapDataLoader*)arg;
    loader->run();
    return NULL;
}

MemMapDataLoader::MemMapDataLoader()
{
    _quit = false;
    pthread_mutex_init(&_quitLock,NULL);
    pthread_mutex_init(&_mapLock,NULL);
}

MemMapDataLoader::~MemMapDataLoader()
{
    pthread_mutex_lock(&_quitLock);
    _quit = true;
    pthread_mutex_unlock(&_quitLock);

    for(int i = 0; i < _threadList.size(); ++i)
    {
	pthread_join(_threadList[i],NULL);
    }

    for(std::map<int,MappedFileInfo>::iterator it = _mMap.begin(); it != _mMap.end(); ++it)
    {
	munmap(it->second.ptr,it->second.fileSize);
	close(it->second.fd);
    }
}

void MemMapDataLoader::init(std::map<int,std::list<BufferJob*> > * fetchQueue, std::map<int,std::list<BufferJob*> > * vboQueue, pthread_mutex_t * queueLock)
{
    _fetchQueue = fetchQueue;
    _vboQueue = vboQueue;
    _queueLock = queueLock;

    _counter = 0;

    //TODO: get from config
    _maxMappedFiles = 700;

    //TODO: get from config
    int threads = 1;

    for(int i = 0; i < threads; ++i)
    {
	pthread_t thread;
	pthread_create(&thread,NULL,runLoader,(void*)this);
	setUseAllCores(thread);
	_threadList.push_back(thread);
    }
}

void MemMapDataLoader::run()
{
    while(1)
    {
	pthread_mutex_lock(&_quitLock);
	if(_quit)
	{
	    pthread_mutex_unlock(&_quitLock);
	    break;
	}
	pthread_mutex_unlock(&_quitLock);

	pthread_mutex_lock(_queueLock);

	BufferJob * job = NULL;
	int context = 0;

	for(std::map<int,std::list<BufferJob*> >::iterator it = _fetchQueue->begin(); it != _fetchQueue->end(); ++it)
	{
	    for(std::list<BufferJob*>::iterator jit = it->second.begin(); jit != it->second.end(); ++jit)
	    {
		if(!(*jit)->processing)
		{
		    job = (*jit);
		    context = it->first;
		    job->processing = true;
		    break;
		}
	    }
	    if(job)
	    {
		break;
	    }
	}

	pthread_mutex_unlock(_queueLock);

	if(job)
	{
		struct timeval start, end;
		gettimeofday(&start,NULL);

	    pthread_mutex_lock(&_mapLock);
	    if(_mMap.find(job->file) == _mMap.end())
	    {
		mapFileForJob(job);
	    }
	    job->dataPtr = (void*)(((char*)_mMap[job->file].ptr) + job->offset);
	    _mMap[job->file].timestamp = _counter;
	    pthread_mutex_unlock(&_mapLock);

	    pthread_mutex_lock(_queueLock);
	    job->processing = false;
	    for(std::list<BufferJob*>::iterator it = (*_fetchQueue)[context].begin(); it != (*_fetchQueue)[context].end(); ++it)
	    {
		if((*it) == job)
		{
		    (*_fetchQueue)[context].erase(it);
		    break;
		}
	    }
	    (*_vboQueue)[context].push_back(job);
	    pthread_mutex_unlock(_queueLock);

		gettimeofday(&end,NULL);
		std::cerr << "MMDL Job time: " << (end.tv_sec - start.tv_sec) + ((end.tv_usec - start.tv_usec) / 1000000.0f) << std::endl;
	}
	else
	{
#ifndef WIN32
	    struct timespec ts;
	    ts.tv_sec = 0;
	    ts.tv_nsec = 50000;
	    nanosleep(&ts,NULL);
#else
		struct timeval start, end;
		gettimeofday(&start,NULL);
		usleep(50);
		gettimeofday(&end,NULL);
		std::cerr << "MMDL Sleep time: " << (end.tv_sec - start.tv_sec) + ((end.tv_usec - start.tv_usec) / 1000000.0f) << std::endl;
#endif
	}

	_counter++;
    }
}

void MemMapDataLoader::mapFileForJob(BufferJob * job)
{
    while(_mMap.size() >= _maxMappedFiles)
    {
	int oldestFile = -1;
	unsigned int age = UINT_MAX;
	
	for(std::map<int,MappedFileInfo>::iterator it = _mMap.begin(); it != _mMap.end(); ++it)
	{
	    if(it->second.timestamp < age)
	    {
		age = it->second.timestamp;
		oldestFile = it->second.fileID;
	    }
	}

	if(oldestFile > -1)
	{
	    std::map<int,MappedFileInfo>::iterator it = _mMap.find(oldestFile);
	    munmap(it->second.ptr,it->second.fileSize);
	    close(it->second.fd);
	    _mMap.erase(oldestFile);
	}
    }

    int fd = open(job->fileName.c_str(),O_RDONLY);
    if(fd == -1)
    {
	std::cerr << "Error opening file: " << job->fileName << std::endl;
	return;
    }

    size_t fileSize;
    struct stat st;
    stat(job->fileName.c_str(),&st);
    fileSize = st.st_size;

    void * mappedPtr = mmap(NULL,fileSize,PROT_READ,MAP_PRIVATE,fd,0);
    if(mappedPtr == (void*)-1)
    {
	std::cerr << "Error mapping file: " << job->fileName << std::endl;
	return;
    }

    _mMap[job->file].fileID = job->file;
    _mMap[job->file].fd = fd;
    _mMap[job->file].ptr = mappedPtr;
    _mMap[job->file].fileSize = fileSize;
}
