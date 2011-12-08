#include "DiskCache.h"
#include "sph-cache.hpp"

#include <config/ConfigManager.h>

#include <iostream>
#include <unistd.h>
#include <time.h>
#include <cstring>
#include <climits>

using namespace cvr;

OpenThreads::Mutex freeThreadLock;
OpenThreads::Mutex listLock;
OpenThreads::Mutex mapLock;
OpenThreads::Mutex pageCountLock;

JobThread::JobThread(JobType jt, std::vector<std::list<std::pair<sph_task*, CopyJobInfo*> > > * readlist, std::vector<std::list<std::pair<sph_task*, CopyJobInfo*> > > * copylist, std::list<JobThread*> * freeThreadList, std::map<int, std::map<int,DiskCacheEntry*> > * cacheMap, std::map<sph_cache*,int> * cacheIndexMap) : _jt(jt), _readList(readlist), _copyList(copylist), _freeThreadList(freeThreadList), _cacheMap(cacheMap), _cacheIndexMap(cacheIndexMap)
{
    _quit = false;
    _readIndex = 0;
    _copyIndex = 0;
}

JobThread::~JobThread()
{
}

void JobThread::run()
{
    while(1)
    {
	_quitLock.lock();

	if(_quit)
	{
	    _quitLock.unlock();
	    return;
	}

	_quitLock.unlock();

	if(_jt == READ_THREAD)
	{
	    read();
	}
	else
	{
	    copy();
	}
    }
}

void JobThread::quit()
{
    _quitLock.lock();

    _quit = true;

    _quitLock.unlock();
}

void JobThread::read()
{
    CopyJobInfo * cji = NULL;
    sph_task * task = NULL;

    listLock.lock();
    if(_readList->size())
    {
	for(int i = 0; i < _readList->size(); i++)
	{
	    _readIndex = _readIndex % _readList->size();
	    if(_readList->at(_readIndex).size())
	    {
		task = _readList->at(_readIndex).front().first;
		cji = _readList->at(_readIndex).front().second;
		_readList->at(_readIndex).pop_front();
		std::cerr << "Read Thread grabbed job f: " << task->f << " i: " << task->i << " t: " << task->timestamp << std::endl;
	    }
	    _readIndex++;
	}
    }
    listLock.unlock();

    if(task)
    {
	if (!task->cache->files[task->f].name.empty())
        {
            if (TIFF *T = TIFFOpen(task->cache->files[task->f].name.c_str(), "r"))
            {
                if (up(T, task->i))
                {
                    uint32 w = task->cache->files[task->f].w;
                    uint32 h = task->cache->files[task->f].h;
                    uint16 c = task->cache->files[task->f].c;
                    uint16 b = task->cache->files[task->f].b;
                
		    readData(task, cji,T,w,h,c,b);
		    std::cerr << "Done reading data f: " << task->f << " i: " << task->i << std::endl;
                    //task.load_texture(T, w, h, c, b);
                    //task->cache->loads.insert(*task);
                }
                TIFFClose(T);
            }
        }
	else
	{
	    delete task;
	}
    }
    else
    {
	// no read task, look for a copy one
	//std::cerr << "Read thread doing copy" << std::endl;
	copy();
    }
}

void JobThread::copy()
{
    std::pair<sph_task*, CopyJobInfo*> taskpair;
    
    listLock.lock();
    if(_copyList->size())
    {
	for(int i = 0; i < _copyList->size(); i++)
	{
	    _copyIndex = _copyIndex % _copyList->size();
	    if(_copyList->at(_copyIndex).size())
	    {
		taskpair = _copyList->at(_copyIndex).front();
		_copyList->at(_copyIndex).pop_front();
	    }
	    _copyIndex++;
	}
    }
    listLock.unlock();

    if(taskpair.first)
    {
	std::cerr << "Copy thread got task f: " << taskpair.first->f << " i: " << taskpair.first->i << std::endl;
	if(_jt == READ_THREAD)
	{
	    std::cerr << "Copy done by read thread." << std::endl;
	}
	unsigned char * currentP = taskpair.second->data;
	unsigned int currentCopied = 0;
	do
	{
	    unsigned int currentValid;
	    taskpair.second->lock.lock();
	    currentValid = taskpair.second->valid;
	    taskpair.second->lock.unlock();

	    if(currentCopied < currentValid)
	    {
		unsigned char * dest = ((unsigned char*)taskpair.first->p) + currentCopied;
		unsigned char * src = taskpair.second->data + currentCopied;
		memcpy(dest,src,currentValid - currentCopied);
		currentCopied = currentValid;
	    }

	} while(currentCopied < taskpair.second->size);
	taskpair.first->cache->loads.insert(*(taskpair.first));
	std::cerr << "Copy thread finished task f: " << taskpair.first->f << " i: " << taskpair.first->i << std::endl;
	delete taskpair.first;
    }
    else
    {
	struct timespec ts;
	ts.tv_sec = 0;
	ts.tv_nsec = 2000000;

	nanosleep(&ts,NULL);
    }
}


void JobThread::readData(sph_task * task, CopyJobInfo * cji, TIFF * T, uint32 w, uint32 h, uint16 c, uint16 b)
{
    // Confirm the page format.
    
    uint32 W, H;
    uint16 C, B;
    
    TIFFGetField(T, TIFFTAG_IMAGEWIDTH,      &W);
    TIFFGetField(T, TIFFTAG_IMAGELENGTH,     &H);
    TIFFGetField(T, TIFFTAG_BITSPERSAMPLE,   &B);
    TIFFGetField(T, TIFFTAG_SAMPLESPERPIXEL, &C);
    
    if (W == w && H == h && B == b && C == c)
    {

	unsigned int linesize = TIFFScanlineSize(T);
	unsigned int totalData = W*H*4;
	unsigned char * data = new unsigned char[totalData];
	unsigned int currentValid = 0;

	cji->lock.lock();
	cji->data = data;
	cji->size = totalData;
	cji->valid = 0;
	cji->lock.unlock();

	listLock.lock();

	_copyList->at((*_cacheIndexMap)[task->cache]).push_back(std::pair<sph_task*,CopyJobInfo*>(task,cji));

	listLock.unlock();

	// Pad a 24-bit image to 32-bit BGRA.

	if (c == 3 && b == 8)
	{
	    if (void *q = malloc(linesize))
	    {
		const uint32 S = w * 4 * b / 8;

		for (uint32 r = 0; r < h; ++r)
		{
		    TIFFReadScanline(T, q, r, 0);

		    for (int j = w - 1; j >= 0; --j)
		    {
			uint8 *s = (uint8 *) q         + j * c * b / 8;
			uint8 *d = (uint8 *) data + r * S + j * 4 * b / 8;

			d[0] = s[2];
			d[1] = s[1];
			d[2] = s[0];
			d[3] = 0xFF;
		    }
		    currentValid += S;
		    cji->lock.lock();
		    cji->valid = currentValid;
		    cji->lock.unlock();
		}
		free(q);
	    }
	}
	else
	{
	    for (uint32 r = 0; r < h; ++r)
	    {
		TIFFReadScanline(T, (uint8 *) data + r * linesize, r, 0);

		currentValid += linesize;
		cji->lock.lock();
		cji->valid = currentValid;
		cji->lock.unlock();
	    }
	}
    }
    else
    {
	delete task;
    }
}

DiskCache::DiskCache(int pages) : _pages(pages)
{
    _nextID = 0;
    _numPages = 0;

    int numReadThreads = ConfigManager::getInt("value","Plugin.PanoViewLOD.ReadThreads",1);
    int numCopyThreads = ConfigManager::getInt("value","Plugin.PanoViewLOD.CopyThreads",3);

    //int cores = OpenThreads::GetNumberOfProcessors();

    for(int i = 0; i < numReadThreads; i++)
    {
	_readThreads.push_back(new JobThread(READ_THREAD, &_readList, &_copyList, &_freeThreadList, &_cacheMap, &_cacheIndexMap));
	//_readThreads[i]->setProcessorAffinity((1+i) % cores);
	_readThreads[i]->startThread();
    }

    freeThreadLock.lock();
    for(int i = 0; i < numCopyThreads; i++)
    {
	std::cerr << "Starting copy thread: " << i << std::endl;
	_copyThreads.push_back(new JobThread(COPY_THREAD, &_readList, &_copyList, &_freeThreadList, &_cacheMap, &_cacheIndexMap));
	//_copyThreads[i]->setProcessorAffinity((1 + numReadThreads + i) % cores);
	_copyThreads[i]->startThread();
	_freeThreadList.push_back(_copyThreads[i]);
    }
    freeThreadLock.unlock();
}

DiskCache::~DiskCache()
{
    for(int i = 0; i < _readThreads.size(); i++)
    {
	if(_readThreads[i]->isRunning())
	{
	    _readThreads[i]->quit();
	    _readThreads[i]->join();
	}
    }

    for(int i = 0; i < _copyThreads.size(); i++)
    {
	if(_copyThreads[i]->isRunning())
	{
	    _copyThreads[i]->quit();
	    _copyThreads[i]->join();
	}
    }

    //TODO: free buffers
}

int DiskCache::add_file(const std::string& name)
{
    int id;
    _fileAddLock.lock();

    if(_fileIDMap.find(name) != _fileIDMap.end())
    {
	id = _fileIDMap[name];
    }
    else
    {
	_fileIDMap[name] = _nextID;
	id = _nextID;

	_cacheMap[id] = std::map<int,DiskCacheEntry*>();

	listLock.lock();
	_readList.push_back(std::list<std::pair<sph_task*, CopyJobInfo*> >());
	listLock.unlock();

	_nextID++;
    }

    _fileAddLock.unlock();

    std::cerr << "add_file called with name: " << name << " given ID: " << id << std::endl;

    return id;
}

void DiskCache::add_task(sph_task * task)
{
    if(_cacheIndexMap.find(task->cache) == _cacheIndexMap.end())
    {
	listLock.lock();
	int index = _cacheIndexMap.size();
	_cacheIndexMap[task->cache] = index;
	_copyList.push_back(std::list<std::pair<sph_task*, CopyJobInfo*> >());
	listLock.unlock();
    }

    if(!_cacheMap[task->f][task->i])
    {
	mapLock.lock();

	_cacheMap[task->f][task->i] = new DiskCacheEntry;
	_cacheMap[task->f][task->i]->timestamp = task->timestamp;
	_cacheMap[task->f][task->i]->f = task->f;
	_cacheMap[task->f][task->i]->i = task->i;

	CopyJobInfo * cji = new CopyJobInfo;
	cji->data = NULL;
	cji->size = 100;
	cji->valid = 0;

	_cacheMap[task->f][task->i]->cji = cji;

	mapLock.unlock();


	listLock.lock();

	std::cerr << "Job started for task f: " << task->f << " i: " << task->i << " t: " << task->timestamp << std::endl;
	_readList[task->f].push_back(std::pair<sph_task*, CopyJobInfo*>(task,cji));

	listLock.unlock();

	
	pageCountLock.lock();

	_numPages++;

	pageCountLock.unlock();

	while(_numPages >= _pages)
	{
	    eject();
	}
    }
    else
    {
	mapLock.lock();

	std::cerr << "DiskCache Hit." << std::endl;
	_cacheMap[task->f][task->i]->timestamp = task->timestamp;
	CopyJobInfo * cji = _cacheMap[task->f][task->i]->cji;
	mapLock.unlock();

	listLock.lock();

	_copyList[_cacheIndexMap[task->cache]].push_back(std::pair<sph_task*, CopyJobInfo*>(task,cji));

	listLock.unlock();
    }
}

void DiskCache::eject()
{
    std::cerr << "DiskCache::eject()" << std::endl;

    mapLock.lock();
    // find oldest referenced page

    int f = -1;
    int i = -1;
    int mint = INT_MAX;

    for(std::map<int, std::map<int,DiskCacheEntry*> >::iterator it = _cacheMap.begin(); it != _cacheMap.end(); it++)
    {
	for(std::map<int,DiskCacheEntry*>::iterator it2 = it->second.begin(); it2 != it->second.end(); it2++)
	{
	    if(it2->second->timestamp < mint)
	    {
		mint = it2->second->timestamp;
		f = it2->second->f;
		i = it2->second->i;
	    }
	}
    }

    if(f == -1)
    {
	std::cerr << "No Min page found!?!?..." << std::endl;
	mapLock.unlock();
	return;
    }

    delete[] _cacheMap[f][i]->cji->data;
    delete _cacheMap[f][i]->cji;
    delete _cacheMap[f][i];
    _cacheMap[f].erase(i);

    mapLock.unlock();

    pageCountLock.lock();
    _numPages--;
    pageCountLock.unlock();
}
