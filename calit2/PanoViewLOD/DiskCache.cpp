#include "DiskCache.h"
#include "sph-cache.hpp"

#include <config/ConfigManager.h>

#include <queue>
#include <iostream>
#include <unistd.h>
#include <time.h>
#include <cstring>
#include <climits>

#define DC_PRINT_DEBUG

using namespace cvr;

OpenThreads::Mutex freeThreadLock;
OpenThreads::Mutex listLock;
OpenThreads::Mutex mapLock;
OpenThreads::Mutex pageCountLock;
OpenThreads::Mutex cleanupLock;
OpenThreads::Mutex loadsLock;
OpenThreads::Mutex ejectLock;

JobThread::JobThread(int id, JobType jt, std::vector<std::list<std::pair<sph_task*, CopyJobInfo*> > > * readlist, std::vector<std::vector<std::list<std::pair<sph_task*, CopyJobInfo*> > > > * copylist, std::list<JobThread*> * freeThreadList, std::map<int, std::map<int,DiskCacheEntry*> > * cacheMap, std::map<sph_cache*,int> * cacheIndexMap) : _jt(jt), _readList(readlist), _copyList(copylist), _freeThreadList(freeThreadList), _cacheMap(cacheMap), _cacheIndexMap(cacheIndexMap)
{
    _quit = false;
    _readIndex = 0;
    _copyCacheNum = 0;
    _copyFileNum = 0;
    _id = id;
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
#ifdef DC_PRINT_DEBUG
		std::cerr << "Read Thread grabbed job f: " << task->f << " i: " << task->i << " t: " << task->timestamp << std::endl;
#endif
		_readIndex++;
		break;
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
#ifdef DC_PRINT_DEBUG
		    std::cerr << "Done reading data f: " << task->f << " i: " << task->i << std::endl;
#endif
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
	cji->lock.lock();
	cji->refs--;
	cji->lock.unlock();
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
   
    bool found = false;
    
    listLock.lock();

    if(_copyList->size())
    {
	_copyCacheNum = _copyCacheNum % _copyList->size();
	_copyFileNum = _copyFileNum % _copyList->at(_copyCacheNum).size();
	int lastFile = _copyFileNum;
	for(int i = 0; i < _copyList->size(); i++)
	{
	    for(; _copyFileNum < _copyList->at(_copyCacheNum).size(); _copyFileNum++)
	    {
		if(_copyList->at(_copyCacheNum)[_copyFileNum].size())
		{
		    for(std::list<std::pair<sph_task*, CopyJobInfo*> >::iterator it = _copyList->at(_copyCacheNum)[_copyFileNum].begin(); it != _copyList->at(_copyCacheNum)[_copyFileNum].end(); it++)
		    {
			if(it->second->valid || !it->second->size)
			{
			    taskpair = *it;
			    _copyList->at(_copyCacheNum)[_copyFileNum].erase(it);
			    found = true;
			    break;
			}
		    }
		    if(found)
		    {
			_copyFileNum++;
			break;
		    }
		    //taskpair = _copyList->at(_copyCacheNum)[_copyFileNum].front();
		    //_copyList->at(_copyCacheNum)[_copyFileNum].pop_front();
		    //found = true;
		    //_copyFileNum++;
		    //break;
		}
	    }
	    if(found)
	    {
		if(_copyFileNum == _copyList->at(_copyCacheNum).size())
		{
		    _copyCacheNum++;
		    _copyCacheNum = _copyCacheNum % _copyList->size();
		    _copyFileNum = 0;
		}
		break;
	    }
	    _copyFileNum = 0;
	    _copyCacheNum++;
	    _copyCacheNum = _copyCacheNum % _copyList->size();  
	}

	if(!found)
	{
	    for(_copyFileNum = 0; _copyFileNum < lastFile; _copyFileNum++)
	    {
		if(_copyList->at(_copyCacheNum)[_copyFileNum].size())
		{
		    for(std::list<std::pair<sph_task*, CopyJobInfo*> >::iterator it = _copyList->at(_copyCacheNum)[_copyFileNum].begin(); it != _copyList->at(_copyCacheNum)[_copyFileNum].end(); it++)
		    {
			if(it->second->valid || !it->second->size)
			{
			    taskpair = *it;
			    _copyList->at(_copyCacheNum)[_copyFileNum].erase(it);
			    found = true;
			    break;
			}
		    }
		    if(found)
		    {
			_copyFileNum++;
			break;
		    }
		    //taskpair = _copyList->at(_copyCacheNum)[_copyFileNum].front();
		    //_copyList->at(_copyCacheNum)[_copyFileNum].pop_front();
		    //found = true;
		    //_copyFileNum++;
		    //break;
		}
	    }
	}
    }

#ifdef DC_PRINT_DEBUG
    int cjobsize = 0;
    for(int i = 0; i < _copyList->size(); i++)
    {
	for(int j = 0; j < _copyList->at(i).size(); j++)
	{
	    cjobsize += _copyList->at(i)[j].size();
	}
    }
    if(cjobsize)
    {
	std::cerr << "Copy Jobs left: " << cjobsize << std::endl;
    }
#endif


    /*if(_copyList->size())
    {
	for(int i = 0; i < _copyList->size(); i++)
	{
	    _copyIndex = _copyIndex % _copyList->size();
	    if(_copyList->at(_copyIndex).size())
	    {
		taskpair = _copyList->at(_copyIndex).front();
		_copyList->at(_copyIndex).pop_front();
		_copyIndex++;
		break;
	    }
	    _copyIndex++;
	}
    }*/
    listLock.unlock();

    if(taskpair.first)
    {
#ifdef DC_PRINT_DEBUG
	std::cerr << "Copy thread: " << _id << " got task f: " << taskpair.first->f << " i: " << taskpair.first->i << std::endl;
	if(_jt == READ_THREAD)
	{
	    std::cerr << "Copy done by read thread." << std::endl;
	}
#endif
	unsigned char * currentP = taskpair.second->data;
	unsigned int currentCopied = 0;
	unsigned int size = 0;
	do
	{
	    unsigned int currentValid;
	    taskpair.second->lock.lock();
	    currentValid = taskpair.second->valid;
	    size = taskpair.second->size;
	    taskpair.second->lock.unlock();

	    if(currentCopied < currentValid)
	    {
		unsigned char * dest = ((unsigned char*)taskpair.first->p) + currentCopied;
		unsigned char * src = taskpair.second->data + currentCopied;
		memcpy(dest,src,currentValid - currentCopied);
		currentCopied = currentValid;
	    }

	} while(currentCopied < size);
	taskpair.second->lock.lock();
	taskpair.second->refs--;
	if(!taskpair.second->size)
	{
	    taskpair.first->valid = false;
	}
	loadsLock.lock();
	taskpair.first->cache->loads.insert(*(taskpair.first));
	loadsLock.unlock();
	taskpair.second->lock.unlock();
#ifdef DC_PRINT_DEBUG
	std::cerr << "Copy thread: " << _id << " finished task f: " << taskpair.first->f << " i: " << taskpair.first->i << std::endl;
#endif
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
	cji->lock.lock();
	cji->valid = 0;
	cji->refs++;

	if(cji->size == 0)
	{
	    // if job is invalid, forward through to copy op
	    task->valid = false;
	    cji->lock.unlock();
	    listLock.lock();

	    _copyList->at((*_cacheIndexMap)[task->cache])[task->f].push_back(std::pair<sph_task*,CopyJobInfo*>(task,cji));

	    listLock.unlock();
	    return;
	}

	unsigned int linesize = TIFFScanlineSize(T);
	unsigned int totalData = W*H*4;
	unsigned char * data = new unsigned char[totalData];
	unsigned int currentValid = 0;

	cji->data = data;
	cji->size = totalData;
	cji->lock.unlock();

	listLock.lock();

	_copyList->at((*_cacheIndexMap)[task->cache])[task->f].push_back(std::pair<sph_task*,CopyJobInfo*>(task,cji));

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
	_readThreads.push_back(new JobThread(numCopyThreads + i,READ_THREAD, &_readList, &_copyList, &_freeThreadList, &_cacheMap, &_cacheIndexMap));
	//_readThreads[i]->setProcessorAffinity((1+i) % cores);
	_readThreads[i]->startThread();
    }

    freeThreadLock.lock();
    for(int i = 0; i < numCopyThreads; i++)
    {
#ifdef DC_PRINT_DEBUG
	std::cerr << "Starting copy thread: " << i << std::endl;
#endif
	_copyThreads.push_back(new JobThread(i, COPY_THREAD, &_readList, &_copyList, &_freeThreadList, &_cacheMap, &_cacheIndexMap));
	//_copyThreads[i]->setProcessorAffinity((1 + numReadThreads + i) % cores);
	_copyThreads[i]->startThread();
	_freeThreadList.push_back(_copyThreads[i]);
    }
    freeThreadLock.unlock();

    _prevFileL = _prevFileR = -1;
    _currentFileL = _currentFileR = -1;
    _nextFileL = _nextFileR = -1;
}

DiskCache::~DiskCache()
{
    for(int i = 0; i < _readThreads.size(); i++)
    {
	if(_readThreads[i]->isRunning())
	{
	    _readThreads[i]->quit();
	    _readThreads[i]->join();
	    delete _readThreads[i];
	}
    }

    for(int i = 0; i < _copyThreads.size(); i++)
    {
	if(_copyThreads[i]->isRunning())
	{
	    _copyThreads[i]->quit();
	    _copyThreads[i]->join();
	    delete _copyThreads[i];
	}
    }

    for(int i = 0; i < _readList.size(); i++)
    {
	for(std::list<std::pair<sph_task*, CopyJobInfo*> >::iterator it = _readList[i].begin(); it != _readList[i].end(); it++)
	{
	    delete it->first;
	}
    }

    for(int i = 0; i < _copyList.size(); i++)
    {
	for(int j = 0; j < _copyList[i].size(); j++)
	{
	    for(std::list<std::pair<sph_task*, CopyJobInfo*> >::iterator it = _copyList[i][j].begin(); it != _copyList[i][j].end(); it++)
	    {
		delete it->first;
	    }
	}
    }

    for(std::map<int, std::map<int,DiskCacheEntry*> >::iterator it = _cacheMap.begin(); it != _cacheMap.end(); it++)
    {
	for(std::map<int,DiskCacheEntry*>::iterator it2 = it->second.begin(); it2 != it->second.end(); it2++)
	{
	    if(it2->second->cji->data)
	    {
		delete[] it2->second->cji->data;
	    }
	    delete it2->second->cji;
	    delete it2->second;
	}
    }
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
	listLock.lock();

	_fileIDMap[name] = _nextID;
	id = _nextID;

	_cacheMap[id] = std::map<int,DiskCacheEntry*>();

	_readList.push_back(std::list<std::pair<sph_task*, CopyJobInfo*> >());

	for(int i = 0; i < _copyList.size(); i++)
	{
	    _copyList[i].push_back(std::list<std::pair<sph_task*, CopyJobInfo*> >());
	}

	listLock.unlock();

	_nextID++;
    }

    _fileAddLock.unlock();

#ifdef DC_PRINT_DEBUG
    std::cerr << "add_file called with name: " << name << " given ID: " << id << std::endl;
#endif

    return id;
}

void DiskCache::add_task(sph_task * task)
{

    
    /*int curPages = 0;
    mapLock.lock();
    for(std::map<int, std::map<int,DiskCacheEntry*> >::iterator it = _cacheMap.begin(); it != _cacheMap.end(); it++)
    {
	curPages += it->second.size();
    }
    mapLock.unlock();
    std::cerr << "Actual Current Pages: " << curPages << std::endl;*/

    cleanup();

    if(task->f != _currentFileL && task->f != _currentFileR)
    {
	task->valid = false;
	loadsLock.lock();
	task->cache->loads.insert(*task);
	loadsLock.unlock();
	delete task;
	return;
    }

    if(_cacheIndexMap.find(task->cache) == _cacheIndexMap.end())
    {
	listLock.lock();
	int index = _cacheIndexMap.size();
	_cacheIndexMap[task->cache] = index;

	_copyList.push_back(std::vector<std::list<std::pair<sph_task*, CopyJobInfo*> > >());

	for(int i = 0; i < _fileIDMap.size(); i++)
	{
	    _copyList.back().push_back(std::list<std::pair<sph_task*, CopyJobInfo*> >());
	}

	listLock.unlock();
    }

    mapLock.lock();
    if(!_cacheMap[task->f][task->i])
    {
	_cacheMap[task->f][task->i] = new DiskCacheEntry;
	_cacheMap[task->f][task->i]->timestamp = task->timestamp;
	_cacheMap[task->f][task->i]->f = task->f;
	_cacheMap[task->f][task->i]->i = task->i;

	CopyJobInfo * cji = new CopyJobInfo;
	cji->data = NULL;
	cji->size = 100;
	cji->valid = 0;
	cji->refs = 2;

	_cacheMap[task->f][task->i]->cji = cji;

	mapLock.unlock();


	listLock.lock();

#ifdef DC_PRINT_DEBUG
	std::cerr << "Job started for task f: " << task->f << " i: " << task->i << " t: " << task->timestamp << std::endl;
#endif
	_readList[task->f].push_back(std::pair<sph_task*, CopyJobInfo*>(task,cji));

	listLock.unlock();


	pageCountLock.lock();

	_numPages++;

	/*while(_numPages >= _pages)
	{
	    pageCountLock.unlock();
	    eject();
	    pageCountLock.lock();
	}*/

	pageCountLock.unlock();
    }
    else
    {
	CopyJobInfo * cji = _cacheMap[task->f][task->i]->cji;

	cji->lock.lock();
	if(!cji->size == 0)
	{
	    _cacheMap[task->f][task->i]->timestamp = task->timestamp;
	    cji->refs++;
	    cji->lock.unlock();

	    mapLock.unlock();

#ifdef DC_PRINT_DEBUG
	    std::cerr << "DiskCache Hit f: " << task->f << " i: " << task->i << " t: " << task->timestamp << std::endl;
#endif

	    listLock.lock();

	    _copyList[_cacheIndexMap[task->cache]][task->f].push_back(std::pair<sph_task*, CopyJobInfo*>(task,cji));

	    listLock.unlock();
	}
	else
	{
	    cji->lock.unlock();
	    mapLock.unlock();

#ifdef DC_PRINT_DEBUG
	    std::cerr << "DiskCache Hit on deleting entry f: " << task->f << " i: " << task->i << std::endl;
#endif

	    task->valid = false;
	    loadsLock.lock();
	    task->cache->loads.insert(*task);
	    loadsLock.unlock();
	    delete task;
	}
    }

    int tries = 3;

    pageCountLock.lock();

    while(_numPages >= _pages && tries > 0)
    {
	pageCountLock.unlock();
	ejectLock.lock();
	eject();
	ejectLock.unlock();
	tries--;
	pageCountLock.lock();
    }

    pageCountLock.unlock();
}

void DiskCache::kill_tasks(int file)
{
    if(file < 0 || file >= _readList.size())
    {
	return;
    }

    listLock.lock();

    std::queue<sph_task> tempq;

    loadsLock.lock();

    for(std::map<sph_cache*,int>::iterator it = _cacheIndexMap.begin(); it != _cacheIndexMap.end(); it++)
    {
	while(!it->first->loads.empty())
	{
	    tempq.push(it->first->loads.remove());
	    if(tempq.front().f == file)
	    {
		tempq.front().valid = false;
	    }
	}

	while(tempq.size())
	{
	    it->first->loads.insert(tempq.front());
	    tempq.pop();
	}
    }

    loadsLock.unlock();

    for(int i = 0; i < _copyList.size(); i++)
    {
        for(std::list<std::pair<sph_task*, CopyJobInfo*> >::iterator it = _copyList[i][file].begin(); it != _copyList[i][file].end(); it++)
        {
            it->first->valid = false;
	    loadsLock.lock();
            it->first->cache->loads.insert(*(it->first));
	    loadsLock.unlock();
	    it->second->lock.lock();
	    it->second->refs--;
	    it->second->lock.unlock();
            delete it->first;
        }
        _copyList[i][file].clear();
    }


    for(std::list<std::pair<sph_task*, CopyJobInfo*> >::iterator it = _readList[file].begin(); it != _readList[file].end(); it++)
    {
	it->first->valid = false;
	loadsLock.lock();
	it->first->cache->loads.insert(*(it->first));
	loadsLock.unlock();

	it->second->lock.lock();
	it->second->refs--;
	it->second->size = 0;
	it->second->lock.unlock();

	cleanupLock.lock();
	_cleanupList.push_back(std::pair<int,int>(it->first->f,it->first->i));
	cleanupLock.unlock();

	delete it->first;

	pageCountLock.lock();
	_numPages--;
	pageCountLock.unlock();
	/*if(!it->second->refs)
	{
	    _cacheMap[it->first->f].erase(it->first->i);
	    it->second->lock.unlock();
	    delete it->second;
	    delete it->first;
	}
	else
	{
	    it->second->size = 0;
	    _cleanupList.push_back(std::pair<int,int>(it->first->f,it->first->i));
	    it->second->lock.unlock();
	}*/
    }

    _readList[file].clear();

    listLock.unlock();
}

void DiskCache::eject()
{
#ifdef DC_PRINT_DEBUG
    std::cerr << "DiskCache::eject()" << std::endl;
#endif

    mapLock.lock();
    // find oldest referenced page

    int f = -1;
    int i = -1;
    int mint = INT_MAX;

    for(std::map<int, std::map<int,DiskCacheEntry*> >::iterator it = _cacheMap.begin(); it != _cacheMap.end(); it++)
    {
	for(std::map<int,DiskCacheEntry*>::iterator it2 = it->second.begin(); it2 != it->second.end(); it2++)
	{
	    if(!it2->second)
	    {
		continue;
	    }

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
    _cacheMap[f][i]->timestamp = INT_MAX;
    mapLock.unlock();

#ifdef DC_PRINT_DEBUG
    std::cerr << "Eject f: " << f << " i: " << i << " t: " << mint << std::endl;
#endif

    _cacheMap[f][i]->cji->lock.lock();

    if(!_cacheMap[f][i]->cji->size)
    {
#ifdef DC_PRINT_DEBUG
	std::cerr << "Already Ejecting f: " << f << " i: " << i << " t: " << mint << std::endl;
#endif
	_cacheMap[f][i]->cji->lock.unlock();

	return;
    }
    
    _cacheMap[f][i]->cji->size = 0;
    _cacheMap[f][i]->cji->lock.unlock();

    std::queue<sph_task> tempq;

    loadsLock.lock();

    for(std::map<sph_cache*,int>::iterator it = _cacheIndexMap.begin(); it != _cacheIndexMap.end(); it++)
    {
	while(!it->first->loads.empty())
	{
	    tempq.push(it->first->loads.remove());
	    if(tempq.front().f == f && tempq.front().i == i)
	    {
		tempq.front().valid = false;
	    }
	}

	while(tempq.size())
	{
	    it->first->loads.insert(tempq.front());
	    tempq.pop();
	}
    }

    loadsLock.unlock();

    cleanupLock.lock();
    _cleanupList.push_back(std::pair<int,int>(f,i));
    cleanupLock.unlock();

    /*delete[] _cacheMap[f][i]->cji->data;
    delete _cacheMap[f][i]->cji;
    delete _cacheMap[f][i];
    _cacheMap[f].erase(i);*/

    pageCountLock.lock();
    _numPages--;
    pageCountLock.unlock();
}

void DiskCache::setLeftFiles(int prev, int curr, int next)
{
    if(curr != _currentFileL)
    {
	if(_currentFileL != -1)
	{
	    kill_tasks(_currentFileL);
	}
    }

    _prevFileL = prev;
    _currentFileL = curr;
    _nextFileL = next;
}

void DiskCache::setRightFiles(int prev, int curr, int next)
{
    if(curr != _currentFileR)
    {
	if(_currentFileR != -1)
	{
	    kill_tasks(_currentFileR);
	}
    }

    _prevFileR = prev;
    _currentFileR = curr;
    _nextFileR = next;
}

void DiskCache::cleanup()
{
    cleanupLock.lock();
    mapLock.lock();

    for(std::list<std::pair<int,int> >::iterator it = _cleanupList.begin(); it != _cleanupList.end(); )
    {
#ifdef DC_PRINT_DEBUG
        std::cerr << "Cleanup f: " << it->first << " i: " << it->second << std::endl;
#endif
	if(_cacheMap.find(it->first) == _cacheMap.end() || _cacheMap[it->first].find(it->second) == _cacheMap[it->first].end())
	{
#ifdef DC_PRINT_DEBUG
	    std::cerr << "Double Cleanup f: " << it->first << " i: " << it->second << std::endl;
#endif
            it = _cleanupList.erase(it);
	    continue;
	}
	_cacheMap[it->first][it->second]->cji->lock.lock();
	if(_cacheMap[it->first][it->second]->cji->refs <= 1)
	{
	    _cacheMap[it->first][it->second]->cji->lock.unlock();
	    if(_cacheMap[it->first][it->second]->cji->data)
	    {
		delete[] _cacheMap[it->first][it->second]->cji->data;
	    }
	    delete _cacheMap[it->first][it->second]->cji;
	    delete _cacheMap[it->first][it->second];
	    _cacheMap[it->first].erase(it->second);
	    it = _cleanupList.erase(it);
	}
	else
	{
	    _cacheMap[it->first][it->second]->cji->lock.unlock();
	    it++;
	}
    }

    mapLock.unlock();
    cleanupLock.unlock();
}
