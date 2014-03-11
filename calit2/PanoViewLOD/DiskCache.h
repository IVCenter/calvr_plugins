#ifndef PANOVIEWLOD_DISK_CACHE_H
#define PANOVIEWLOD_DISK_CACHE_H

#include <OpenThreads/Thread>
#include <OpenThreads/Mutex>

#include <map>
#include <string>
#include <list>
#include <vector>

#include <tiffio.h>

#include <sys/time.h>

struct sph_task;
class sph_cache;

enum JobType
{
    READ_THREAD = 0,
    COPY_THREAD
};

struct CopyJobInfo
{
    unsigned char * data;
    unsigned int size;
    unsigned int valid;
    int refs;
    OpenThreads::Mutex lock;
    struct timeval diskReadStart;
};

struct DiskCacheEntry
{
    CopyJobInfo * cji;
    int timestamp;
    int f;
    int i;
};

class JobThread: public OpenThreads::Thread
{
    public:
        JobThread(int id, JobType jt, std::vector<std::list<std::pair<sph_task*, CopyJobInfo*> > > * readlist, std::vector<std::vector<std::list<std::pair<sph_task*, CopyJobInfo*> > > > * copylist, std::map<int, std::map<int,DiskCacheEntry*> > * cacheMap, std::map<sph_cache*,int> * cacheIndexMap);
        virtual ~JobThread();

        void run();
        void quit();

    protected:
        void read();
        void copy();

        void readData(sph_task * task, CopyJobInfo* cji, TIFF * T, uint32 w, uint32 h, uint16 c, uint16 b);

        OpenThreads::Mutex _quitLock;
        bool _quit;

        int _readIndex;
        int _copyCacheNum;
        int _copyFileNum;

        JobType _jt;
        std::vector<std::list<std::pair<sph_task*, CopyJobInfo*> > > * _readList;
        std::vector<std::vector<std::list<std::pair<sph_task*, CopyJobInfo*> > > > * _copyList;
        std::map<int, std::map<int,DiskCacheEntry*> > * _cacheMap;
        std::map<sph_cache*,int> * _cacheIndexMap;

        int _id;
};

class DiskCache
{
    public:
        DiskCache(int pages);
        virtual ~DiskCache();

        bool isRunning()
        {
            return _running;
        }

        void start();
        void stop();

        int add_file(const std::string& name);
        void add_task(sph_task * task);
        void add_prefetch_task(sph_task * task);

        void setLeftFiles(int prev, int curr, int next);
        void setRightFiles(int prev, int curr, int next);
        void kill_tasks(int file);

        void getReadStats(unsigned int & pages, double & totalTime);
        void addPageTime(double time);

    protected:
        void cleanup();

        int _numReadThreads;
        int _numCopyThreads;

        int _pages;
        int _numPages;

        int _prefetchPages;
        int _prefetchNumPages;

        int _prevFileL;
        int _currentFileL;
        int _nextFileL;
        int _prevFileR;
        int _currentFileR;
        int _nextFileR;

        bool eject();

        bool _running;

        OpenThreads::Mutex _fileAddLock;
        std::map<std::string, int> _fileIDMap;
        int _nextID;

        std::vector<JobThread*> _readThreads;
        std::vector<JobThread*> _copyThreads;

        std::vector<std::list<std::pair<sph_task*, CopyJobInfo*> > > _readList;
        std::vector<std::vector<std::list<std::pair<sph_task*, CopyJobInfo*> > > > _copyList;
        std::vector<std::list<std::pair<sph_task*, CopyJobInfo*> > > _prefetchList;

        std::list<std::pair<int,int> > _cleanupList;

        std::map<int, std::map<int,DiskCacheEntry*> > _cacheMap;
        std::map<int, std::map<int,DiskCacheEntry*> > _prefetchMap;

        std::map<sph_cache*,int> _cacheIndexMap;
        
        // disk copy stats
        OpenThreads::Mutex _readStatsLock;
        unsigned int _pagesRead;
        double _totalReadTime;
};

#endif
