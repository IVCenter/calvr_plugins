#ifndef _THREADMAP_H
#define _THREADMAP_H

#include <map>
#include <OpenThreads/Mutex>
#include <OpenThreads/ScopedLock>

using namespace std;

template <typename K, typename T>
class ThreadMap
{
    private:
	    OpenThreads::Mutex _lock;
        std::map<K, T> _map;

    public:
	    ThreadMap();
        ThreadMap(const ThreadMap& tq);
	    ~ThreadMap();
        bool get(K, T&);
        void add(K, T);
};
#endif
