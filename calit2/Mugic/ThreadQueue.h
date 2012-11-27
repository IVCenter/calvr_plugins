#ifndef _THREADQUEUE_H
#define _THREADQUEUE_H

#include <queue>
#include <OpenThreads/Mutex>
#include <OpenThreads/ScopedLock>
#include <OpenThreads/Condition>

using namespace std;

template <typename T>
class ThreadQueue
{
    private:
	    OpenThreads::Mutex _lock;
        std::queue<T> _queue;
		OpenThreads::Condition * _waiting;

    public:
	    ThreadQueue();
        ThreadQueue(const ThreadQueue& tq);
	    ~ThreadQueue();
        bool get(T&);
        int size();
        void add(T);
		void setCondition(OpenThreads::Condition*);
        OpenThreads::Condition* getCondition();
};
#endif
