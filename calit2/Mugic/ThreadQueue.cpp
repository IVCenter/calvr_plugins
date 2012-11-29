#include "ThreadQueue.h"

#include <osg/Node>

template <typename T>
ThreadQueue<T>::ThreadQueue() : _waiting (NULL)
{}

// override and do nothing
template <typename T>
ThreadQueue<T>::ThreadQueue(const ThreadQueue<T> &tq) 
{}

template <typename T>
ThreadQueue<T>::~ThreadQueue() 
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_lock);
    while( !_queue.empty())
    {
        T object = _queue.front();
        _queue.pop();   
    }    
}

template <typename T>
bool ThreadQueue<T>::get(T& object)
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_lock);
    if( _queue.empty() )
        return false;

    object = _queue.front();
    _queue.pop();
    return true;   
}

template <typename T>
int ThreadQueue<T>::size()
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_lock);
    return _queue.size();
}

template <typename T>
void ThreadQueue<T>::add(T object)
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_lock);
    _queue.push(object);

	if( _waiting )
        _waiting->signal();
}

template <typename T>
void ThreadQueue<T>::setCondition(OpenThreads::Condition* condition)
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_lock);
    _waiting = condition;    
}

template <typename T>
OpenThreads::Condition* ThreadQueue<T>::getCondition()
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_lock);
    return _waiting;
}

template class ThreadQueue<std::string>;
template class ThreadQueue<osg::Node* >;
