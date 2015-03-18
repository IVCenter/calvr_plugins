#include "ThreadMap.h"

#include <osg/Node>
#include <osgText/Font>

template <typename K, typename T>
ThreadMap<K, T>::ThreadMap() 
{}

// override and do nothing
template <typename K, typename T>
ThreadMap<K, T>::ThreadMap(const ThreadMap<K, T> &tq) 
{}

template <typename K, typename T>
ThreadMap<K, T>::~ThreadMap() 
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_lock);
    _map.erase( _map.begin(), _map.end());
}

template <typename K, typename T>
bool ThreadMap<K, T>::get(K key, T& object)
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_lock);
    if( _map.find(key) == _map.end() )
        return false;

    object = _map[key];
    return true;   
}

template <typename K, typename T>
void ThreadMap<K, T>::add(K key, T object)
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_lock);
    _map.insert(std::pair<K, T> (key, object));
}

template class ThreadMap<std::string, std::string>;
template class ThreadMap<std::string, osgText::Font* >;
