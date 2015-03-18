#include "BasicShape.h"

#include <sstream>
#include <string>
#include <iostream>

using namespace std;

BasicShape::BasicShape(): _dirty(false)
{
}

BasicShape::~BasicShape()
{
}

bool BasicShape::isDirty()
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
    return _dirty;
}

void BasicShape::createParameter(std::string name, Type* type)
{
   std::map<std::string, Type* >::iterator it = _paramMapping.find(name);
   if( it == _paramMapping.end() )
        _paramMapping[name] = type;
}

void BasicShape::setParameter(std::string command, std::string param)
{
    //std::string substring, value, searchParam;
    std::string substring, searchParam;
    searchParam.append(param).append("=");
    size_t found = command.find(searchParam);
    if(found != std::string::npos)
    {
        substring = command.substr(found + searchParam.size());

        // set the value
        std::map<std::string, Type* >::iterator it = _paramMapping.find(param);
        if( it != _paramMapping.end() )
        {
            it->second->setValue(substring);
        }
    }
}

Type* BasicShape::getParameter(std::string name)
{
   std::map<std::string, Type* >::iterator it = _paramMapping.find(name);
   if( it != _paramMapping.end() )
   {
        return it->second;
   }
}

void BasicShape::setName(std::string name)
{
    if( !name.empty())
    {
        _name = name;
    }
}

std::string BasicShape::getName()
{
   return _name;
}
