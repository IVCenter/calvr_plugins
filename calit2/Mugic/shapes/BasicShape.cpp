#include "BasicShape.h"

#include <sstream>
#include <string>
#include <iostream>

//#include "../Variables.h"

using namespace std;

unsigned int BasicShape::counter;

BasicShape::BasicShape(): _dirty(false)
{
    getOrCreateVertexBufferObject()->setUsage(GL_STREAM_DRAW);
    setUseDisplayList(false);
    setUseVertexBufferObjects(true);
    setUpdateCallback(new ShapeUpdateCallback());
}

BasicShape::~BasicShape()
{
}


bool BasicShape::isDirty()
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
    return _dirty;
}


void BasicShape::addParameter(std::string command, std::string param)
{
    std::string substring, value, searchParam;
    searchParam.append(param).append("=");
    size_t found = command.find(searchParam);
    if(found != std::string::npos)
    {
        // extract value
        substring = command.substr(found + searchParam.size());
        std::stringstream ss(substring);
        ss >> value;
        addLocalParam(param, value);
    }
}

// if add empty string will remove reference
void BasicShape::addLocalParam(std::string varName, std::string param)
{
    std::map<std::string, std::string>::iterator it =  _localParams.find(varName);
    if( it != _localParams.end() )
    {
        if( param.empty() )
            _localParams.erase(it);
        else
            it->second = param;
    }
    else
    {
        _localParams[varName] = param;
    }
}

void BasicShape::setParameter(std::string varName, std::string& value)
{
   std::map<std::string, std::string>::iterator it = _localParams.find(varName);
   if( it != _localParams.end() ) // check local params
   {
        value = it->second;
   }
}


// TODO need to redo so its recursive
void BasicShape::setParameter(std::string varName, float& value)
{
   std::map<std::string, std::string>::iterator it = _localParams.find(varName);
   if( it != _localParams.end() ) // check local params
   {
        float tempValue;

        std::stringstream lss (it->second);
        if( lss >> tempValue )
        {
            value = tempValue;
        }
/*
        else // check variables (as ss was not a number)
        {
            std::string tvalue;
            if ( Variables::getInstance()->get(it->second, tvalue) )
            {
                std::stringstream gss (tvalue);
                if( gss >> tempValue )
                    value = tempValue;
                
                std::cerr << "Global variable " << value << std::endl;
            }
        }
*/
   }
}

void BasicShape::setName(std::string name)
{
    if(name.empty())
    {
        std::stringstream ss;
        ss << "element";
        ss << counter;
        counter++;
        _name = ss.str();
    }
    else
    {
        _name = name;
    }
}

std::string BasicShape::getName()
{
   return _name;
}
