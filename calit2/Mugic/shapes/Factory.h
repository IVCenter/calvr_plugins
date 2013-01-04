#ifndef _FACTORY_
#define _FACTORY_

#include "ShapeFactory.h"
#include "BasicShape.h"

template <class T> 
class Factory : public ShapeFactory
{
    public:
        BasicShape* createInstance(std::string command, std::string name)
        {
            return new T(command, name);
        }
};
#endif
