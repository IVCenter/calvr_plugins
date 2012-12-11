#ifndef _SHAPEFACTORY_
#define _SHAPEFACTORY_

#include "BasicShape.h"

class ShapeFactory 
{
public:
    virtual BasicShape* createInstance(std::string, std::string) = 0;
};
#endif
