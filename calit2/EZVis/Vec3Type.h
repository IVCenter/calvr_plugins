#ifndef _VEC3TYPE_
#define _VEC3TYPE_

#include <iostream>
#include <string>
#include <set>
#include <osg/Vec3d>
#include "Type.h"

// create type of variable that can be added
class Vec3Type : public Type
{
    public:
        Vec3Type(std::string value); 
        Vec3Type();
        
        osg::Vec3d getValue() { return _value; };
        virtual void setValue(std::string value);
        virtual Vec3Type* asVec3Type() { return this; };
        
    protected:
        virtual ~Vec3Type();
        osg::Vec3d _value;
};

#endif
