#ifndef _VEC4TYPE_
#define _VEC4TYPE_

#include <iostream>
#include <string>
#include <set>
#include <osg/Vec4d>
#include "Type.h"

// create type of variable that can be added
class Vec4Type : public Type
{
    public:
        Vec4Type(std::string value); 
        Vec4Type(); 
        
        osg::Vec4d getValue() { return _value; };
        virtual void setValue(std::string value);
        virtual Vec4Type* asVec4Type() { return this; };
        
    protected:
        virtual ~Vec4Type();
        osg::Vec4d _value;
};
#endif
