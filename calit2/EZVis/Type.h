#ifndef _TYPE_
#define _TYPE_

#include <iostream>
#include <string>
#include <set>

class FloatType;
class StringType;
class Vec3Type;
class Vec4Type;
class BoolType;

// create type of variable that can be added
class Type
{
    public:
        virtual void setValue(std::string value) = 0;
        virtual StringType* asStringType() { return NULL; };
        virtual FloatType* asFloatType() { return NULL; };
	    virtual Vec3Type* asVec3Type() { return NULL; };
        virtual Vec4Type* asVec4Type() { return NULL; };
        virtual BoolType* asBoolType() { return NULL; };
};
#endif
