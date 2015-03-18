#ifndef _BOOLTYPE_
#define _BOOLTYPE_

#include <iostream>
#include <string>
#include <set>
#include "Type.h"

// create type of variable that can be added
class BoolType : public Type
{
    public:
        BoolType(std::string value); 
        BoolType(); 
        
        bool getValue() { return _value; };
        virtual void setValue(std::string value);

        virtual StringType* asStringType() { return NULL; };
        virtual FloatType* asFloatType() { return NULL; };
        virtual Vec3Type* asVec3Type() { return NULL; };
        virtual Vec4Type* asVec4Type() { return NULL; };
        virtual BoolType* asBoolType() { return this; };
        
    protected:
        virtual ~BoolType();
        bool _value;
};

#endif
