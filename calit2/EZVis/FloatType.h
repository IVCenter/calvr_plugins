#ifndef _FLOATTYPE_
#define _FLOATTYPE_

#include <iostream>
#include <string>
#include <set>
#include "Type.h"

// create type of variable that can be added
class FloatType : public Type
{
    public:
        FloatType(std::string value); 
        FloatType(); 
        
        double getValue() { return _value; };
        virtual void setValue(std::string value);
        virtual FloatType* asFloatType() { return this; };
        
    protected:
        virtual ~FloatType();
        double _value;
};

#endif
