#ifndef _STRINGTYPE_
#define _STRINGTYPE_

#include <iostream>
#include <string>
#include <set>
#include "Type.h"

// create type of variable that can be added
class StringType : public Type
{
    public:
        StringType(std::string value); 
        StringType();
         
        virtual void setValue(std::string value);
        std::string getValue() { return _value; };
        virtual StringType* asStringType() { return this; };

    protected:
        virtual ~StringType();
        std::string _value;
};
#endif
