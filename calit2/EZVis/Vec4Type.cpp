#include "Vec4Type.h"
#include <sstream>

Vec4Type::Vec4Type() 
{
    _value.set(1.0, 1.0, 1.0, 1.0);
}

Vec4Type::Vec4Type(std::string value) 
{
    setValue(value);
}

void Vec4Type::setValue(std::string value)
{
    std::istringstream ss(value);
    std::string token;
    int index = 0;

    while(std::getline(ss, token, ',')) 
    {
            std::stringstream val(token);
            val >> _value[index];
            index++;        
   
            // stop going outside bounds 
            if( index > 3 )
                break;
    }
}

Vec4Type::~Vec4Type()
{
}
