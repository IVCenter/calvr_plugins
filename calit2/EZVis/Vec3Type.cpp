#include "Vec3Type.h"
#include <sstream>

Vec3Type::Vec3Type()
{
    _value.set(0.0, 0.0, 0.0);
}

Vec3Type::Vec3Type(std::string value) 
{
    setValue(value);
}

void Vec3Type::setValue(std::string value)
{
    std::string delimiter(",");

    std::istringstream ss(value);
    std::string token;
    int index = 0;

    while(std::getline(ss, token, ',')) 
    {
            std::stringstream val(token);
            val >> _value[index];
            index++;
            
            // stop going outside bounds
            if( index > 2 )
                break;        
    }
}

Vec3Type::~Vec3Type()
{
}
