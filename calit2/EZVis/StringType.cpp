#include "StringType.h"

StringType::StringType() : _value("")
{
}

StringType::StringType(std::string value) 
{
    setValue(value);
}

StringType::~StringType()
{
}

void StringType::setValue(std::string value)
{
    size_t found = value.find("\"", 1);
    if(found != std::string::npos)
    {
        _value = value.substr(1, found - 1);
    }
} 
