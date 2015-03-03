#include "FloatType.h"
#include <sstream>

FloatType::FloatType() : _value(0.0)
{
}

FloatType::FloatType(std::string value) : _value(0.0)
{
    setValue(value);
}

void FloatType::setValue(std::string value)
{
    std::stringstream ss(value);
    ss >> _value;
}

FloatType::~FloatType()
{
}
