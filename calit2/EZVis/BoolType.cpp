#include "BoolType.h"
#include <sstream>

BoolType::BoolType() : _value(0)
{
}

BoolType::BoolType(std::string value) : _value(0)
{
    setValue(value);
}

void BoolType::setValue(std::string value)
{
    std::stringstream ss(value);
    ss >> _value;
}

BoolType::~BoolType()
{
}
