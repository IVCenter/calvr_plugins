#include "BasicShape.h"

#include <fcntl.h>
#include <iostream>
#include <cstring>

#include <osg/Matrix>
#include <osg/CullFace>
#include <osgDB/ReadFile>

using namespace osg;
using namespace std;


BasicShape::BasicShape()
{

}



BasicShape::~BasicShape()
{

}

BasicShape::BasicShape(int _type, string _name)
{
	type = _type;
	name = _name;
}

int BasicShape::getType()
{
	return type;
}

int BasicShape::getId()
{
	return id;
}

void BasicShape::setName(string _name)
{
	name = _name;
}

string BasicShape::getName()
{
	return name;
}

void BasicShape::setColor(Vec4d& _color)
{
	color = _color;
}


Vec4d BasicShape::getColor()
{
	return color;
}

void BasicShape::setGradient(Vec4d& _gradient)
{
	gradient = _gradient;
}

Vec4d BasicShape::getGradient()
{
	return gradient;
}
