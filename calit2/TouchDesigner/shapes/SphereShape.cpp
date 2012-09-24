#include "SphereShape.h"

#include <fcntl.h>
#include <iostream>
#include <cstring>

#include <osg/Matrix>
#include <osg/CullFace>
#include <osgDB/ReadFile>

using namespace osg;
using namespace std;

SphereShape::SphereShape()
{
	name = "def";
	center = Vec3d(random()*600,0,random()*600);
	radius = 100;
	color = Vec4d(0.0,0.5,1.0,0.8);
	gradient = Vec4d(0.0,0.0,0.0,0.0);
	degree = 10;
	type = 10;
	
	generate();
}

SphereShape::~SphereShape()
{

}

SphereShape::SphereShape(string _name)
{
	name = _name;
	center = Vec3d(0,0,0);
	radius = 10;
	color = Vec4d(0.0,0.5,1.0,0.3);	
	gradient = Vec4d(0.0,0.0,0.0,0.0);
	degree = 10;
	type = 10;
	
	generate();
}



// name, center, radius, color, gradient, tesselation
SphereShape::SphereShape(string _name, Vec3d& _center, double _radius, Vec4d& _color, Vec4d& _gradient, int _deg)
{
	name = _name;
	center = _center;
	radius = _radius;
	color = _color;
	gradient = _gradient;
	degree = _deg;
	type = 10;
	
	generate();
}

// name, center, raidus, color, gradient
SphereShape::SphereShape(string _name, Vec3d& _center, double _radius, Vec4d& _color, Vec4d& _gradient)
{
	name = _name;
	center = _center;
	radius = _radius;
	color = _color;
	gradient = _gradient;	
	degree = 10;
	type = 10;
	
	generate();
}

// name, center, radius, color, tesselation
SphereShape::SphereShape(string _name, Vec3d& _center, double _radius, Vec4d& _color, int _deg)
{
	name = _name;
	center = _center;
	radius = _radius;
	color = _color;	
	gradient = color;	
	degree = _deg;
	type = 10;
	
	generate();
}

// name, center, radius, color
SphereShape::SphereShape(string _name, Vec3d& _center, double _radius, Vec4d& _color)
{
	name = _name;
	center = _center;
	radius = _radius;
	color = _color;		
	gradient = color;
	degree = 10;
	type = 10;
	
	generate();
}

// name, center, radius
SphereShape::SphereShape(string _name, Vec3d& _center, double _radius)
{
	name = _name;
	center = _center;
	radius = _radius;
	color = Vec4d(0.0,0.5,1.0,0.3);	
	gradient = color;
	degree = 10;
	type = 10;
	
	generate();
}

// name, center
SphereShape::SphereShape(string _name, Vec3d& _center)
{
	name = _name;
	center = _center;	
	radius = 10;
	color = Vec4d(0.0,0.5,1.0,0.3);	
	gradient = color;
	degree = 10;
	type = 10;
	
	generate();
}

// center
SphereShape::SphereShape(Vec3d& _center)
{
	center = _center;
	name = "";	
	radius = 10;
	color = Vec4d(0.0,0.5,1.0,0.3);	
	gradient = color;
	degree = 10;
	type = 10;
	
	generate();
}


void SphereShape::setCenter(Vec3d& _center)
{
	center = _center;
}

Vec3d SphereShape::getCenter()
{
	return center;
}

void SphereShape::setRadius(double _radius)
{
	radius = _radius;
}

double SphereShape::getRadius()
{
	return radius;
}


int SphereShape::getType()
{
	return type;
}

int SphereShape::getId()
{
	return id;
}

void SphereShape::setName(string _name)
{
	name = _name;
}

string SphereShape::getName()
{
	return name;
}

void SphereShape::setColor(Vec4d& _color)
{
	color = _color;
}


Vec4d SphereShape::getColor()
{
	return color;
}

void SphereShape::setGradient(Vec4d& _gradient)
{
	gradient = _gradient;
}

Vec4d SphereShape::getGradient()
{
	return gradient;
}

void SphereShape::generate()
{
	/********************************************************************
	*																        *
	*					Vertices									    *
	*																        *
	********************************************************************
	*/
	int numVertices = 0;

	vertices = new Vec3Array;
	for (int i=0; i <= 360; i = i + degree )
	{	
		double angle = (i*2*PI)/360;		
		double x,z;
		x = cos(angle)*radius;
		z = sin(angle)*radius;


		vertices->push_back(center);
		vertices->push_back(Vec3d(x,0,z)+center);	
		numVertices+=2;		
	}

	vertices->push_back(center);
	setVertexArray(vertices);

	/********************************************************************
	*								          *
	*				colors				    *
	*								          *
	********************************************************************
	*/

	colors = new Vec4Array(numVertices);
	for (int g = 0; g < numVertices; g+=2)
	{
		(*colors)[g].set(color.x(), color.y(), color.z(), color.w());
		(*colors)[g+1].set(gradient.x(), gradient.y(), gradient.z(), gradient.w());	
	}

	setColorArray(colors);	
	setColorBinding(Geometry::BIND_PER_VERTEX);



	addPrimitiveSet(new DrawArrays(PrimitiveSet::TRIANGLE_STRIP, 0,2+(360*2/degree)));


	/********************************************************************
	*								                    *
	*			stateset and material			    *
	*	  							                  *
	********************************************************************
	*/

	// stateset and material
	StateSet* state = getOrCreateStateSet();
	state->setMode(GL_BLEND,StateAttribute::ON|StateAttribute::OVERRIDE);
	Material* mat = new Material(); 
	mat->setAlpha(Material::FRONT_AND_BACK, 0.1);
	mat->setColorMode(Material::AMBIENT_AND_DIFFUSE);
	state->setAttributeAndModes(mat,StateAttribute::ON | StateAttribute::OVERRIDE);


	/********************************************************************
	*								          *
	*			blending				    *
	*								          *
	********************************************************************
	*/


	// blending

	BlendFunc* bf = new BlendFunc(BlendFunc::SRC_ALPHA, BlendFunc::ONE_MINUS_SRC_ALPHA );
	state->setAttributeAndModes(bf);

	state->setRenderingHint(StateSet::TRANSPARENT_BIN);
	state->setMode(GL_LIGHTING, StateAttribute::ON);

	setStateSet(state);
}

void SphereShape::updateLocation()
{
	int numVertices = 0;
	int index = 0;


	for (int i=0; i <= 360; i = i + degree )
	{	
		double angle = (i*2*PI)/360;		
		double x,z;
		x = cos(angle)*radius;
		z = sin(angle)*radius;


		(*vertices)[index] = center;
		index++;
		(*vertices)[index] = Vec3d(x,0,z)+center;	
		index++;
		numVertices+=2;		

	}

	(*vertices)[index] = center;
	setVertexArray(vertices);

}

void SphereShape::updateColor()
{
	for (int g = 0; g < (720/degree)+1; g+=2)
	{
		(*colors)[g].set(color.x(), color.y(), color.z(), color.w());
		(*colors)[g+1].set(gradient.x(), gradient.y(), gradient.z(), gradient.w());	
	}
	setColorArray(colors);	
}

void SphereShape::scale(int _scale)
{
	radius = _scale * radius;
}

void SphereShape::setAll(Vec3d& c1, double rad, Vec4d& rgba1, Vec4d& rgba2)
{
	center = c1;
	radius = rad;
	color = rgba1;
	gradient = rgba2;
}

void SphereShape::updateAll()
{
	updateLocation();
	updateColor();
}


void SphereShape::setId(int _id)
{
	id = _id;
}
