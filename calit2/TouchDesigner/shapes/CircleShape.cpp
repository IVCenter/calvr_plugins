#include "CircleShape.h"

#include <fcntl.h>
#include <iostream>
#include <cstring>

#include <osg/Matrix>
#include <osg/CullFace>
#include <osgDB/ReadFile>

using namespace osg;
using namespace std;

CircleShape::CircleShape()
{
	setUseVertexBufferObjects(true);
	name = "def";
	center = Vec3d(random()*600,0,random()*600);
	radius = 100;
	color = Vec4d(0.0,0.5,1.0,0.8);
	gradient = Vec4d(0.0,0.0,0.0,0.0);
	degree = 10;
	type = 5;

	generate();
}

CircleShape::~CircleShape()
{

}

CircleShape::CircleShape(string _name)
{
	setUseVertexBufferObjects(true);
	name = _name;
	center = Vec3d(0,0,0);
	radius = 10;
	color = Vec4d(0.0,0.5,1.0,0.3);	
	gradient = Vec4d(0.0,0.0,0.0,0.0);
	degree = 10;
	type = 5;

	generate();
}



// name, center, radius, color, gradient, tesselation
CircleShape::CircleShape(string _name, Vec3d& _center, double _radius, Vec4d& _color, Vec4d& _gradient, int _deg)
{
	setUseVertexBufferObjects(true);
	name = _name;
	center = _center;
	radius = _radius;
	color = _color;
	gradient = _gradient;
	degree = _deg;
	type = 5;

	generate();
}

// name, center, raidus, color, gradient
CircleShape::CircleShape(string _name, Vec3d& _center, double _radius, Vec4d& _color, Vec4d& _gradient)
{
 	setUseVertexBufferObjects(true);
	name = _name;
	center = _center;
	radius = _radius;
	color = _color;
	gradient = _gradient;	
	degree = 10;
	type = 5;

	generate();
}

// name, center, radius, color, tesselation
CircleShape::CircleShape(string _name, Vec3d& _center, double _radius, Vec4d& _color, int _deg)
{
	setUseVertexBufferObjects(true);
	name = _name;
	center = _center;
	radius = _radius;
	color = _color;	
	gradient = color;	
	degree = _deg;
	type = 5;

	generate();
}

// name, center, radius, color
CircleShape::CircleShape(string _name, Vec3d& _center, double _radius, Vec4d& _color)
{
	setUseVertexBufferObjects(true);
	name = _name;
	center = _center;
	radius = _radius;
	color = _color;		
	gradient = color;
	degree = 10;
	type = 5;

	generate();
}

// name, center, radius
CircleShape::CircleShape(string _name, Vec3d& _center, double _radius)
{
	setUseVertexBufferObjects(true);
	name = _name;
	center = _center;
	radius = _radius;
	color = Vec4d(0.0,0.5,1.0,0.3);	
	gradient = color;
	degree = 10;
	type = 5;

	generate();
}

// name, center
CircleShape::CircleShape(string _name, Vec3d& _center)
{
	setUseVertexBufferObjects(true);
	name = _name;
	center = _center;	
	radius = 10;
	color = Vec4d(0.0,0.5,1.0,0.3);	
	gradient = color;
	degree = 10;
	type = 5;

	generate();
}

// center
CircleShape::CircleShape(Vec3d& _center)
{
	setUseVertexBufferObjects(true);
	center = _center;
	name = "";	
	radius = 10;
	color = Vec4d(0.0,0.5,1.0,0.3);	
	gradient = color;
	degree = 10;
	type = 5;

	generate();
}


void CircleShape::setCenter(Vec3d& _center)
{
	center = _center;
}

Vec3d CircleShape::getCenter()
{
	return center;
}

void CircleShape::setRadius(double _radius)
{
	radius = _radius;
}

double CircleShape::getRadius()
{
	return radius;
}


int CircleShape::getType()
{
	return type;
}

int CircleShape::getId()
{
	return id;
}

void CircleShape::setName(string _name)
{
	name = _name;
}

string CircleShape::getName()
{
	return name;
}

void CircleShape::setColor(Vec4d& _color)
{
	color = _color;
}


Vec4d CircleShape::getColor()
{
	return color;
}

void CircleShape::setGradient(Vec4d& _gradient)
{
	gradient = _gradient;
}

Vec4d CircleShape::getGradient()
{
	return gradient;
}

void CircleShape::generate()
{
	/********************************************************************
	*																    *
	*					Vertices									    *
	*																    *
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
	*								    *
	*				colors				    *
	*								    *
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
	*								    *
	*			stateset and material			    *
	*	  							    *
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
	*								    *
	*			blending				    *
	*								    *
	********************************************************************
	*/


	// blending

	BlendFunc* bf = new BlendFunc(BlendFunc::SRC_ALPHA, BlendFunc::ONE_MINUS_SRC_ALPHA );
	state->setAttributeAndModes(bf);

	state->setRenderingHint(StateSet::TRANSPARENT_BIN);
	state->setMode(GL_LIGHTING, StateAttribute::ON);

	setStateSet(state);
}

void CircleShape::updateLocation()
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
		//printf("New center is %f %f %f\n", (*vertices)[0].x(), (*vertices)[0].y(), (*vertices)[0].z());		
		index++;
		(*vertices)[index] = Vec3d(x,0,z)+center;	
		index++;
		numVertices+=2;		

	}
	//printf("New center is %f %f %f\n", (*vertices)[0].x(), (*vertices)[0].y(), (*vertices)[0].z());

	(*vertices)[index] = center;
	//setVertexArray(vertices);
}

void CircleShape::updateColor()
{
	for (int g = 0; g < (720/degree)+1; g+=2)
	{
		(*colors)[g].set(color.x(), color.y(), color.z(), color.w());
		(*colors)[g+1].set(gradient.x(), gradient.y(), gradient.z(), gradient.w());	
	}
	setColorArray(colors);	
}

void CircleShape::scale(int _scale)
{
	radius = _scale * radius;
}

void CircleShape::setAll(Vec3d& c1, double rad, Vec4d& rgba1, Vec4d& rgba2)
{
	center = c1;
	radius = rad;
	color = rgba1;
	gradient = rgba2;
}

void CircleShape::updateAll()
{
	updateLocation();
	updateColor();
}


void CircleShape::setId(int _id)
{
	id = _id;
}
