#ifndef _SphereShape_
#define _SphereShape_

#include <osg/MatrixTransform>
#include <osg/Vec3d>
#include <osg/Vec4>
#include <osg/StateSet>
#include <osg/Material>
#include <osg/BlendFunc>


#include <string>
#include <vector>
#include <iostream>

#include "BasicShape.h"

using namespace std;
using namespace osg;

class SphereShape : public BasicShape
{
friend class ShapeHelper;
public:        

	SphereShape();

	virtual ~SphereShape();

	// name
	SphereShape(string);


	// name, center, radius, color, gradient, tesselation, 
	SphereShape(string, Vec3d&, double, Vec4d&, Vec4d&, int);

	// name, center, raidus, color, gradient
	SphereShape(string, Vec3d&, double, Vec4d&, Vec4d&);

	// name, center, radius, color, tesselation
	SphereShape(string, Vec3d&, double, Vec4d&, int);

	// name, center, radius, color
	SphereShape(string, Vec3d&, double, Vec4d&);

	// name, center, radius
	SphereShape(string, Vec3d&, double);

	// name, center
	SphereShape(string, Vec3d&);

	// center
	SphereShape(Vec3d&);



	int getType();

	int getId();

	void setName(string);

	string getName();

	void setColor(Vec4d&);

	Vec4d getColor();

	void setGradient(Vec4d&);

	Vec4d getGradient();

	void setCenter(Vec3d&);

	Vec3d getCenter();

		
	// sets the Radius directly
	void setRadius(double);

	double getRadius();

	void updateLocation();
	
	void updateColor();
	

	// multiplies radius by param	
	void scale(int);

	void setAll(Vec3d&, double, Vec4d&, Vec4d&);

	void updateAll();
	
	



protected:
	// identify the type of shape, 1 triangle, 2 cirlce, 3 rectangle
	int type;

	// numeric id of this shape
	int id;

	// string identifier of this shape
	string name;

	// (solid) primary color of shape, default skyBlue(0.0,0.5,1.0,0.3);
	Vec4d color;

	// (gradient) secondary color of shape, if specified
	Vec4d gradient;

	// radius of sphere, default = 10
	double radius;

	// center point of sphere, default = (0,0,0)
	Vec3d center;

	// degree tesselation of sphere, higher number = less triangles used to draw sphere, default = 10
	int degree;

	// points of the sphere
	Vec3Array* vertices;

	// color vertices of sphere
	Vec4Array* colors;

	// calculates vertices and stuff for sphere
	void generate();

	int numVertices;
	
	void setId(int);
	
};

#endif
