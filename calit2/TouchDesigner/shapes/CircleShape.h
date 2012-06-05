#ifndef _CircleShape_
#define _CircleShape_

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

class CircleShape: public BasicShape {
	friend class ShapeHelper;
public:

	CircleShape();

	virtual ~CircleShape();

	// name
	CircleShape( string);

	// name, center, radius, color, gradient, tesselation, 
	CircleShape(string, Vec3d&, double, Vec4d&, Vec4d&, int);

	// name, center, raidus, color, gradient
	CircleShape(string, Vec3d&, double, Vec4d&, Vec4d&);

	// name, center, radius, color, tesselation
	CircleShape(string, Vec3d&, double, Vec4d&, int);

	// name, center, radius, color
	CircleShape(string, Vec3d&, double, Vec4d&);

	// name, center, radius
	CircleShape(string, Vec3d&, double);

	// name, center
	CircleShape(string, Vec3d&);

	// center
	CircleShape(Vec3d&);

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

	// radius of circle, default = 100
	double radius;

	// center point of circle, default = (0,0,0)
	Vec3d center;

	// degree tesselation of circle, higher number = less triangles used to draw circle, default = 10
	int degree;

	// points of the circle
	Vec3Array* vertices;

	// color vertices of cirlce
	Vec4Array* colors;

	// calculates vertices and stuff for circle
	void generate();

	int numVertices;

	void setId(int);

};

#endif
