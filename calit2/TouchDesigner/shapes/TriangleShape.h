#ifndef _TriangleShape_
#define _TriangleShape_

#include <osg/MatrixTransform>
#include <osg/Vec3d>
#include <osg/Vec4>
#include <osg/StateSet>
#include <osg/Material>
#include <osg/BlendFunc>


#include <string>
#include <vector>
#include "BasicShape.h"

using namespace std;
using namespace osg;

class TriangleShape : public BasicShape
{
friend class ShapeHelper;
public:        

	TriangleShape();
	virtual ~TriangleShape();

	// name
	TriangleShape(string);


	// name, p1, p2, p3, color1, color2, color3
	TriangleShape(string, Vec3d&, Vec3d&, Vec3d&, Vec4d&, Vec4d&, Vec4d&);

	// name, p1, p2, p3, color1, color2
	TriangleShape(string, Vec3d&, Vec3d&, Vec3d&, Vec4d&, Vec4d&);

	// name, p1, p2, p3, color
	TriangleShape(string, Vec3d&, Vec3d&, Vec3d&, Vec4d&);
	
	// name, p1, p2, p3
	TriangleShape(string, Vec3d&, Vec3d&, Vec3d&);

	// name, center, length, color
	// this generates a solid color equalateral triangle with center point _center
	TriangleShape(string, Vec3d&, int, Vec4d&);

	// name, center, length, color, gradient
	// this generates a vertical gradient equalateral triangle with center point _center
	TriangleShape(string, Vec3d&, int, Vec4d&, Vec4d&);

	// name, center, length
	TriangleShape(string, Vec3d&, int);




	int getType();

	int getId();

	void setName(string);

	string getName();

	void setColor1(Vec4d&);

	Vec4d getColor1();

	void setColor2(Vec4d&);

	Vec4d getColor2();

	void setColor3(Vec4d&);

	Vec4d getColor3();

	void setCenter(Vec3d&);
	
	void setLength(int);

	int getLength();

	void updateAll();
	
	void updateLocation();
	
	void updateColor();
	
	void setPoints(Vec3d&,Vec3d&,Vec3d&);



protected:
	// identify the type of shape, 1 triangle, 2 cirlce, 3 rectangle
	int type;

	// numeric id of this shape
	int id;

	// string identifier of this shape
	string name;

	// (solid) primary color of shape, default lavender(0.4,0.5,1.0,0.3);
	Vec4d color1;

	// (gradient) secondary color of shape, if specified
	Vec4d color2;

	// tertiary color of shape, if specified
	Vec4d color3;

	// radius of triangle, default = 10
	int length;

	// center point of triangle, default = (0,0,0)
	Vec3d center;

	// point1 of triangle
	Vec3d p1;

	// point2 of triangle
	Vec3d p2;

	// point3 of triangle
	Vec3d p3;

	// vertices of the triangle
	Vec3Array* vertices;

	// color vertices of cirlce
	Vec4Array* colors;

	void generate();

	bool genVer;
	
	void setId(int);

};

#endif
