#ifndef _RectShape_
#define _RectShape_

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

class RectShape : public BasicShape
{
friend class ShapeHelper;
public:        

	RectShape();
	virtual ~RectShape();

	// name
	RectShape(string);


	// name, p1, p2, p3, p4, color1, color2
	RectShape(string, Vec3d&, Vec3d&, Vec3d&, Vec3d&, Vec4d&, Vec4d&);

	// name, p1, p2, p3, p4, color1 
	RectShape(string, Vec3d&, Vec3d&, Vec3d&, Vec3d&, Vec4d&);

	// name, p1, p2, p3, p4
	RectShape(string, Vec3d&, Vec3d&, Vec3d&, Vec3d&);

	// name, center, width, height, color
	// this generates a solid color rectangle center point _center
	RectShape(string, Vec3d&, int, int, Vec4d&);


	// name, center, width, height, color, gradient	
	RectShape(string, Vec3d&, int, int, Vec4d&, Vec4d&);

	// name, center, width, height
	RectShape(string, Vec3d&, int, int);




	int getType();

	int getId();

	void setName(string);

	string getName();

	void setColor1(Vec4d&);

	Vec4d getColor1();

	void setColor2(Vec4d&);

	Vec4d getColor2();	

	void setCenter(Vec3d&);
	
	void setHeight(int);

	int getHeight();

	void setWidth(int);

	int getWidth();

	void updateAll();

	void updateLocation();
	
	void updateColor();

	void setPoints(Vec3d&,Vec3d&,Vec3d&,Vec3d&);


protected:
	// identify the type of shape, 1 triangle, 2 cirlce, 3 rectangle
	int type;

	// numeric id of this shape
	int id;

	// string identifier of this shape
	string name;

	// (solid) primary color of shape, default (0.8,0.5,0.3,0.3);
	Vec4d color1;

	// (gradient) secondary color of shape, if specified
	Vec4d color2;	

	// width default = 10
	int width;

	// height, default = 10
	int height;

	// center point of triangle, default = (0,0,0)
	Vec3d center;

	// point1 of rectangle
	Vec3d p1;

	// point2 of rectangle
	Vec3d p2;

	// point3 of rectangle
	Vec3d p3;

	// point4 of rectangle
	Vec3d p4;

	// vertices of the triangle
	Vec3Array* vertices;

	// color vertices of cirlce
	Vec4Array* colors;

	void generate();
	
	bool genVer;
	
	void setId(int);

};

#endif
