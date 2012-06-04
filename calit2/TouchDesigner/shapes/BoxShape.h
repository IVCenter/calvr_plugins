#ifndef _BoxShape_
#define _BoxShape_

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

class BoxShape : public BasicShape
{
friend class ShapeHelper;
public:        

	BoxShape();
	virtual ~BoxShape();

	// name
	BoxShape(string);


	// name, p1, p2, p3, p4, p5, p6, p7, p8, color1, color2
	BoxShape(string, Vec3d&, Vec3d&, Vec3d&, Vec3d&, Vec3d&, Vec3d&, Vec3d&, Vec3d&, Vec4d&, Vec4d&);

	// name, p1, p2, p3, p4, p5, p6, p7, p8, color1 
	BoxShape(string, Vec3d&, Vec3d&, Vec3d&, Vec3d&, Vec3d&, Vec3d&, Vec3d&, Vec3d&, Vec4d&);

	// name, p1, p2, p3, p4, p5, p6, p7, p8
	BoxShape(string, Vec3d&, Vec3d&, Vec3d&, Vec3d&, Vec3d&, Vec3d&, Vec3d&, Vec3d&);

	// name, center, width, height, depth, color
	// this generates a solid color Boxangle center point _center
	BoxShape(string, Vec3d&, int, int, int, Vec4d&);


	// name, center, width, height, depth, color, gradient	
	BoxShape(string, Vec3d&, int, int, int, Vec4d&, Vec4d&);

	// name, center, width, height, depth
	BoxShape(string, Vec3d&, int, int, int);




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

	void setPoints(Vec3d&,Vec3d&,Vec3d&,Vec3d&,Vec3d&,Vec3d&,Vec3d&,Vec3d&);


protected:
	// identify the type of shape, 8: individually assigned points, 9: centered
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
	
	//depth, default = 10
	int depth;

	// center point of box, default = (0,0,0)
	Vec3d center;

	// point1 of Boxangle
	Vec3d p1;

	// point2 of Boxangle
	Vec3d p2;

	// point3 of Boxangle
	Vec3d p3;

	// point4 of Boxangle
	Vec3d p4;
	
		// point1 of Boxangle
	Vec3d p5;

	// point2 of Boxangle
	Vec3d p6;

	// point3 of Boxangle
	Vec3d p7;

	// point4 of Boxangle
	Vec3d p8;

	// vertices of the box
	Vec3Array* vertices;

	// color vertices of box
	Vec4Array* colors;

	void generate();
	
	bool genVer;
	
	void setId(int);

};

#endif
