#ifndef _PyramidShape_
#define _PyramidShape_

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

class PyramidShape : public BasicShape
{
friend class ShapeHelper;
public:        

	PyramidShape(); //defaults to 3 sides
	virtual ~PyramidShape();

	// name
	PyramidShape(string); //defaults to 3 sides

  /*********3 sided pyramid requires 4 points*************/
	// name, p1, p2, p3, p4, color1, color2, color3
	PyramidShape(string, Vec3d&, Vec3d&, Vec3d&, Vec3d&, Vec4d&, Vec4d&, Vec4d&);

	// name, p1, p2, p3, p4, color1, color2
	PyramidShape(string, Vec3d&, Vec3d&, Vec3d&, Vec3d&, Vec4d&, Vec4d&);

	// name, p1, p2, p3, p4, color
	PyramidShape(string, Vec3d&, Vec3d&, Vec3d&, Vec3d&, Vec4d&);
	
	// name, p1, p2, p3, p4
	PyramidShape(string, Vec3d&, Vec3d&, Vec3d&, Vec3d&);



  /*********4 sided pyramid requires 5 points*************/
	// name, p1, p2, p3, p4, p5, color1, color2, color3
	PyramidShape(string, Vec3d&, Vec3d&, Vec3d&, Vec3d&, Vec3d&, Vec4d&, Vec4d&, Vec4d&);

	// name, p1, p2, p3, p4, p5, color1, color2
	PyramidShape(string, Vec3d&, Vec3d&, Vec3d&, Vec3d&, Vec3d&, Vec4d&, Vec4d&);

	// name, p1, p2, p3, p4, p5, color
	PyramidShape(string, Vec3d&, Vec3d&, Vec3d&, Vec3d&, Vec3d&, Vec4d&);
	
	// name, p1, p2, p3, p4, p5
	PyramidShape(string, Vec3d&, Vec3d&, Vec3d&, Vec3d&, Vec3d&);



	// name, center, sides, height, length, width, color
	// this generates a solid color equalateral triangle with center point _center
	PyramidShape(string, Vec3d&, int, int, int, int, Vec4d&);

	// name, center, sides, height, length, width, color, gradient
	// this generates a vertical gradient equalateral triangle with center point _center
	PyramidShape(string, Vec3d&, int, int, int, int, Vec4d&, Vec4d&);

	// name, center, sides, height, length, width
	PyramidShape(string, Vec3d&, int, int, int, int);




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
	
	void setPoints(Vec3d&,Vec3d&,Vec3d&,Vec3d&);
	void setPoints(Vec3d&,Vec3d&,Vec3d&,Vec3d&,Vec3d&);
	



protected:
	// identify the type of shape, 6,7 pyramids
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

	// height of pyramids, default = 10
	int height;

	// center point of pyramids, default = (0,0,0)
	Vec3d center;

	// point1 of pyramids
	Vec3d p1;

	// point2 of pyramids
	Vec3d p2;

	// point3 of pyramids
	Vec3d p3;
	
	// point4 of pyramids
	Vec3d p4;
	
	// point5 of pyramids, if any
	Vec3d p5;

  //how many sides the pyramid has
  int sides;

  //length and width (width only for 3 sided)
  int width;
  int length;
  

	// vertices of the pyramids
	Vec3Array* vertices;

	// color vertices of pyramids
	Vec4Array* colors;

	void generate();

	bool genVer;
	
	void setId(int);

};

#endif
