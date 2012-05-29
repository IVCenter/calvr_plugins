#ifndef _BasicShape_
#define _BasicShape_

#include <osg/MatrixTransform>
#include <osg/Vec3d>
#include <osg/Vec4>
#include <osg/StateSet>
#include <osg/Material>
#include <osg/BlendFunc>
#include <osg/Geometry>

#include <string>
#include <vector>

using namespace std;
using namespace osg;

class BasicShape : public Geometry
{
public:        
	BasicShape();
	virtual ~BasicShape();

	// type, name
	BasicShape(int, string);

	

	int getType();

	int getId();

	void setName(string);

	string getName();

	void setColor(Vec4d&);

	Vec4d getColor();

	void setGradient(Vec4d&);

	Vec4d getGradient();

protected:
	// identify the type of shape, 1&2 triangle, 3&4 rectangle, 5 circle
	int type;

	// numeric id of this shape
	int id;

	// string identifier of this shape
	string name;

	// (solid) primary color of shape
	Vec4d color;

	// (gradient) secondary color of shape, if specified
	Vec4d gradient;
	

};

#endif
