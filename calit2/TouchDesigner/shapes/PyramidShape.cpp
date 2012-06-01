#include "PyramidShape.h"

#include <fcntl.h>
#include <iostream>
#include <cstring>

#include <osg/Matrix>
#include <osg/CullFace>
#include <osgDB/ReadFile>

using namespace osg;
using namespace std;





PyramidShape::PyramidShape()
{
	name = "";
	center = Vec3d(0,0,0);
	length = 10;
	color1 = Vec4d(0.6,0.2,0.8,0.3);		
	type = 1;
	genVer= true;
	generate();
}



PyramidShape::~PyramidShape()
{

}

PyramidShape::PyramidShape(string _name)
{
	name = "";
	center = Vec3d(0,0,0);
	length = 10;
	color1 = Vec4d(0.6,0.2,0.8,0.3);		
	type = 1;
	genVer= true;
	generate();
}


// name, p1, p2, p3, color1, color2, color3
PyramidShape::PyramidShape(string _name, Vec3d& _p1, Vec3d& _p2, Vec3d& _p3, Vec4d& _c1, Vec4d& _c2, Vec4d& _c3)
{
	name = _name;
	p1 = _p1;
	p2 = _p2;
	p3 = _p3;
	color1 = _c1;
	color2 = _c2;
	color3 = _c3;	
	type = 2;
	genVer= false;
	generate();
}

// name, p1, p2, p3, color1, color2
PyramidShape::PyramidShape(string _name, Vec3d& _p1, Vec3d& _p2, Vec3d& _p3, Vec4d& _c1, Vec4d& _c2)
{
	name = _name;
	p1 = _p1;
	p2 = _p2;
	p3 = _p3;
	color1 = _c1;
	color2 = _c2;
	color3 = _c2;
	type = 2;
	genVer= false;
	generate();
}

// name, p1, p2, p3, color
PyramidShape::PyramidShape(string _name, Vec3d& _p1, Vec3d& _p2, Vec3d& _p3, Vec4d& _c1)
{
	name = _name;
	p1 = _p1;
	p2 = _p2;
	p3 = _p3;
	color1 = _c1;
	color2 = _c1;
	color3 = _c1;
	type = 2;
	genVer= false;
	generate();
}


// name, p1, p2, p3
PyramidShape::PyramidShape(string _name, Vec3d& _p1, Vec3d& _p2, Vec3d& _p3)
{
	name = _name;
	p1 = _p1;
	p2 = _p2;
	p3 = _p3;
	color1 = Vec4d(0.6,0.2,0.8,0.3);
	color2 = color1;
	color3 = color1;
	type = 2;
	genVer= false;
	generate();
}

// name, center, length, color
PyramidShape::PyramidShape(string _name, Vec3d& _center, int _length, Vec4d& _color)
{
	name = _name;
	center = _center;
	length = _length;
	color1 = _color;	
	color2 = _color;	
	type = 1;
	genVer= true;
	generate();
}

// name, center, length, color, gradient
PyramidShape::PyramidShape(string _name, Vec3d& _center, int _length, Vec4d& _color, Vec4d& _c2)
{
	name = _name;
	center = _center;
	length = _length;
	color1 = _color;
	color2 = _c2;		
	type = 1;
	genVer= true;
	generate();
}


// name, center, length
PyramidShape::PyramidShape(string _name, Vec3d& _center, int _length)
{
	name = _name;
	center = _center;
	length = _length;
	color1 = Vec4d(0.6,0.2,0.8,0.3);
	color2 = color1;		
	type = 1;
	genVer= true;
	generate();
}


void PyramidShape::setCenter(Vec3d& _center)
{
	center = _center;
}

void PyramidShape::setLength(int _length)
{
	length = _length;
}

int PyramidShape::getLength()
{
	return length;
}


int PyramidShape::getType()
{
	return type;
}

int PyramidShape::getId()
{
	return id;
}

void PyramidShape::setName(string _name)
{
	name = _name;
}

string PyramidShape::getName()
{
	return name;
}

void PyramidShape::setColor1(Vec4d& _color)
{
	color1 = _color;
}


Vec4d PyramidShape::getColor1()
{
	return color1;
}


void PyramidShape::setColor2(Vec4d& _color)
{
	color2 = _color;
}


Vec4d PyramidShape::getColor2()
{
	return color2;
}

void PyramidShape::setColor3(Vec4d& _color)
{
	color3 = _color;
}


Vec4d PyramidShape::getColor3()
{
	return color3;
}

void PyramidShape::setPoints(Vec3d& _p1, Vec3d& _p2, Vec3d& _p3)
{
	p1 = _p1;
	p2 = _p2;
	p3 = _p3;
}

void PyramidShape::generate()	
{
	/****************************************************************
	 *								*
	 *			Vertices				*
	 *								*
	 ****************************************************************
	 */
	


	// check if all 3 points are present, otherwise use default method to calculate vertices
	if(!genVer)
	{		
		vertices = new Vec3Array(3);
		(*vertices)[0].set(p1);
		(*vertices)[1].set(p2);
		(*vertices)[2].set(p3);		
	}
	else if (genVer)
	{
		vertices = new Vec3Array();
		// top point
		Vec3d v1(center.x(), center.y(), center.z()+length/2);
		// bottom left point
		Vec3d v2((center.x()-cos(120)*length/2), center.y(), (center.z()-sin(-60)*length/2));
		// bottom right point
		Vec3d v3((center.x()-cos(60)*length/2), center.y(), (center.z()-sin(-60)*length/2));
				
		vertices->push_back(center);
		vertices->push_back(v1);
		vertices->push_back(center);
		vertices->push_back(v2);
		vertices->push_back(center);
		vertices->push_back(v3);	
		vertices->push_back(center);
		vertices->push_back(v1);
			
		
	}

	setVertexArray(vertices);

	/****************************************************************
	 *								*
	 *			normals					*
	 *								*
	 ****************************************************************
	 */


	Vec3Array* normals = new Vec3Array(1);
	(*normals)[0].set(1.0f, 1.0f, 0.0f);
	setNormalArray(normals);
	setNormalBinding(Geometry::BIND_OVERALL);

	/****************************************************************
	 *								*
	 *			colors					*
	 *								*
	 ****************************************************************
	 */
	
	if (!genVer)
	{
		colors = new Vec4Array(3);
		(*colors)[0].set(color1.x(), color1.y(), color1.z(), color1.w());
		(*colors)[1].set(color2.x(), color2.y(), color2.z(), color2.w());
		(*colors)[2].set(color3.x(), color3.y(), color3.z(), color3.w());
		addPrimitiveSet(new DrawArrays(PrimitiveSet::TRIANGLES, 0, 3));

	}
	else
	{
		colors = new Vec4Array();
		colors->push_back(color1);
		colors->push_back(color2);
		colors->push_back(color1);
		colors->push_back(color2);
		colors->push_back(color1);
		colors->push_back(color2);
		colors->push_back(color1);
		colors->push_back(color2);
		
		
		
		addPrimitiveSet(new DrawArrays(PrimitiveSet::TRIANGLE_STRIP, 0, 8));	
	}
	
	
	
	setColorArray(colors);
	setColorBinding(Geometry::BIND_PER_VERTEX);	
	

	
	/****************************************************************
	 *								*
	 *		stateset and material				*
	 *								*
	 ****************************************************************
	 */

	
	StateSet* state = getOrCreateStateSet();
	state->setMode(GL_BLEND,StateAttribute::ON|StateAttribute::OVERRIDE);
	Material* mat = new Material(); 
	mat->setAlpha(Material::FRONT_AND_BACK, 0.1);
	mat->setColorMode(Material::AMBIENT_AND_DIFFUSE);
	state->setAttributeAndModes(mat,StateAttribute::ON | StateAttribute::OVERRIDE);


	/****************************************************************
	 *								*
	 *			blending				*
	 *								*
	 ****************************************************************
	 */

	BlendFunc* bf = new BlendFunc(BlendFunc::SRC_ALPHA, BlendFunc::ONE_MINUS_SRC_ALPHA );
	state->setAttributeAndModes(bf);

	state->setRenderingHint(StateSet::TRANSPARENT_BIN);
	state->setMode(GL_LIGHTING, StateAttribute::ON);

	setStateSet(state);


	
}

void PyramidShape::updateLocation()
{
	// check if all 3 points are present, otherwise use default method to calculate vertices
	if(p1.valid() && p2.valid() && p3.valid() && !genVer)
	{		
		(*vertices)[0].set(p1);
		(*vertices)[1].set(p2);
		(*vertices)[2].set(p3);		
	}
	else if (genVer)
	{
		// top point
		Vec3d v1(center.x(), center.y(), center.z()+length/2);
		// bottom left point
		Vec3d v2((center.x()-cos(120)*length/2), center.y(), (center.z()-sin(-60)*length/2));
		// bottom right point
		Vec3d v3((center.x()-cos(60)*length/2), center.y(), (center.z()-sin(-60)*length/2));
				
		(*vertices)[0].set(center);
		(*vertices)[1].set(v1);
		(*vertices)[2].set(center);
		(*vertices)[3].set(v2);
		(*vertices)[4].set(center);
		(*vertices)[5].set(v3);	
		(*vertices)[6].set(center);
		(*vertices)[7].set(v1);		
	}
	setVertexArray(vertices);
}

void PyramidShape::updateColor()
{
	
	if (!genVer)
	{
		(*colors)[0].set(color1.x(), color1.y(), color1.z(), color1.w());
		(*colors)[1].set(color2.x(), color2.y(), color2.z(), color2.w());
		(*colors)[2].set(color3.x(), color3.y(), color3.z(), color3.w());
	}
	else
	{
		(*colors)[0] = color1;
		(*colors)[1] = color2;
		(*colors)[2] = color1;
		(*colors)[3] = color2;
		(*colors)[4] = color1;
		(*colors)[5] = color2;
		(*colors)[6] = color1;
		(*colors)[7] = color2;
	}
	setColorArray(colors);
}

void PyramidShape::updateAll()
{
	updateLocation();
	updateColor();
}



void PyramidShape::setId(int _id)
{
	id = _id;
}
