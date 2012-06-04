#include "RectShape.h"

#include <fcntl.h>
#include <iostream>
#include <cstring>

#include <osg/Matrix>
#include <osg/CullFace>
#include <osgDB/ReadFile>

using namespace osg;
using namespace std;


RectShape::RectShape()
{
	name = "";
	center = Vec3d(0,0,0);
	width = 100;
	height = 100;
	color1 = Vec4d(0.8,0.5,0.3,0.3);		
	type = 3;
	genVer= true;
	generate();
}

RectShape::~RectShape()
{

}

RectShape::RectShape(string _name)
{
	name = _name;
	center = Vec3d(0,0,0);
	width = 100;
	height = 100;
	color1 = Vec4d(0.8,0.5,0.3,0.3);		
	type = 3;
	genVer= true;
	generate();
}


// name, p1, p2, p3, p4, color1, color2
RectShape::RectShape(string _name, Vec3d& _p1, Vec3d& _p2, Vec3d& _p3, Vec3d& _p4, Vec4d& _c1, Vec4d& _c2)
{
	name = _name;
	p1 = _p1;
	p2 = _p2;
	p3 = _p3;
	p4 = _p4;
	color1 = _c1;
	color2 = _c2;	
	type = 4;
	genVer= false;
}

// name, p1, p2, p3, p4, color1
RectShape::RectShape(string _name, Vec3d& _p1, Vec3d& _p2, Vec3d& _p3, Vec3d& _p4, Vec4d& _c1)
{
	name = _name;
	p1 = _p1;
	p2 = _p2;
	p3 = _p3;
	p4 = _p4;
	color1 = _c1;	
	color2 = _c1;
	type = 4;
	genVer= false;
	generate();
}

// name, p1, p2, p3, p4
RectShape::RectShape(string _name, Vec3d& _p1, Vec3d& _p2, Vec3d& _p3, Vec3d& _p4)
{
	name = _name;
	p1 = _p1;
	p2 = _p2;
	p3 = _p3;
	p4 = _p4;
	color1 = Vec4d(0.8,0.5,0.3,0.3);
	color2 = color1;
	type = 4;
	genVer= false;
	generate();
}

// name, center, width, height, color
RectShape::RectShape(string _name, Vec3d& _center, int _width, int _height, Vec4d& _color)
{
	name = _name;
	center = _center;
	width = _width;
	height = _height;
	color1 = _color;	
	color2 = _color;	
	type = 3;
	genVer= true;
	generate();
}


// name, center, width, height, color, gradient
RectShape::RectShape(string _name, Vec3d& _center, int _width, int _height, Vec4d& _color, Vec4d& _color2)
{
	name = _name;
	center = _center;
	width = _width;
	height = _height;
	color1 = _color;	
	color2 = _color2;	
	type = 3;
	genVer= true;
	generate();
}



// name, center, width
RectShape::RectShape(string _name, Vec3d& _center, int _width, int _height)
{
	name = _name;
	center = _center;
	width = _width;
	height = _height;
	color1 = Vec4d(0.8,0.5,0.3,0.3);	
	color2 = color1;	
	type = 3;
	genVer= true;
	generate();
}


void RectShape::setCenter(Vec3d& _center)
{
	center = _center;
}

void RectShape::setWidth(int _width)
{
	width = _width;
}

int RectShape::getWidth()
{
	return width;
}

void RectShape::setHeight(int _height)
{
	height = _height;
}

int RectShape::getHeight()
{
	return height;
}


int RectShape::getType()
{
	return type;
}

int RectShape::getId()
{
	return id;
}

void RectShape::setName(string _name)
{
	name = _name;
}

string RectShape::getName()
{
	return name;
}

void RectShape::setColor1(Vec4d& _color)
{
	color1 = _color;
}


Vec4d RectShape::getColor1()
{
	return color1;
}


void RectShape::setColor2(Vec4d& _color)
{
	color2 = _color;
}


Vec4d RectShape::getColor2()
{
	return color2;
}

void RectShape::setPoints(Vec3d& _p1, Vec3d& _p2, Vec3d& _p3, Vec3d& _p4)
{
	p1 = _p1;
	p2 = _p2;
	p3 = _p3;
	p4 = _p4;
}


void RectShape::generate()	
{
	/****************************************************************
	 *								*
	 *			Vertices				*
	 *								*
	 ****************************************************************
	 */
	vertices = new Vec3Array();


	// check if all 4 points are present, otherwise use default method to calculate vertices
	if(!genVer)
	{		
		vertices->push_back(p1);
		vertices->push_back(p2);
		vertices->push_back(p3);		
		vertices->push_back(p4);
		addPrimitiveSet(new DrawArrays(PrimitiveSet::QUADS, 0, 4));	
	}
	else if (genVer)
	{		
		vertices->push_back(center);
		vertices->push_back(Vec3d((-(width/2))+center.x(), center.y(), -(height/2)+center.z()));   // bottom left
		vertices->push_back(center);
		vertices->push_back(Vec3d((width/2)+center.x(), center.y(), -(height/2)+center.z()));   // bottom right
		vertices->push_back(center);
		vertices->push_back(Vec3d((width/2)+center.x(), center.y(), (height/2)+center.z()));    // top right
		vertices->push_back(center);
		vertices->push_back(Vec3d((-(width/2))+center.x(), center.y(), (height/2)+center.z()));    // top left
		vertices->push_back(center);
		vertices->push_back(Vec3d((-(width/2))+center.x(), center.y(), -(height/2)+center.z()));   // bottom left
		addPrimitiveSet(new DrawArrays(PrimitiveSet::TRIANGLE_STRIP, 0,10));
	
	}

	setVertexArray(vertices);

	/****************************************************************
	 *								*
	 *			normals	  				*
	 *								*
	 ****************************************************************
	 */


	Vec3Array* normals = new Vec3Array(1);
	(*normals)[0].set(1.0f, 1.0f, 1.0f);
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
		colors = new Vec4Array(4);
		(*colors)[0].set(color1.x(), color1.y(), color1.z(), color1.w());
		(*colors)[1].set(color1.x(), color1.y(), color1.z(), color1.w());
		(*colors)[2].set(color2.x(), color2.y(), color2.z(), color2.w());		
		(*colors)[3].set(color2.x(), color2.y(), color2.z(), color2.w());	
	}
	else
	{
		colors = new Vec4Array(10);
		(*colors)[0].set(color1.x(), color1.y(), color1.z(), color1.w());
		(*colors)[1].set(color2.x(), color2.y(), color2.z(), color2.w());		
		(*colors)[2].set(color1.x(), color1.y(), color1.z(), color1.w());
		(*colors)[3].set(color2.x(), color2.y(), color2.z(), color2.w());
		(*colors)[4].set(color1.x(), color1.y(), color1.z(), color1.w());
		(*colors)[5].set(color2.x(), color2.y(), color2.z(), color2.w());
		(*colors)[6].set(color1.x(), color1.y(), color1.z(), color1.w());
		(*colors)[7].set(color2.x(), color2.y(), color2.z(), color2.w());
		(*colors)[8].set(color1.x(), color1.y(), color1.z(), color1.w());	
		(*colors)[9].set(color2.x(), color2.y(), color2.z(), color2.w());
	
	}
	setColorArray(colors);
	setColorBinding(Geometry::BIND_PER_VERTEX);	

	


	
	/****************************************************************
	 *								*
	 *			stateset and material			*
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

void RectShape::updateLocation()
{
	// check if all 4 points are present, otherwise use default method to calculate vertices
	if(!genVer)
	{		
		(*vertices)[0] = p1;
		(*vertices)[1] = p2;
		(*vertices)[2] = p3;		
		(*vertices)[3] = p4;	
	}
	else if (genVer)	
	{		
		Vec3d bl(-(width/2)+center.x(), center.y(), -(height/2)+center.z());
		Vec3d br((width/2)+center.x(), center.y(), -(height/2)+center.z());
		Vec3d tr((width/2)+center.x(), center.y(), (height/2)+center.z());
		Vec3d tl((-(width/2))+center.x(), center.y(), (height/2)+center.z());
		
		
		(*vertices)[0] = center;
		(*vertices)[1] = bl;  // bottom left
		(*vertices)[2] = center;
		(*vertices)[3] = br;   // bottom right
		(*vertices)[4] = center;
		(*vertices)[5] = tr;   // top right
		(*vertices)[6] = center;
		(*vertices)[7] = tl;    // top left
		(*vertices)[8] = center;
		(*vertices)[9] = bl;   // bottom left
	
	}

	setVertexArray(vertices);

}

void RectShape::updateColor()
{
	if (!genVer)
	{		
		(*colors)[0].set(color1.x(), color1.y(), color1.z(), color1.w());
		(*colors)[1].set(color1.x(), color1.y(), color1.z(), color1.w());
		(*colors)[2].set(color2.x(), color2.y(), color2.z(), color2.w());		
		(*colors)[3].set(color2.x(), color2.y(), color2.z(), color2.w());	
	}
	else
	{
		(*colors)[0].set(color1.x(), color1.y(), color1.z(), color1.w());
		(*colors)[1].set(color2.x(), color2.y(), color2.z(), color2.w());		
		(*colors)[2].set(color1.x(), color1.y(), color1.z(), color1.w());
		(*colors)[3].set(color2.x(), color2.y(), color2.z(), color2.w());
		(*colors)[4].set(color1.x(), color1.y(), color1.z(), color1.w());
		(*colors)[5].set(color2.x(), color2.y(), color2.z(), color2.w());
		(*colors)[6].set(color1.x(), color1.y(), color1.z(), color1.w());
		(*colors)[7].set(color2.x(), color2.y(), color2.z(), color2.w());
		(*colors)[8].set(color1.x(), color1.y(), color1.z(), color1.w());	
		(*colors)[9].set(color2.x(), color2.y(), color2.z(), color2.w());
	
	}
	setColorArray(colors);
}

void RectShape::updateAll()
{
	updateLocation();
	updateColor();
}

void RectShape::setId(int _id)
{
	id = _id;
}
