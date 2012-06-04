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
	type = 7;
	sides = 3;
	genVer= true;
	generate();
}



PyramidShape::~PyramidShape()
{

}

PyramidShape::PyramidShape(string _name)
{
	name = _name;
	center = Vec3d(0,0,0);
	length = 10;
	color1 = Vec4d(0.6,0.2,0.8,0.3);		
	type = 7;
	sides = 3;
	genVer= true;
	generate();
}

/****************3 sided*****************************/
// name, p1, p2, p3, p4, color1, color2, color3
PyramidShape::PyramidShape(string _name, Vec3d& _p1, Vec3d& _p2, Vec3d& _p3, Vec3d& _p4, Vec4d& _c1, Vec4d& _c2, Vec4d& _c3)
{
	name = _name;
	p1 = _p1;
	p2 = _p2;
	p3 = _p3;
	p4 = _p4;
	p5 = NULL;
	color1 = _c1;
	color2 = _c2;
	color3 = _c3;	
	type = 6;
	sides = 3;
	genVer= false;
	generate();
}

// name, p1, p2, p3, p4, color1, color2
PyramidShape::PyramidShape(string _name, Vec3d& _p1, Vec3d& _p2, Vec3d& _p3, Vec3d& _p4, Vec4d& _c1, Vec4d& _c2)
{
	name = _name;
	p1 = _p1;
	p2 = _p2;
	p3 = _p3;
	p4 = _p4;
	p5 = NULL;
	color1 = _c1;
	color2 = _c2;
	color3 = _c2;
	type = 6;
	sides = 3;
	genVer= false;
	generate();
}

// name, p1, p2, p3, p4, color
PyramidShape::PyramidShape(string _name, Vec3d& _p1, Vec3d& _p2, Vec3d& _p3, Vec3d& _p4, Vec4d& _c1)
{
	name = _name;
	p1 = _p1;
	p2 = _p2;
	p3 = _p3;
	p4 = _p4;
	p5 = NULL;
	color1 = _c1;
	color2 = _c1;
	color3 = _c1;
	type = 6;
	sides = 3;
	genVer= false;
	generate();
}


// name, p1, p2, p3, p4
PyramidShape::PyramidShape(string _name, Vec3d& _p1, Vec3d& _p2, Vec3d& _p3, Vec3d& _p4)
{
	name = _name;
	p1 = _p1;
	p2 = _p2;
	p3 = _p3;
	p4 = _p4;
	p5 = NULL;
	color1 = Vec4d(0.6,0.2,0.8,0.3);
	color2 = color1;
	color3 = color1;
	type = 6;
	sides = 3;
	genVer= false;
	generate();
}


/*************4 sided**************************/
// name, p1, p2, p3, p4, p5, color1, color2, color3
PyramidShape::PyramidShape(string _name, Vec3d& _p1, Vec3d& _p2, Vec3d& _p3, Vec3d& _p4, Vec3d& _p5, Vec4d& _c1, Vec4d& _c2, Vec4d& _c3)
{
	name = _name;
	p1 = _p1;
	p2 = _p2;
	p3 = _p3;
	p4 = _p4;
	p5 = _p5;
	color1 = _c1;
	color2 = _c2;
	color3 = _c3;	
	type = 6;
	sides = 4;
	genVer= false;
	generate();
}

// name, p1, p2, p3, p4, p5, color1, color2
PyramidShape::PyramidShape(string _name, Vec3d& _p1, Vec3d& _p2, Vec3d& _p3, Vec3d& _p4, Vec3d& _p5,  Vec4d& _c1, Vec4d& _c2)
{
	name = _name;
	p1 = _p1;
	p2 = _p2;
	p3 = _p3;
	p4 = _p4;
	p5 = _p5;
	color1 = _c1;
	color2 = _c2;
	color3 = _c2;
	type = 6;
	sides = 4;
	genVer= false;
	generate();
}

// name, p1, p2, p3, p4, p5, color
PyramidShape::PyramidShape(string _name, Vec3d& _p1, Vec3d& _p2, Vec3d& _p3, Vec3d& _p4, Vec3d& _p5, Vec4d& _c1)
{
	name = _name;
	p1 = _p1;
	p2 = _p2;
	p3 = _p3;
	p4 = _p4;
	p5 = _p5;
	color1 = _c1;
	color2 = _c1;
	color3 = _c1;
	type = 6;
	sides = 4;
	genVer= false;
	generate();
}


// name, p1, p2, p3, p4, p5
PyramidShape::PyramidShape(string _name, Vec3d& _p1, Vec3d& _p2, Vec3d& _p3, Vec3d& _p4, Vec3d& _p5)
{
	name = _name;
	p1 = _p1;
	p2 = _p2;
	p3 = _p3;
	p4 = _p4;
	p5 = _p5;
	color1 = Vec4d(0.6,0.2,0.8,0.3);
	color2 = color1;
	color3 = color1;
	type = 6;
	sides = 4;
	genVer= false;
	generate();
}





/***********gen with centers****************/
// name, center, sides, height, length, width, color
PyramidShape::PyramidShape(string _name, Vec3d& _center, int _sides, int _height, int _length, int _width, Vec4d& _color)
{
	name = _name;
	center = _center;
	height = _height;
	width = _width;
	length = _length;
	color1 = _color;	
	color2 = _color;	
	type = 7;
	sides = _sides;
	genVer= true;
	generate();
}

// name, center, sides, height, length, width, color, gradient
PyramidShape::PyramidShape(string _name, Vec3d& _center, int _sides, int _height, int _length, int _width, Vec4d& _color, Vec4d& _c2)
{
	name = _name;
	center = _center;
	height = _height;
	width = _width;
	length = _length;
	color1 = _color;
	color2 = _c2;		
	type = 7;
	sides = _sides;
	genVer= true;
	generate();
}


// name, center, sides, height, length, width
PyramidShape::PyramidShape(string _name, Vec3d& _center, int _sides, int _height, int _length, int _width)
{
	name = _name;
	center = _center;
	height = _height;
	width = _width;
	length = _length;
	color1 = Vec4d(0.6,0.2,0.8,0.3);
	color2 = color1;		
	type = 7;
	sides = _sides;
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

void PyramidShape::setPoints(Vec3d& _p1, Vec3d& _p2, Vec3d& _p3, Vec3d& _p4)
{
	p1 = _p1;
	p2 = _p2;
	p3 = _p3;
	p4 = _p4;
	p5 = NULL;
}
void PyramidShape::setPoints(Vec3d& _p1, Vec3d& _p2, Vec3d& _p3, Vec3d& _p4, Vec3d& _p5)
{
	p1 = _p1;
	p2 = _p2;
	p3 = _p3;
	p4 = _p4;
	p5 = _p5;
}

void PyramidShape::generate()	
{
	/****************************************************************
	 *								*
	 *			Vertices				*
	 *								*
	 ****************************************************************
	 */
	
  // 3 sided pyramid:                 3                4 sided pyramid:         4
  //                                 / \                                       / \
  //                                /   \                                     /   \
  //                               /  1  \                                   1-----2
  //                              0-------2                                 /       \
  //                                                                       0---------3


  vertices = new Vec3Array();

	// check if all points are present, otherwise use default method to calculate vertices
	if(!genVer)
	{		
	  if(sides == 3) {
	    vertices->push_back(p1);
	    vertices->push_back(p2);
	    vertices->push_back(p3);
	    vertices->push_back(p4);
		}
		else if(sides == 4) {
		  vertices->push_back(p1);
	    vertices->push_back(p2);
	    vertices->push_back(p3);
	    vertices->push_back(p4);
	    vertices->push_back(p5);
		}
	
	}
	else if (genVer)
	{
	  if(sides == 4) {
		  // top point
	  	Vec3d top(center.x(), center.y(), center.z()+height);
	  	// front left point
	  	Vec3d v1(( center.x()-(width/2) , center.y()-(length/2) , center.z() ));
		  // back left point
	  	Vec3d v2(( center.x()-(width/2) , center.y()+(length/2) , center.z() ));
	  	// back right point
	  	Vec3d v3(( center.x()+(width/2) , center.y()+(length/2) , center.z() ));
	  	// front right point
	  	Vec3d v4(( center.x()+(width/2) , center.y()-(length/2) , center.z() ));
				

	  	vertices->push_back(v1);
		  vertices->push_back(v2);
		  vertices->push_back(v3);	
		  vertices->push_back(v4);
		  vertices->push_back(top);
		  
		  //base
		  DrawElementsUInt* base = new DrawElementsUInt(PrimitiveSet::QUADS, 0);
		  base->push_back(0);
		  base->push_back(3);
		  base->push_back(2);
		  base->push_back(4);
		  addPrimitiveSet(base);
		  //front face
		  DrawElementsUInt* frontface = new DrawElementsUInt(PrimitiveSet::TRIANGLES, 0);
		  frontface->push_back(0);
		  frontface->push_back(1);
	  	frontface->push_back(4);
		  addPrimitiveSet(frontface);
		  //back face
		  DrawElementsUInt* backface = new DrawElementsUInt(PrimitiveSet::TRIANGLES, 0);
		  backface->push_back(3);
		  backface->push_back(4);
	  	backface->push_back(2);
		  addPrimitiveSet(backface);
		  //left face
		  DrawElementsUInt* leftface = new DrawElementsUInt(PrimitiveSet::TRIANGLES, 0);
		  leftface->push_back(0);
		  leftface->push_back(4);
	  	leftface->push_back(3);
		  addPrimitiveSet(leftface);
		  //right face
		  DrawElementsUInt* rightface = new DrawElementsUInt(PrimitiveSet::TRIANGLES, 0);
		  rightface->push_back(4);
		  rightface->push_back(1);
	  	rightface->push_back(2);
		  addPrimitiveSet(rightface);
	  
		  
		}
		else if(sides == 3) {
		  // top point
	  	Vec3d top(center.x(), center.y(), center.z()+height);
	  	// front left point
	  	Vec3d v1((center.x()-(width/2), center.y()-(w/(2*sqrt(3))) , center.z()));
		  // back point
	  	Vec3d v2((center.x() , center.y()+(width/sqrt(3)) , center.z());
	  	// front right point
	  	Vec3d v3((center.x()+(width/2) , center.y()-(width/(2*sqrt(3)) , center.z());
	  
		  vertices->push_back(v1);
		  vertices->push_back(v2);
		  vertices->push_back(v3);	
		  vertices->push_back(top);
		  
		  //base
		  DrawElementsUInt* base = new DrawElementsUInt(PrimitiveSet::TRIANGLES, 0);
		  base->push_back(0);
		  base->push_back(1);
		  base->push_back(2);
		  addPrimitiveSet(base);
		  //front face
		  DrawElementsUInt* frontface = new DrawElementsUInt(PrimitiveSet::TRIANGLES, 0);
		  frontface->push_back(0);
		  frontface->push_back(2);
	  	frontface->push_back(3);
		  addPrimitiveSet(frontface);
		  //left face
		  DrawElementsUInt* leftface = new DrawElementsUInt(PrimitiveSet::TRIANGLES, 0);
		  leftface->push_back(0);
		  leftface->push_back(3);
	  	leftface->push_back(1);
		  addPrimitiveSet(leftface);
		  //right face
		  DrawElementsUInt* rightface = new DrawElementsUInt(PrimitiveSet::TRIANGLES, 0);
		  rightface->push_back(2);
		  rightface->push_back(1);
	  	rightface->push_back(3);
		  addPrimitiveSet(rightface);
		}
		
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
	
	//if (!genVer)
	//{
	  if(sides == 3) {
		  colors = new Vec4Array(4);
		  (*colors)[0].set(color1.x(), color1.y(), color1.z(), color1.w());
		  (*colors)[1].set(color2.x(), color2.y(), color2.z(), color2.w());
		  (*colors)[2].set(color3.x(), color3.y(), color3.z(), color3.w());
		  (*colors)[3].set(color3.x(), color3.y(), color3.z(), color3.w());
		  //addPrimitiveSet(new DrawArrays(PrimitiveSet::TRIANGLES, 0, 3));
    }
    else if(sides == 4) {
      colors = new Vec4Array(5);
		  (*colors)[0].set(color1.x(), color1.y(), color1.z(), color1.w());
		  (*colors)[1].set(color2.x(), color2.y(), color2.z(), color2.w());
		  (*colors)[2].set(color3.x(), color3.y(), color3.z(), color3.w());
		  (*colors)[3].set(color2.x(), color2.y(), color2.z(), color2.w());
		  (*colors)[4].set(color3.x(), color3.y(), color3.z(), color3.w());
		  //addPrimitiveSet(new DrawArrays(PrimitiveSet::TRIANGLES, 0, 3));		  
    }
	//}
	/*else
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
		
		
		
		//addPrimitiveSet(new DrawArrays(PrimitiveSet::TRIANGLE_STRIP, 0, 8));	
	}*/
	
	
	
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
	// check if all points are present, otherwise use default method to calculate vertices
	if(sides == 3 && p1.valid() && p2.valid() && p3.valid() && p4.valid() && !genVer)
	{		
		(*vertices)[0].set(p1);
		(*vertices)[1].set(p2);
		(*vertices)[2].set(p3);
		(*vertices)[3].set(p4);		
	}
	else if(sides == 4 && p1.valid() && p2.valid() && p3.valid() && p4.valid() && p5.valid() && !genVer)
	{
	  (*vertices)[0].set(p1);
		(*vertices)[1].set(p2);
		(*vertices)[2].set(p3);
		(*vertices)[3].set(p4);	
		(*vertices)[4].set(p1);
	}
	else if (genVer)
	{
	  if(sides == 3) {
		 // top point
	  	Vec3d top(center.x(), center.y(), center.z()+height);
	  	// front left point
	  	Vec3d v1((center.x()-(width/2), center.y()-(w/(2*sqrt(3))) , center.z()));
		  // back point
	  	Vec3d v2((center.x() , center.y()+(width/sqrt(3)) , center.z());
	  	// front right point
	  	Vec3d v3((center.x()+(width/2) , center.y()-(width/(2*sqrt(3)) , center.z());
	  
		  (*vertices)[0].set(v1);
		  (*vertices)[1].set(v2);
		  (*vertices)[2].set(v3);
		  (*vertices)[3].set(top);
		}
		else if(sides == 4) {
		  // top point
	  	Vec3d top(center.x(), center.y(), center.z()+height);
	  	// front left point
	  	Vec3d v1(( center.x()-(width/2) , center.y()-(length/2) , center.z() ));
		  // back left point
	  	Vec3d v2(( center.x()-(width/2) , center.y()+(length/2) , center.z() ));
	  	// back right point
	  	Vec3d v3(( center.x()+(width/2) , center.y()+(length/2) , center.z() ));
	  	// front right point
	  	Vec3d v4(( center.x()+(width/2) , center.y()-(length/2) , center.z() ));
				
      (*vertices)[0].set(v1);
		  (*vertices)[1].set(v2);
		  (*vertices)[2].set(v3);
		  (*vertices)[3].set(v4);
		  (*vertices)[4].set(top);
		} 
	}
	setVertexArray(vertices);
}

void PyramidShape::updateColor()
{
	
	if (sides == 3)
	{
		(*colors)[0].set(color1.x(), color1.y(), color1.z(), color1.w());
		(*colors)[1].set(color2.x(), color2.y(), color2.z(), color2.w());
		(*colors)[2].set(color3.x(), color3.y(), color3.z(), color3.w());
		(*colors)[3].set(color3.x(), color3.y(), color3.z(), color3.w());
	}
	else if(sides == 4)
	{
		(*colors)[0].set(color1.x(), color1.y(), color1.z(), color1.w());
		(*colors)[1].set(color2.x(), color2.y(), color2.z(), color2.w());
		(*colors)[2].set(color3.x(), color3.y(), color3.z(), color3.w());
		(*colors)[3].set(color3.x(), color3.y(), color3.z(), color3.w());
		(*colors)[4].set(color3.x(), color3.y(), color3.z(), color3.w());
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
