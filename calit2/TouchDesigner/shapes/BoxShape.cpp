#include "BoxShape.h"

#include <fcntl.h>
#include <iostream>
#include <cstring>

#include <osg/Matrix>
#include <osg/CullFace>
#include <osgDB/ReadFile>

using namespace osg;
using namespace std;


BoxShape::BoxShape()
{
	name = "";
	center = Vec3d(0,0,0);
	width = 100;
	height = 100;
	color1 = Vec4d(0.8,0.5,0.3,0.3);		
	type = 9;
	genVer= true;
	generate();
}

BoxShape::~BoxShape()
{

}

BoxShape::BoxShape(string _name)
{
	name = _name;
	center = Vec3d(0,0,0);
	width = 100;
	height = 100;
	color1 = Vec4d(0.8,0.5,0.3,0.3);		
	type = 9;
	genVer= true;
	generate();
}


// name, p1, p2, p3, p4, p5, p6, p7, p8, color1, color2
BoxShape::BoxShape(string _name, Vec3d& _p1, Vec3d& _p2, Vec3d& _p3, Vec3d& _p4, 
                                        Vec3d& _p5, Vec3d& _p6, Vec3d& _p7, Vec3d& _p8, Vec4d& _c1, Vec4d& _c2)
{
	name = _name;
	p1 = _p1;
	p2 = _p2;
	p3 = _p3;
	p4 = _p4;
	p5 = _p5;
	p6 = _p6;
	p7 = _p7;
	p8 = _p8;
	color1 = _c1;
	color2 = _c2;	
	type = 8;
	genVer= false;
	generate();
}

// name, p1, p2, p3, p4, p5, p6, p7, p8, color1
BoxShape::BoxShape(string _name, Vec3d& _p1, Vec3d& _p2, Vec3d& _p3, Vec3d& _p4, 
                                  Vec3d& _p5, Vec3d& _p6, Vec3d& _p7, Vec3d& _p8, Vec4d& _c1)
{
	name = _name;
	p1 = _p1;
	p2 = _p2;
	p3 = _p3;
	p4 = _p4;
	p5 = _p5;
	p6 = _p6;
	p7 = _p7;
	p8 = _p8;
	color1 = _c1;	
	color2 = _c1;
	type = 8;
	genVer= false;
	generate();
}

// name, p1, p2, p3, p4, p5, p6, p7, p8
BoxShape::BoxShape(string _name, Vec3d& _p1, Vec3d& _p2, Vec3d& _p3, Vec3d& _p4, Vec3d& _p5, Vec3d& _p6, Vec3d& _p7, Vec3d& _p8)
{
	name = _name;
	p1 = _p1;
	p2 = _p2;
	p3 = _p3;
	p4 = _p4;
	p5 = _p5;
	p6 = _p6;
	p7 = _p7;
	p8 = _p8;
	color1 = Vec4d(0.8,0.5,0.3,0.3);
	color2 = color1;
	type = 8;
	genVer= false;
	generate();
}

// name, center, width, height, color
BoxShape::BoxShape(string _name, Vec3d& _center, int _width, int _height, Vec4d& _color)
{
	name = _name;
	center = _center;
	width = _width;
	height = _height;
	color1 = _color;	
	color2 = _color;	
	type = 9;
	genVer= true;
	generate();
}


// name, center, width, height, color, gradient
BoxShape::BoxShape(string _name, Vec3d& _center, int _width, int _height, Vec4d& _color, Vec4d& _color2)
{
	name = _name;
	center = _center;
	width = _width;
	height = _height;
	color1 = _color;	
	color2 = _color2;	
	type = 9;
	genVer= true;
	generate();
}



// name, center, width
BoxShape::BoxShape(string _name, Vec3d& _center, int _width, int _height)
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


void BoxShape::setCenter(Vec3d& _center)
{
	center = _center;
}

void BoxShape::setWidth(int _width)
{
	width = _width;
}

int BoxShape::getWidth()
{
	return width;
}

void BoxShape::setHeight(int _height)
{
	height = _height;
}

int BoxShape::getHeight()
{
	return height;
}


int BoxShape::getType()
{
	return type;
}

int BoxShape::getId()
{
	return id;
}

void BoxShape::setName(string _name)
{
	name = _name;
}

string BoxShape::getName()
{
	return name;
}

void BoxShape::setColor1(Vec4d& _color)
{
	color1 = _color;
}


Vec4d BoxShape::getColor1()
{
	return color1;
}


void BoxShape::setColor2(Vec4d& _color)
{
	color2 = _color;
}


Vec4d BoxShape::getColor2()
{
	return color2;
}

void BoxShape::setPoints(Vec3d& _p1, Vec3d& _p2, Vec3d& _p3, Vec3d& _p4, Vec3d& _p5, Vec3d& _p6, Vec3d& _p7, Vec3d& _p8)
{
	p1 = _p1;
	p2 = _p2;
	p3 = _p3;
	p4 = _p4;
	p5 = _p5;
	p6 = _p6;
	p7 = _p7;
	p8 = _p8;
}


void BoxShape::generate()	
{
	/****************************************************************
	 *								*
	 *			Vertices				*
	 *								*
	 ****************************************************************
	 */
	vertices = new Vec3Array();

  //assuming box points coming this way:        5-----------6
  //                                           /|          /|
  //                                          / |         / |
  //                                         /  |        /  |
  //                                        0---4-------3---7
  //                                        |  /        |  /
  //                                        | /         | / 
  //                                        |/          |/   
  //                                        1-----------2 




	// check if all 4 points are present, otherwise use default method to calculate vertices
	if(!genVer)
	{		
		vertices->push_back(p1);
		vertices->push_back(p2);
		vertices->push_back(p3);		
		vertices->push_back(p4);
		vertices->push_back(p5);
		vertices->push_back(p6);
		vertices->push_back(p7);		
		vertices->push_back(p8);
		
		//setVertexArray(vertices);
			
		
	}
	else if (genVer)
	{		                                                    // -y = coming out of screen?
		vertices->push_back(Vec3( center.x()-(width/2) , center.y()-(depth/2) , center.z()+(height/2) ));
		vertices->push_back(Vec3( center.x()-(width/2) , center.y()-(depth/2) , center.z()-(height/2) ));
		vertices->push_back(Vec3( center.x()+(width/2) , center.y()-(depth/2) , center.z()-(height/2) ));		
		vertices->push_back(Vec3( center.x()+(width/2) , center.y()-(depth/2) , center.z()+(height/2) ));
		vertices->push_back(Vec3( center.x()-(width/2) , center.y()+(depth/2) , center.z()-(height/2) ));
		vertices->push_back(Vec3( center.x()-(width/2) , center.y()+(depth/2) , center.z()+(height/2) ));
		vertices->push_back(Vec3( center.x()+(width/2) , center.y()+(depth/2) , center.z()+(height/2) ));		
		vertices->push_back(Vec3( center.x()+(width/2) , center.y()+(depth/2) , center.z()-(height/2) ));
		
		//setVertexArray(vertices);
	
	}

	  setVertexArray(vertices);
  	
  	//front face
		DrawElementsUInt* frontface = new DrawElementsUInt(PrimitiveSet::QUADS, 0);
		frontface->push_back(0);
		frontface->push_back(1);
		frontface->push_back(2);
		frontface->push_back(3);
		addPrimitiveSet(frontface);
		//back face
		DrawElementsUInt* backface = new DrawElementsUInt(PrimitiveSet::QUADS, 0);
		backface->push_back(4);
		backface->push_back(5);
		backface->push_back(6);
		backface->push_back(7);
		addPrimitiveSet(backface);
		//top face
		DrawElementsUInt* topface = new DrawElementsUInt(PrimitiveSet::QUADS, 0);
		topface->push_back(5);
		topface->push_back(0);
		topface->push_back(3);
		topface->push_back(6);
		addPrimitiveSet(topface);
		//bottom face
		DrawElementsUInt* bottomface = new DrawElementsUInt(PrimitiveSet::QUADS, 0);
		bottomface->push_back(1);
		bottomface->push_back(4);
		bottomface->push_back(7);
		bottomface->push_back(2);
		addPrimitiveSet(bottomface);
		//left face
		DrawElementsUInt* leftface = new DrawElementsUInt(PrimitiveSet::QUADS, 0);
		leftface->push_back(0);
		leftface->push_back(5);
		leftface->push_back(4);
		leftface->push_back(1);
		addPrimitiveSet(leftface);
		//right face
		DrawElementsUInt* rightface = new DrawElementsUInt(PrimitiveSet::QUADS, 0);
		rightface->push_back(3);
		rightface->push_back(2);
		rightface->push_back(7);
		rightface->push_back(6);
		addPrimitiveSet(rightface);



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
	/*if (!genVer)
	{
		colors = new Vec4Array(8);
		(*colors)[0].set(color1.x(), color1.y(), color1.z(), color1.w());
		(*colors)[1].set(color1.x(), color1.y(), color1.z(), color1.w());
		(*colors)[2].set(color2.x(), color2.y(), color2.z(), color2.w());		
		(*colors)[3].set(color2.x(), color2.y(), color2.z(), color2.w());	
		(*colors)[4].set(color1.x(), color1.y(), color1.z(), color1.w());
		(*colors)[5].set(color1.x(), color1.y(), color1.z(), color1.w());
		(*colors)[6].set(color2.x(), color2.y(), color2.z(), color2.w());		
		(*colors)[7].set(color2.x(), color2.y(), color2.z(), color2.w());	
		
	}
	else
	{*/
		colors = new Vec4Array(8);
		(*colors)[0].set(color1.x(), color1.y(), color1.z(), color1.w());
		(*colors)[1].set(color2.x(), color2.y(), color2.z(), color2.w());		
		(*colors)[2].set(color1.x(), color1.y(), color1.z(), color1.w());
		(*colors)[3].set(color2.x(), color2.y(), color2.z(), color2.w());
		(*colors)[4].set(color1.x(), color1.y(), color1.z(), color1.w());
		(*colors)[5].set(color2.x(), color2.y(), color2.z(), color2.w());
		(*colors)[6].set(color1.x(), color1.y(), color1.z(), color1.w());
		(*colors)[7].set(color2.x(), color2.y(), color2.z(), color2.w());
		//(*colors)[8].set(color1.x(), color1.y(), color1.z(), color1.w());	
		//(*colors)[9].set(color2.x(), color2.y(), color2.z(), color2.w());
	
	//}
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

void BoxShape::updateLocation()
{
	// check if all points are present, otherwise use default method to calculate vertices
	if(!genVer)
	{		
		(*vertices)[0] = p1;
		(*vertices)[1] = p2;
		(*vertices)[2] = p3;		
		(*vertices)[3] = p4;
		(*vertices)[4] = p5;
		(*vertices)[5] = p6;
		(*vertices)[6] = p7;		
		(*vertices)[7] = p8;		
	}
	else if (genVer)	
	{		
	  Vec3d ftl( center.x()-(width/2) , center.y()-(depth/2) , center.z()+(height/2) ));
		Vec3d fbl( center.x()-(width/2) , center.y()-(depth/2) , center.z()-(height/2) ));
		Vec3d fbr( center.x()+(width/2) , center.y()-(depth/2) , center.z()-(height/2) ));		
		Vec3d ftr( center.x()+(width/2) , center.y()-(depth/2) , center.z()+(height/2) ));
		Vec3d bbl( center.x()-(width/2) , center.y()+(depth/2) , center.z()-(height/2) ));
		Vec3d btl( center.x()-(width/2) , center.y()+(depth/2) , center.z()+(height/2) ));
		Vec3d btr( center.x()+(width/2) , center.y()+(depth/2) , center.z()+(height/2) ));		
		Vec3d bbr( center.x()+(width/2) , center.y()+(depth/2) , center.z()-(height/2) ));

		(*vertices)[0] = ftl;   //front top left
		(*vertices)[1] = fbl;   //front bottom left
		(*vertices)[2] = fbr;
		(*vertices)[3] = ftr;
		(*vertices)[4] = bbl;   //back bottom left
		(*vertices)[5] = btl;
		(*vertices)[6] = btr;   //back top right
		(*vertices)[7] = bbr;

	
	}

	setVertexArray(vertices);

}

void BoxShape::updateColor()
{
	/*if (!genVer)
	{		
		(*colors)[0].set(color1.x(), color1.y(), color1.z(), color1.w());
		(*colors)[1].set(color1.x(), color1.y(), color1.z(), color1.w());
		(*colors)[2].set(color2.x(), color2.y(), color2.z(), color2.w());		
		(*colors)[3].set(color2.x(), color2.y(), color2.z(), color2.w());	
	}
	else
	{*/
		(*colors)[0].set(color1.x(), color1.y(), color1.z(), color1.w());
		(*colors)[1].set(color2.x(), color2.y(), color2.z(), color2.w());		
		(*colors)[2].set(color1.x(), color1.y(), color1.z(), color1.w());
		(*colors)[3].set(color2.x(), color2.y(), color2.z(), color2.w());
		(*colors)[4].set(color1.x(), color1.y(), color1.z(), color1.w());
		(*colors)[5].set(color2.x(), color2.y(), color2.z(), color2.w());
		(*colors)[6].set(color1.x(), color1.y(), color1.z(), color1.w());
		(*colors)[7].set(color2.x(), color2.y(), color2.z(), color2.w());
	//	(*colors)[8].set(color1.x(), color1.y(), color1.z(), color1.w());	
		//(*colors)[9].set(color2.x(), color2.y(), color2.z(), color2.w());
	
	//}
	setColorArray(colors);
}

void BoxShape::updateAll()
{
	updateLocation();
	updateColor();
}

void BoxShape::setId(int _id)
{
	id = _id;
}
