/* Cube Shape for Mugic */

#include "CubeShape.h"

#include <osg/Geometry>
#include <osg/Material>
#include <osg/Texture2D>
#include <osgDB/ReadFile>
#include <osg/PolygonMode>
#include <cvrConfig/ConfigManager.h>

#include <string>
#include <vector>
#include <iostream>

/* Constructor for Cube */
CubeShape::CubeShape(std::string command, std::string name)
{

	_type = SimpleShape::CUBE;
	BasicShape::setName(name);

	_vertices = new osg::Vec3Array(24);
	_colors = new osg::Vec4Array(24);
	_textures = new osg::Vec2Array(24);

	setPosition( osg::Vec3(0.0, 0.0, 0.0), 1.0, 1.0, 1.0 );
	setColor( osg::Vec4(1.0, 1.0, 1.0, 1.0), osg::Vec4(1.0, 1.0, 1.0, 1.0), osg::Vec4(1.0, 1.0, 1.0, 1.0), osg::Vec4(1.0, 1.0, 1.0, 1.0),
		  osg::Vec4(1.0, 1.0, 1.0, 1.0), osg::Vec4(1.0, 1.0, 1.0, 1.0), osg::Vec4(1.0, 1.0, 1.0, 1.0), osg::Vec4(1.0, 1.0, 1.0, 1.0));
	setTextureCoords(); //only happens once
	update(command);
	
	setVertexArray(_vertices);
	setColorArray(_colors);
	setTexCoordArray(0, _textures);
	setColorBinding(osg::Geometry::BIND_PER_VERTEX);
	
	//properly add all faces here using addPrimitiveSet - using indices for desired vertex combinations
	unsigned short front_indices[] = {0, 1, 2, 3};
	unsigned short right_indices[] = {4, 5, 6, 7};
	unsigned short back_indices[] = {8, 9, 10, 11};
	unsigned short left_indices[] = {12, 13, 14, 15};
	unsigned short top_indices[] = {16, 17, 18, 19};
	unsigned short bottom_indices[] = {20, 21, 22, 23};

	addPrimitiveSet(new osg::DrawElementsUShort(osg::PrimitiveSet::QUADS, 4, front_indices ));
	addPrimitiveSet(new osg::DrawElementsUShort(osg::PrimitiveSet::QUADS, 4, right_indices));
	addPrimitiveSet(new osg::DrawElementsUShort(osg::PrimitiveSet::QUADS, 4, back_indices));
	addPrimitiveSet(new osg::DrawElementsUShort(osg::PrimitiveSet::QUADS, 4, left_indices));
	addPrimitiveSet(new osg::DrawElementsUShort(osg::PrimitiveSet::QUADS, 4, top_indices));
	addPrimitiveSet(new osg::DrawElementsUShort(osg::PrimitiveSet::QUADS, 4, bottom_indices));

	osg::StateSet* state = getOrCreateStateSet();
	state -> setMode(GL_BLEND, osg::StateAttribute::ON);

	//possibly needed?
	/*osg::PolygonMode* polyMode = new osg::PolygonMode();
	polyMode->setMode(osg::PolygonMode::FRONT_AND_BACK, osg::PolygonMode::LINE);
	state->setAttribute(polyMode);*/

	osg::Material* mat = new osg::Material();
	mat -> setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
	state ->  setAttributeAndModes(mat, osg::StateAttribute::ON);

	setTextureImage("");

}


/* Destructor */
CubeShape::~CubeShape()
{
}


/* Set position of the Cube - need 8 vertices */
void CubeShape::setPosition(osg::Vec3 p, float width, float height, float depth )
{

	//front face
	(*_vertices)[0].set( p[0] - (width/2), p[1] - (depth/2), p[2] - (height/2) ); //bottom left front
	(*_vertices)[1].set( p[0] + (width/2), p[1] - (depth/2), p[2] - (height/2) ); //bottom right front
	(*_vertices)[2].set( p[0] + (width/2), p[1] - (depth/2), p[2] + (height/2) ); //top right front
	(*_vertices)[3].set( p[0] - (width/2), p[1] - (depth/2), p[2] + (height/2) ); //top left front
	
	//right face
	(*_vertices)[4].set( p[0] + (width/2), p[1] - (depth/2), p[2] - (height/2) ); //bottom right front
	(*_vertices)[5].set( p[0] + (width/2), p[1] + (depth/2), p[2] - (height/2) ); //bottom right back
	(*_vertices)[6].set( p[0] + (width/2), p[1] + (depth/2), p[2] + (height/2) ); //top right back
	(*_vertices)[7].set( p[0] + (width/2), p[1] - (depth/2), p[2] + (height/2) ); //top right front

	//back face
	(*_vertices)[8].set( p[0] + (width/2), p[1] + (depth/2), p[2] - (height/2) ); //bottom right back
	(*_vertices)[9].set( p[0] - (width/2), p[1] + (depth/2), p[2] - (height/2) ); //bottom left back
	(*_vertices)[10].set( p[0] - (width/2), p[1] + (depth/2), p[2] + (height/2) ); //top left back
	(*_vertices)[11].set( p[0] + (width/2), p[1] + (depth/2), p[2] + (height/2) ); //top right back

	//left face
	(*_vertices)[12].set( p[0] - (width/2), p[1] + (depth/2), p[2] - (height/2) ); //bottom left back
	(*_vertices)[13].set( p[0] - (width/2), p[1] - (depth/2), p[2] - (height/2) ); //bottom left front
	(*_vertices)[14].set( p[0] - (width/2), p[1] - (depth/2), p[2] + (height/2) ); //top left front
	(*_vertices)[15].set( p[0] - (width/2), p[1] + (depth/2), p[2] + (height/2) ); //top left back

	//top face

	(*_vertices)[16].set( p[0] - (width/2), p[1] - (width/2), p[2] + (height/2) ); //top left front
	(*_vertices)[17].set( p[0] + (width/2), p[1] - (width/2), p[2] + (height/2) ); //top right front
	(*_vertices)[18].set( p[0] + (width/2), p[1] + (depth/2), p[2] + (height/2) ); //top right back
	(*_vertices)[19].set( p[0] - (width/2), p[1] + (depth/2), p[2] + (height/2) ); //top left back

	//bottom face
	(*_vertices)[20].set( p[0] - (width/2), p[1] + (depth/2), p[2] - (height/2) ); //bottom left back
	(*_vertices)[21].set( p[0] + (width/2), p[1] + (depth/2), p[2] - (height/2) ); //bottom right back
	(*_vertices)[22].set( p[0] + (width/2), p[1] - (depth/2), p[2] - (height/2) ); //bottom right front
	(*_vertices)[23].set( p[0] - (width/2), p[1] - (depth/2), p[2] - (height/2) ); //bottom left front

}

/* set Color of the Cube - can set for each vertex */
void CubeShape::setColor(osg::Vec4 c0, osg::Vec4 c1, osg::Vec4 c2, osg::Vec4 c3,
			 osg::Vec4 c4, osg::Vec4 c5, osg::Vec4 c6, osg::Vec4 c7)
{
	
	//front face
	(*_colors)[0].set( c0[0], c0[1], c0[2], c0[3] ); //bottom left front
	(*_colors)[1].set( c1[0], c1[1], c1[2], c1[3] ); //bottom right front
	(*_colors)[2].set( c5[0], c5[1], c5[2], c5[3] ); //top right front
	(*_colors)[3].set( c4[0], c4[1], c4[2], c4[3] ); //top left front
	
	//right face
	(*_colors)[4].set( c1[0], c1[1], c1[2], c1[3] ); //bottom right front
	(*_colors)[5].set( c2[0], c2[1], c2[2], c2[3] ); //bottom right back

	(*_colors)[6].set( c6[0], c6[1], c6[2], c6[3] ); //top right back
	(*_colors)[7].set( c5[0], c5[1], c5[2], c5[3] ); //top right front

	//back face
	(*_colors)[8].set( c2[0], c2[1], c2[2], c2[3] ); //bottom right back
	(*_colors)[9].set( c3[0], c3[1], c3[2], c3[3] ); //bottom left back
	(*_colors)[10].set( c7[0], c7[1], c7[2], c7[3] ); //top left back
	(*_colors)[11].set( c6[0], c6[1], c6[2], c6[3] ); //top right back

	//left face
	(*_colors)[12].set( c3[0], c3[1], c3[2], c3[3] ); //bottom left back
	(*_colors)[13].set( c0[0], c0[1], c0[2], c0[3] ); //bottom left front
	(*_colors)[14].set( c4[0], c4[1], c4[2], c4[3] ); //top left front
	(*_colors)[15].set( c7[0], c7[1], c7[2], c7[3] ); //top left back

	//top face
	(*_colors)[16].set( c4[0], c4[1], c4[2], c4[3] ); //top left front
	(*_colors)[17].set( c5[0], c5[1], c5[2], c5[3] ); //top right front
	(*_colors)[18].set( c6[0], c6[1], c6[2], c6[3] ); //top right back
	(*_colors)[19].set( c7[0], c7[1], c7[2], c7[3] ); //top left back

	//bottom face
	(*_colors)[20].set( c3[0], c3[1], c3[2], c3[3] ); //bottom left back
	(*_colors)[21].set( c2[0], c2[1], c2[2], c2[3] ); //bottom right back
	(*_colors)[22].set( c1[0], c1[1], c1[2], c1[3] ); //bottom right front
	(*_colors)[23].set( c0[0], c0[1], c0[2], c0[3] ); //bottom left front

	//if w is not 1.0, then opacity in effect?
	if( c0[3] != 1.0 || c1[3] != 1.0 || c2[3] != 1.0 || c3[3] != 1.0 ||
	    c4[3] != 1.0 || c5[3] != 1.0 || c6[3] != 1.0 || c7[3] != 1.0)
		getOrCreateStateSet() -> setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
	else
		getOrCreateStateSet() -> setRenderingHint(osg::StateSet::DEFAULT_BIN);

}

void CubeShape::setTextureCoords()
{
	
	//this is only set once, for convenience sake
	//front faceube texturing
	(*_textures)[0].set( 0.3333333, 0.5 ); //bottom left front
	(*_textures)[1].set( 0.6666666, 0.5 ); //bottom right front
	(*_textures)[2].set( 0.6666666, 0.75 ); //top right front
	(*_textures)[3].set( 0.3333333, 0.75 ); //top left front
	
	//right face
	(*_textures)[4].set( 0.6666666, 0.5 ); //bottom right front
	(*_textures)[5].set( 1.0, 0.5 ); //bottom right back
	(*_textures)[6].set( 1.0, 0.75 ); //top right back
	(*_textures)[7].set( 0.6666666, 0.75 ); //top right front

	//back face

	(*_textures)[8].set( 0.6666666, 0.25 ); //bottom right back
	(*_textures)[9].set( 0.3333333, 0.25 ); //bottom left back
	(*_textures)[10].set( 0.3333333, 0.0 ); //top left back
	(*_textures)[11].set( 0.6666666, 0.0 ); //top right back

	//left face
	(*_textures)[12].set( 0.0, 0.5 ); //bottom left back
	(*_textures)[13].set( 0.3333333, 0.5 ); //bottom left front
	(*_textures)[14].set( 0.3333333, 0.75 ); //top left front
	(*_textures)[15].set( 0.0, 0.75 ); //top left back

	//top face
	(*_textures)[16].set( 0.3333333, 0.75 ); //top left front
	(*_textures)[17].set( 0.6666666, 0.75 ); //top right front
	(*_textures)[18].set( 0.6666666, 1.0 ); //top right back
	(*_textures)[19].set( 0.3333333, 1.0 ); //top left back

	//bottom face
	(*_textures)[20].set( 0.3333333, 0.25 ); //bottom left back
	(*_textures)[21].set( 0.6666666, 0.25 ); //bottom right back
	(*_textures)[22].set( 0.6666666, 0.5 ); //bottom right front
	(*_textures)[23].set( 0.3333333, 0.5 ); //bottom left front

}

void CubeShape::setTextureImage(std::string tex_name)
{

	
	osg::StateSet* state = getOrCreateStateSet();
	osg::Texture2D* tex = new osg::Texture2D;
	osg::Image* image = new osg::Image;
    	tex->setDataVariance(osg::Object::DYNAMIC);

	//Whether to load an image or not
	if(tex_name.empty())
	{
		_texture_name = "";
		tex->setImage(image);
		state->setTextureAttributeAndModes(0, tex, osg::StateAttribute::OFF);
	}
	else
	{
		std::string file_path = cvr::ConfigManager::getEntry("dir", "Plugin.Mugic.Texture", "");
		_texture_name = file_path + tex_name;
		image = osgDB::readImageFile(_texture_name);

    		//testing
    		if(!image)
		{
			std::cout << "Image does not exist." << std::endl;
			_texture_name = "";
			state->setTextureAttributeAndModes(0, tex, osg::StateAttribute::OFF);
			return;
		}

		tex->setImage(image);
    		state->setTextureAttributeAndModes(0, tex, osg::StateAttribute::ON);
	}

}

/* update Cube with passed command */
void CubeShape::update(std::string command)
{
	
	OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
	_dirty = true;

	//check for changed values
	addParameter(command, "x");
	addParameter(command, "y");
	addParameter(command, "z");

	addParameter(command, "width");
	addParameter(command, "height");
	addParameter(command, "depth");
	
	addParameter(command, "r1");
	addParameter(command, "g1");
	addParameter(command, "b1");
	addParameter(command, "a1");

	addParameter(command, "r2");
	addParameter(command, "g2");
	addParameter(command, "b2");
	addParameter(command, "a2");

	addParameter(command, "r3");
	addParameter(command, "g3");
	addParameter(command, "b3");
	addParameter(command, "a3");

	addParameter(command, "r4");
	addParameter(command, "g4");
	addParameter(command, "b4");
	addParameter(command, "a4");

	addParameter(command, "r5");
	addParameter(command, "g5");
	addParameter(command, "b5");
	addParameter(command, "a5");

	addParameter(command, "r6");
	addParameter(command, "g6");
	addParameter(command, "b6");
	addParameter(command, "a6");

	addParameter(command, "r7");
	addParameter(command, "g7");
	addParameter(command, "b7");
	addParameter(command, "a7");

	addParameter(command, "r8");
	addParameter(command, "g8");
	addParameter(command, "b8");
	addParameter(command, "a8");

	addParameter(command, "texture");

}

/* update Cube if parameters have changed recently */
void CubeShape::update()
{

	OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
	if(!_dirty)
		return;

	osg::Vec3 p1( (*_vertices)[0] );
	osg::Vec4 c1( (*_colors)[0] );
	osg::Vec4 c2( (*_colors)[1] );
	osg::Vec4 c3( (*_colors)[2] );
	osg::Vec4 c4( (*_colors)[3] );
	osg::Vec4 c5( (*_colors)[4] );
	osg::Vec4 c6( (*_colors)[5] );
	osg::Vec4 c7( (*_colors)[6] );
	osg::Vec4 c8( (*_colors)[7] );

	float width = (*_vertices)[1].x() - (*_vertices)[0].x();
	float height = (*_vertices)[3].z() - (*_vertices)[0].z();
	float depth = (*_vertices)[5].y() - (*_vertices)[0].y();

	std::string tex_name = _texture_name;

	//adjust center position
	p1.x() = p1.x() + (width/2);
	p1.y() = p1.y() + (depth/2);
	p1.z() = p1.z() + (height/2);

	setParameter("x", p1.x());
	setParameter("y", p1.y());
	setParameter("z", p1.z());
	
	setParameter("width", width);
	setParameter("height", height);
	setColor( c1, c2, c3, c4, c5, c6, c7, c8 );
	setParameter("depth", depth);

	setParameter("r1", c1.r());
	setParameter("g1", c1.g());
	setParameter("b1", c1.b());
	setParameter("a1", c1.a());

	setParameter("r2", c2.r());
	setParameter("g2", c2.g());
	setParameter("b2", c2.b());
	setParameter("a2", c2.a());

	setParameter("r3", c3.r());
	setParameter("g3", c3.g());
	setParameter("b3", c3.b());
	setParameter("a3", c3.a());

	setParameter("r4", c4.r());
	setParameter("g4", c4.g());
	setParameter("b4", c4.b());
	setParameter("a4", c4.a());

	setParameter("r5", c5.r());
	setParameter("g5", c5.g());
	setParameter("b5", c5.b());
	setParameter("a5", c5.a());

	setParameter("r6", c6.r());
	setParameter("g6", c6.g());
	setParameter("b6", c6.b());
	setParameter("a6", c6.a());

	setParameter("r7", c7.r());
	setParameter("g7", c7.g());
	setParameter("b7", c7.b());
	setParameter("a7", c7.a());

	setParameter("r8", c8.r());
	setParameter("g8", c8.g());
	setParameter("b8", c8.b());
	setParameter("a8", c8.a());

	setParameter("texture", tex_name);

	setPosition( p1, width, height, depth );
	//std::cout << width << " " << height << " " << depth << "\n";
	setColor( c1, c2, c3, c4, c5, c6, c7, c8 );
	setTextureImage(tex_name);
	_vertices -> dirty();
	_colors -> dirty();
	_textures -> dirty();
	dirtyBound();

	//reset flag
	_dirty = false;

}

