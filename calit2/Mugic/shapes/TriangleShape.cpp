#include "TriangleShape.h"

#include <osg/Geometry>
#include <osg/Material>
#include <osg/Texture2D>
#include <osgDB/ReadFile>
#include <cvrConfig/ConfigManager.h>

#include <string>
#include <vector>
#include <iostream>

TriangleShape::TriangleShape(std::string command, std::string name) 
{
    _type = SimpleShape::TRIANGLE;

    BasicShape::setName(name);
    
    _vertices = new osg::Vec3Array(3);
    _colors = new osg::Vec4Array(3);
    _textures = new osg::Vec2Array(3);
   
    setPosition(osg::Vec3(-0.5, 0.0, -0.5), osg::Vec3(0.5, 0.0, -0.5), osg::Vec3(0.0, 0.0, 0.5));
    setColor(osg::Vec4(1.0, 1.0, 1.0, 1.0),osg::Vec4(1.0, 1.0, 1.0, 1.0),osg::Vec4(1.0, 1.0, 1.0, 1.0));
    setTextureCoords(osg::Vec2(0.0, 0.0), osg::Vec2(1.0, 0.0), osg::Vec2(0.5, 1.0));
    update(command);

    setVertexArray(_vertices); 
    setColorArray(_colors);
    setTexCoordArray(0, _textures);
    setColorBinding(osg::Geometry::BIND_PER_VERTEX);
    addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::TRIANGLES,0,3));

    osg::StateSet* state = getOrCreateStateSet();
    osg::Material* mat = new osg::Material();
    state->setMode(GL_BLEND, osg::StateAttribute::ON);
    mat->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
    state->setAttributeAndModes(mat, osg::StateAttribute::ON);

    //additional texture setup
    setTextureImage("");
}

TriangleShape::~TriangleShape()
{
}

void TriangleShape::setPosition(osg::Vec3 p0, osg::Vec3 p1, osg::Vec3 p2)
{
    (*_vertices)[0].set(p0[0], p0[1], p0[2]);    
    (*_vertices)[1].set(p1[0], p1[1], p1[2]);    
    (*_vertices)[2].set(p2[0], p2[1], p2[2]);    
}

void TriangleShape::setColor(osg::Vec4 c0, osg::Vec4 c1, osg::Vec4 c2)
{
    (*_colors)[0].set(c0[0], c0[1], c0[2], c0[3]);    
    (*_colors)[1].set(c1[0], c1[1], c1[2], c1[3]);    
    (*_colors)[2].set(c2[0], c2[1], c2[2], c2[3]);

    if( (c0[3] != 1.0) || (c1[3] != 1.0) || (c2[3] != 1.0))
        getOrCreateStateSet()->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
    else
        getOrCreateStateSet()->setRenderingHint(osg::StateSet::DEFAULT_BIN);

}

void TriangleShape::setTextureCoords(osg::Vec2 t1, osg::Vec2 t2, osg::Vec2 t3)
{

	(*_textures)[0].set(t1[0], t1[1]);
	(*_textures)[1].set(t2[0], t2[1]);
	(*_textures)[2].set(t3[0], t3[1]);

}

void TriangleShape::setTextureImage(std::string tex_name)
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

void TriangleShape::update(std::string command)
{
	OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
    _dirty = true;

    // check for changed values
    addParameter(command, "x1");
    addParameter(command, "y1");
    addParameter(command, "z1");
    addParameter(command, "r1");
    addParameter(command, "g1");
    addParameter(command, "b1");
    addParameter(command, "a1");

	addParameter(command, "x2");
    addParameter(command, "y2");
    addParameter(command, "z2");
    addParameter(command, "r2");
    addParameter(command, "g2");
    addParameter(command, "b2");
    addParameter(command, "a2");

	addParameter(command, "x3");
    addParameter(command, "y3");
    addParameter(command, "z3");
    addParameter(command, "r3");
    addParameter(command, "g3");
    addParameter(command, "b3");
    addParameter(command, "a3");

    addParameter(command, "texture");
    addParameter(command, "t1s");
    addParameter(command, "t1t");
    addParameter(command, "t2s");
    addParameter(command, "t2t");
    addParameter(command, "t3s");
    addParameter(command, "t3t");
}

void TriangleShape::update()
{
	OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
    if( !_dirty )
        return;

    osg::Vec3 p1((*_vertices)[0]);
    osg::Vec4 c1((*_colors)[0]);
    osg::Vec2 t1((*_textures)[0]);
    osg::Vec2 t2((*_textures)[1]);
    osg::Vec2 t3((*_textures)[2]);
    std::string tex_name = _texture_name;

    setParameter("x1", p1.x()); 
    setParameter("y1", p1.y()); 
    setParameter("z1", p1.z()); 
    setParameter("r1", c1.r()); 
    setParameter("g1", c1.g()); 
    setParameter("b1", c1.b()); 
    setParameter("a1", c1.a()); 

	osg::Vec3 p2((*_vertices)[1]);
    osg::Vec4 c2((*_colors)[1]);

    setParameter("x2", p2.x()); 
    setParameter("y2", p2.y()); 
    setParameter("z2", p2.z()); 
    setParameter("r2", c2.r()); 
    setParameter("g2", c2.g()); 
    setParameter("b2", c2.b()); 
    setParameter("a2", c2.a()); 

	osg::Vec3 p3((*_vertices)[2]);
    osg::Vec4 c3((*_colors)[2]);

    setParameter("x3", p3.x()); 
    setParameter("y3", p3.y()); 
    setParameter("z3", p3.z()); 
    setParameter("r3", c3.r()); 
    setParameter("g3", c3.g()); 
    setParameter("b3", c3.b()); 
    setParameter("a3", c3.a()); 

    setParameter("texture", tex_name);
    setParameter("t1s", t1[0]);
    setParameter("t1t", t1[1]);
    setParameter("t2s", t2[0]);
    setParameter("t2t", t2[1]);
    setParameter("t3s", t3[0]);
    setParameter("t3t", t3[0]);

    setPosition(p1, p2, p3);
    setColor(c1, c2 ,c3);
    setTextureCoords(t1, t2, t3);
    setTextureImage(tex_name);

    _colors->dirty();
    _vertices->dirty();
    _textures->dirty();
    dirtyBound();
    
	// reset flag
    _dirty = false;
}

