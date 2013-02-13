#include "RectangleShape.h"

#include <osg/Geometry>
#include <osg/Material>
#include <osg/Texture2D>
#include <osgDB/ReadFile>
#include <cvrConfig/ConfigManager.h>

#include <string>
#include <vector>
#include <iostream>

RectangleShape::RectangleShape(std::string command, std::string name) 
{
    _type = SimpleShape::RECTANGLE;

    BasicShape::setName(name);
    
    _vertices = new osg::Vec3Array(4);
    _colors = new osg::Vec4Array(4);
    _textures = new osg::Vec2Array(4);
    
    setPosition(osg::Vec3(0.0, 0.0, 0.0), 1.0, 1.0);
    setColor(osg::Vec4(1.0, 1.0, 1.0, 1.0));
    setTextureCoords(osg::Vec2(0.0, 0.0), osg::Vec2(1.0, 0.0), osg::Vec2(1.0, 1.0), osg::Vec2(0.0, 1.0));
    update(command);
    
    setVertexArray(_vertices); 
    setColorArray(_colors);
    setTexCoordArray(0, _textures); 
    setColorBinding(osg::Geometry::BIND_OVERALL);
    addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUADS,0,4));

    osg::StateSet* state = getOrCreateStateSet();
    state->setMode(GL_BLEND, osg::StateAttribute::ON);
    osg::Material* mat = new osg::Material();
    mat->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
    state->setAttributeAndModes(mat, osg::StateAttribute::ON);

    //additional texture setup
    setTextureImage("");

}

RectangleShape::~RectangleShape()
{
}

void RectangleShape::setPosition(osg::Vec3 p, float width, float height)
{
        (*_vertices)[0].set(p[0] - (width/2), p[1], p[2] - (height/2));
	(*_vertices)[1].set(p[0] + (width/2), p[1], p[2] - (height/2));
	(*_vertices)[2].set(p[0] + (width/2), p[1], p[2] + (height/2));
	(*_vertices)[3].set(p[0] - (width/2), p[1], p[2] + (height/2));
}

void RectangleShape::setColor(osg::Vec4 c0)
{
    (*_colors)[0].set(c0[0], c0[1], c0[2], c0[3]);
    
    if(c0[3] != 1.0)
        getOrCreateStateSet()->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
    else
        getOrCreateStateSet()->setRenderingHint(osg::StateSet::DEFAULT_BIN);

}

void RectangleShape::setTextureCoords(osg::Vec2 t1, osg::Vec2 t2, osg::Vec2 t3, osg::Vec2 t4)
{

	//set texture coordinates
	(*_textures)[0].set(t1[0], t1[1]);
	(*_textures)[1].set(t2[0], t2[1]);
	(*_textures)[2].set(t3[0], t3[1]);
	(*_textures)[3].set(t4[0], t4[1]);

}

void RectangleShape::setTextureImage(std::string tex_name)
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

void RectangleShape::update(std::string command)
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
    _dirty = true;

    // check for changed values
    addParameter(command, "x");
    addParameter(command, "y");
    addParameter(command, "z");
    addParameter(command, "width");
    addParameter(command, "height");
    addParameter(command, "r");
    addParameter(command, "g");
    addParameter(command, "b");
    addParameter(command, "a");

    addParameter(command, "texture");    
    addParameter(command, "t1s");
    addParameter(command, "t1t");
    addParameter(command, "t2s");
    addParameter(command, "t2t");
    addParameter(command, "t3s");
    addParameter(command, "t3t");
    addParameter(command, "t4s");
    addParameter(command, "t4t");
}

void RectangleShape::update()
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
    if( !_dirty )
        return;

    osg::Vec3 p1((*_vertices)[0]);
    osg::Vec4 c1((*_colors)[0]);

    std::string tex_name = _texture_name;
    osg::Vec2 t1((*_textures)[0]);
    osg::Vec2 t2((*_textures)[1]);
    osg::Vec2 t3((*_textures)[2]);
    osg::Vec2 t4((*_textures)[3]);

    float width = (*_vertices)[1].x() - (*_vertices)[0].x();
    float height = (*_vertices)[2].z() - (*_vertices)[1].z();

    //adjust center point
    p1.x() = p1.x() + (width/2);
    p1.z() = p1.z() + (height/2);

    setParameter("x", p1.x()); 
    setParameter("y", p1.y()); 
    setParameter("z", p1.z()); 
    setParameter("width", width); 
    setParameter("height", height); 
    setParameter("r", c1.r()); 
    setParameter("g", c1.g()); 
    setParameter("b", c1.b()); 
    setParameter("a", c1.a()); 

    setParameter("texture", tex_name);
    setParameter("t1s", t1[0]);
    setParameter("t1t", t1[1]);
    setParameter("t2s", t2[0]);
    setParameter("t2t", t2[1]);
    setParameter("t3s", t3[0]);
    setParameter("t3t", t3[1]);
    setParameter("t4s", t4[0]);
    setParameter("t4t", t4[1]);

    setPosition(p1, width, height);
    setColor(c1);
    setTextureCoords(t1, t2, t3, t4);
    setTextureImage(tex_name);
    _vertices->dirty();
    _colors->dirty();
    _textures->dirty();
    dirtyBound();

    // reset flag
    _dirty = false;
}

