#include "CircleShape.h"

#include <osg/Geometry>
#include <osg/Material>
#include <osg/Texture2D>
#include <osgDB/ReadFile>
#include <cvrConfig/ConfigManager.h>

#include <string>
#include <vector>
#include <iostream>

CircleShape::CircleShape(std::string command, std::string name) 
{
    _type = SimpleShape::CIRCLE;

    BasicShape::setName(name);
    _numFaces = 20;
    
    _vertices = new osg::Vec3Array(_numFaces + 2);
    _colors = new osg::Vec4Array(_numFaces + 2);
    _textures = new osg::Vec2Array (_numFaces + 2);
    
    setPosition(osg::Vec3(0.0, 0.0, 0.0), 1.0);
    setColor(osg::Vec4(1.0, 1.0, 1.0, 1.0),osg::Vec4(1.0, 1.0, 1.0, 1.0));
    setTextureCoords(osg::Vec2(0.5, 0.5), 0.5);
    update(command);
    
    setVertexArray(_vertices); 
    setColorArray(_colors);
    setTexCoordArray(0, _textures);
    setColorBinding(osg::Geometry::BIND_PER_VERTEX);
    addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::TRIANGLE_FAN,0,_numFaces + 2));

    osg::StateSet* state = getOrCreateStateSet();
    state->setMode(GL_BLEND, osg::StateAttribute::ON);
    state->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    //osg::Material* mat = new osg::Material();
    //mat->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
    //state->setAttributeAndModes(mat, osg::StateAttribute::ON);

    setTextureImage("");
}

CircleShape::~CircleShape()
{
}

void CircleShape::setPosition(osg::Vec3 p, float radius)
{
    // first point center
    (*_vertices)[0].set(p[0], p[1], p[2]);
    
    // compute exterior points anti-clockwise
    float portion = -osg::PI * 2 / _numFaces;
    osg::Vec3d pos;

    for (int i = 1; i < _numFaces + 2; i++) 
    {
        pos = p;
        pos.x()+= cos(portion * i) * radius;
        pos.z()+= sin(portion * i) * radius;
        (*_vertices)[i].set(pos.x(), pos.y(), pos.z());
    }
}

void CircleShape::setColor(osg::Vec4 c0, osg::Vec4 c1)
{
    (*_colors)[0].set(c0[0], c0[1], c0[2], c0[3]);
    
    for(int i = 1; i < (int)_colors->size(); i++)
    {
        (*_colors)[i].set(c1[0], c1[1], c1[2], c1[3]);
    }

    if( (c0[3] != 1.0) || (c1[3] != 1.0))
        getOrCreateStateSet()->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
    else
        getOrCreateStateSet()->setRenderingHint(osg::StateSet::DEFAULT_BIN);
}

void CircleShape::setTextureCoords(osg::Vec2 center, float radius)
{

	//first texture coordinate is center of circle
	if(center[0] < 1.0 && center[1] < 1.0 )
	{
		if(center[0] > 0 && center[1] > 0)
		{
			(*_textures)[0].set(center[0], center[1]);
			_texRadius = radius;
		}
	}

	//check whether radius goes out of bounds here
	float xposadd, yposadd, xpossub, ypossub;
	xposadd = center[0]+radius;
	yposadd = center[1]+radius;
	xpossub = center[0]-radius;
	ypossub = center[1]-radius;

	if(xposadd > 1.0 || yposadd > 1.0 || xpossub < 0 || ypossub < 0)
		return;

	//then calculate the other coordinates according to given radius
	float portion = -osg::PI * 2 / _numFaces;
	osg::Vec2d pos;

	for(int i = 1; i < _numFaces + 2; i++)
	{
		pos = center;
		pos[0] += cos(portion * i) * radius;
		pos[1] += sin(portion * i) * radius;
		(*_textures)[i].set(pos[0], pos[1]);
	}

}

void CircleShape::setTextureImage(std::string tex_name)
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

void CircleShape::update(std::string command)
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
    _dirty = true;

    if( !command.empty() )
    {
        // check for changed values
        addParameter(command, "x");
        addParameter(command, "y");
        addParameter(command, "z");
        addParameter(command, "r1");
        addParameter(command, "g1");
        addParameter(command, "b1");
        addParameter(command, "a1");
        addParameter(command, "r2");
        addParameter(command, "g2");
        addParameter(command, "b2");
        addParameter(command, "a2");
        addParameter(command, "radius");
	addParameter(command, "texture");
	addParameter(command, "texcenters");
	addParameter(command, "texcentert");
	addParameter(command, "texrad");
    }
}

void CircleShape::update()
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
    if( !_dirty )
        return;

    osg::Vec3 p1((*_vertices)[0]);
    osg::Vec4 c1((*_colors)[0]);
    osg::Vec4 c2((*_colors)[1]);
    float radius = (*_vertices)[1].x() - (*_vertices)[0].x();
    osg::Vec2 texCenter((*_textures)[0]);
    float texRad = _texRadius;
    std::string tex_name = _texture_name;

    setParameter("x", p1.x()); 
    setParameter("y", p1.y()); 
    setParameter("z", p1.z()); 
    setParameter("radius", radius); 
    setParameter("r1", c1.r()); 
    setParameter("g1", c1.g()); 
    setParameter("b1", c1.b()); 
    setParameter("a1", c1.a()); 
    setParameter("r2", c2.r()); 
    setParameter("g2", c2.g()); 
    setParameter("b2", c2.b()); 
    setParameter("a2", c2.a());
    setParameter("texture", tex_name);
    setParameter("texcenters", texCenter[0]);
    setParameter("texcentert", texCenter[1]);
    setParameter("texrad", texRad);

    setPosition(p1, radius);
    setColor(c1, c2);
    setTextureCoords(texCenter, texRad);
    setTextureImage(tex_name);
    _vertices->dirty();
    _colors->dirty();
    _textures->dirty();
    dirtyBound();

    // reset flag
    _dirty = false;
}
