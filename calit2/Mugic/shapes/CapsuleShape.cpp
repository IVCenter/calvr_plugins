/* Capsule Shape for Mugic */

#include "CapsuleShape.h"

#include <osg/ShapeDrawable>
#include <osg/Material>
#include <osg/Program>
#include <osg/Shader>
#include <cvrConfig/ConfigManager.h>

#include <string>
#include <vector>
#include <iostream>

/*Constructor for Capsule*/
CapsuleShape::CapsuleShape(std::string command, std::string name)
{

	_type = SimpleShape::CAPSULE;
	BasicShape::setName(name);

	//create capsule and add to geode as shape drawable
	_capsule = new osg::Capsule();
	setShape(_capsule);

	setPosition(osg::Vec3(0.0, 0.0, 0.0), 1.0, 1.0);
	setShapeColor(osg::Vec4(1.0, 1.0, 1.0, 1.0));
	update(command);

	osg::StateSet* state = getOrCreateStateSet();
	state -> setMode(GL_BLEND, osg::StateAttribute::ON);

	osg::Material* mat = new osg::Material();
	mat -> setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
	state ->  setAttributeAndModes(mat, osg::StateAttribute::ON);

	setShaders("", "");

}

/*Destructor*/
CapsuleShape::~CapsuleShape()
{
}

/*set Position of Capsule - radius and height*/
void CapsuleShape::setPosition(osg::Vec3 position, float radius, float height)
{

	_capsule->setRadius(radius);
	_capsule->setHeight(height);
	_capsule->setCenter(position);

	_radius = radius;
	_height = height;
	_center = position;

	//for update purposes
	dirtyDisplayList();

}

/*set Color of Capsule*/
void CapsuleShape::setShapeColor(osg::Vec4 color)
{

	setColor(color);
	_color = color;

	if(color[3] != 1.0)
		getOrCreateStateSet()->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
	else
		getOrCreateStateSet()->setRenderingHint(osg::StateSet::DEFAULT_BIN);

}

/*set Shaders for Capsule*/
void CapsuleShape::setShaders(std::string vert_file, std::string frag_file)
{

	if(vert_file.compare(_vertex_shader) == 0 && frag_file.compare(_fragment_shader) == 0)
		return;

	osg::StateSet* state = getOrCreateStateSet();
	osg::Program* prog = new osg::Program();
	osg::Shader* vert = new osg::Shader(osg::Shader::VERTEX);
	osg::Shader* frag = new osg::Shader(osg::Shader::FRAGMENT);

	_vertex_shader = vert_file;
	_fragment_shader = frag_file;

	//try to load shader files
	std::string file_path = cvr::ConfigManager::getEntry("dir", "Plugin.Mugic.Shader", "");
	if(!_vertex_shader.empty())
	{
		
		bool loaded = vert->loadShaderSourceFromFile(file_path + _vertex_shader);
		if(!loaded)
		{
			std::cout << "could not load vertex shader." << std::endl;
			_vertex_shader = "";
		}
		else
		{
			prog->addShader(vert);
		}

	}

	if(!_fragment_shader.empty())
	{

		bool loaded = frag->loadShaderSourceFromFile(file_path + _fragment_shader);
		if(!loaded)
		{
			std::cout << "could not load fragment shader." << std::endl;
			_fragment_shader = "";
		}
		else
		{
			prog->addShader(frag);
		}

	}

	state->setAttributeAndModes(prog, osg::StateAttribute::ON);

}

/*update capsule with passed command*/
void CapsuleShape::update(std::string command)
{

	OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
	_dirty = true;

	//check for changed values
	addParameter(command, "radius");
	addParameter(command, "height");
	
	addParameter(command, "x");
	addParameter(command, "y");
	addParameter(command, "z");

	addParameter(command, "r1");
	addParameter(command, "g1");
	addParameter(command, "b1");
	addParameter(command, "a1");

	addParameter(command, "vertex");
	addParameter(command, "fragment");

}


/*update*/
void CapsuleShape::update()
{

	OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
	if(!_dirty)
		return;

	float radius = _radius;
	float height = _height;
	osg::Vec3 center = _center;
	osg::Vec4 color = _color;
	std::string vert = _vertex_shader;
	std::string frag = _fragment_shader;

	setParameter("radius", radius);
	setParameter("height", height);

	setParameter("x", center.x());
	setParameter("y", center.y());
	setParameter("z", center.z());

	setParameter("r1", color[0]);
	setParameter("g1", color[1]);
	setParameter("b1", color[2]);
	setParameter("a1", color[3]);
	
	setParameter("vertex", vert);
	setParameter("fragment", frag);

	setPosition(center, radius, height);
	setShapeColor(color);
	setShaders(vert, frag);
	dirtyBound();

	//reset flag
	_dirty = false;

}

