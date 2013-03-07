#include "ModelShape.h"

#include <osg/Node>
#include <osg/Geometry>
#include <osg/Material>
#include <osg/Texture2D>
#include <osgDB/ReadFile>
#include <cvrConfig/ConfigManager.h>

#include <string>
#include <vector>
#include <iostream>

ModelShape::ModelShape(std::string command, std::string name)
{

	_type = SimpleShape::MODEL;
	
	BasicShape::setName(name);

	//extract file name from command
	std::string substring, value, searchParam;
	searchParam.append("file").append("=");
	size_t found = command.find(searchParam);
	if(found != std::string::npos)
	{
		// extract value
		substring = command.substr(found + searchParam.size());
		std::stringstream ss(substring);
		ss >> value;
	}

	//read in the model and return a node
	setModel(value);
	setShaders("", "");
	update(command);

}

ModelShape::~ModelShape()
{
}

void ModelShape::setModel(std::string file)
{
	
	std::string file_path = cvr::ConfigManager::getEntry("dir", "Plugin.Mugic.Model", "");
	std::string def = "pawn.wrl";

	//Check whether file name given
	if(file.empty())
	{
		//give default?
		std::cout << "No model file specified; reading in default model" << std::endl;
		_modelNode = osgDB::readNodeFile(file_path + def);
	}
	else
	{
		_modelNode = osgDB::readNodeFile(file_path + file);
		
    		//testing
    		if(_modelNode == NULL)
		{
			std::cout << "Could not load model. Either does not exist or is incorrect file name." << std::endl;
			std::cout << "reading in default model..." << std::endl;
			_modelNode = osgDB::readNodeFile(file_path + def);

			return;
		}
	}

		//set node state
		osg::StateSet* state = new osg::StateSet();
		state->setMode(GL_BLEND, osg::StateAttribute::ON);
		osg::Material* mat = new osg::Material();
		mat->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
		state->setAttributeAndModes(mat, osg::StateAttribute::ON);
		_modelNode->setStateSet(state);
		
}

osg::Node* ModelShape::getModelNode()
{
	return _modelNode;
}

osg::MatrixTransform* ModelShape::getMatrixParent()
{
	return _modelNode->osg::Node::getParent(0)->asTransform()->asMatrixTransform();
}

void ModelShape::setShaders(std::string vert_file, std::string frag_file)
{
	std::cout << "inside setShader call" << std::endl;

	if(vert_file.compare(_vertex_shader) == 0 && frag_file.compare(_fragment_shader) == 0)
		return;

	osg::StateSet* state = _modelNode->getOrCreateStateSet();
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
			std::cout << "adding vertex shader" << std::endl;
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

	state->setAttributeAndModes(prog, osg::StateAttribute::ON | osg::StateAttribute::OVERRIDE);

}

//only update shaders
void ModelShape::update(std::string command)
{
	
	OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
	_dirty = true;

	//check for changed values
	addParameter(command, "vertex");
	addParameter(command, "fragment");

}

void ModelShape::update()
{

	OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
	if(!_dirty)
		return;

	std::string vert_name = _vertex_shader;
	std::string frag_name = _fragment_shader;

	setParameter("vertex", vert_name);
	setParameter("fragment", frag_name);

	setShaders(vert_name, frag_name);
	dirtyBound();

	//reset flag
	_dirty = false; 

}
