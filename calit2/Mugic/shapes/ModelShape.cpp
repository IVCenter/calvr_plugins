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
	//update(command);

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
		_model_name = def;
		_modelNode = osgDB::readNodeFile(file_path + def);
	}
	else
	{
		_model_name = file;
		_modelNode = osgDB::readNodeFile(file_path + file);
		
    		//testing
    		if(_modelNode == NULL)
		{
			std::cout << "Could not load model. Either does not exist or is incorrect file name." << std::endl;
			std::cout << "reading in default model..." << std::endl;
			_model_name = def;
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

//for now, update does nothing, as there's nothing to update
void ModelShape::update(std::string command)
{
	
	/*OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
	_dirty = true;

	//check for changed values
	addParameter(command, "file"); */

}

void ModelShape::update()
{

	/*OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
	if(!_dirty)
		return;

	std::string modelName = _model_name;

	setParameter("file", modelName);

	setModel(modelName);
	dirtyBound();

	//reset flag
	_dirty = false; */

}
