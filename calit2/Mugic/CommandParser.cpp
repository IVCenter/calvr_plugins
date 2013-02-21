#include "CommandParser.h"

#include "shapes/PointShape.h"
#include "shapes/TriangleShape.h"
#include "shapes/QuadShape.h"
#include "shapes/CircleShape.h"
#include "shapes/LineShape.h"
#include "shapes/RectangleShape.h"
#include "shapes/TextShape.h"
#include "shapes/CubeShape.h"
#include "shapes/ModelShape.h"

#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>

// command thread constructor
CommandParser::CommandParser(ThreadQueue<std::string>* queue, MainNode* root) : _queue(queue), _root(root)
{
	_mkill = false;

    // create shape definitions
    _shapeDefinitions["point"] = new Factory<PointShape>();
    _shapeDefinitions["triangle"] = new Factory<TriangleShape>();
    _shapeDefinitions["quad"] = new Factory<QuadShape>();
    _shapeDefinitions["circle"] = new Factory<CircleShape>();
    _shapeDefinitions["line"] = new Factory<LineShape>();
    _shapeDefinitions["rectangle"] = new Factory<RectangleShape>();
    _shapeDefinitions["text"] = new Factory<TextShape>();
    _shapeDefinitions["cube"] = new Factory<CubeShape>();
    _shapeDefinitions["model"] = new Factory<ModelShape>();

    // init mutex 
    _mutex.lock();

    _condition = _queue->getCondition();
        
	start(); //starts the thread
}

void CommandParser::run() 
{
	while ( true ) 
	{
		// wait for queue addition signal
        _condition->wait(&_mutex);
        if( _mkill )
            return;

        std::string command;
        while( _queue->get(command) )
        {
            parseCommand(command);
        }
	}
}

// parse commands and update nodequeue
void CommandParser::parseCommand(std::string command)
{
   command.erase(std::remove(command.begin(), command.end(), '\n'), command.end());
   
   std::vector<std::string> tokens;
   
   std::stringstream ss;
   ss << command;
   std::string value;

   while( getline(ss, value, ' '))
   {
        tokens.push_back(value);
   }

   // smallest number of params needed is 2
   if(tokens.size() < 2)
       return;

   // check initial command
   std::transform(tokens[0].begin(), tokens[0].end(), tokens[0].begin(), ::tolower);

   std::string commandType = tokens[0]; 
   std::string elementName = tokens[1];
  
   // look for an update command for an existing object 
   if( commandType.compare("update") == 0)
   {
        // extract key and look up shape in map to update
        if( _shapes.find(elementName) != _shapes.end() )
        {
            // update existing shape
            _shapes[elementName]->update(command);
        }
   }

   //command to rotate an object
   else if( commandType.compare("rotate") == 0 )
   {
	//extract key and look up shape in map
	if( _shapes.find(elementName) != _shapes.end() )
	{
		//call rotation
		rotate(elementName, command);
	}
   }

   //command to scale an object
   else if( commandType.compare("scale") == 0 )
   {
	//extract key and look up shape in map
	if( _shapes.find(elementName) != _shapes.end() )
	{
		//call scaling
		scale(elementName, command);
	}
   }

   //command to translate an object
   else if( commandType.compare("translate") == 0 )
   {
	//extract key and look up shape in map
	if( _shapes.find(elementName) != _shapes.end() )
	{
		//call translation
		translate(elementName, command);
	}
   }

   //command to delete all objects
   else if( commandType.compare("delete") == 0)
   {
       if(elementName.compare("all") == 0)
       {
            removeAll();     
       }
       else
       {
            remove(elementName);
       }
   }

   //reading in model files
   else if( commandType.compare("model") == 0)
   {

	//create a new object if it doesn't exist or override old one
	remove(elementName);

	//create basic shape that holds the node
	ModelShape* newShape = new ModelShape(command, elementName);
	
	//take the node within the new shape and add it to scenegraph
	osg::MatrixTransform* matrix = new osg::MatrixTransform();
	matrix->addChild(newShape->getModelNode());
	_root->addElement(matrix);

	//get name from object, add to table
	_shapes[newShape->BasicShape::getName()] = newShape; 
   }

/*
   else if( commandType.compare("var") == 0 && !elementName.empty()) // check for variable
   {
        Variables::getInstance()->add(elementName, getParameter(command, "value"));
   }
*/
   else // new object defined need to create TODO test for valid commands
   {
       // create a new object if it doesnt exist or override old one
       remove(elementName);

       // make sure shape exists in table to create
       if( _shapeDefinitions.find(commandType) != _shapeDefinitions.end() )
       {
            // object created
            BasicShape* newShape = _shapeDefinitions[commandType]->createInstance(command, elementName);

            // create a new geode to add to scenegraph
            //osg::ref_ptr<osg::Geode> geode = new osg::Geode();
            osg::Geode* geode = new osg::Geode();
            geode->setDataVariance(osg::Object::DYNAMIC);
            geode->addDrawable(newShape->asDrawable());
	    //_root->addElement(geode);

	    //add in MatrixTransform for manipulations
	    osg::MatrixTransform* matrix = new osg::MatrixTransform();
	    matrix->addChild(geode);
            _root->addElement(matrix);
            
            // get name from object created and add to table
            _shapes[newShape->getName()] = newShape;
       }
   }
}

void CommandParser::removeAll()
{
       // create a new object if it doesnt exist or override old one
       std::map<std::string, BasicShape* >::iterator it;
       while( _shapes.begin() != _shapes.end() )
       {
            it = _shapes.begin();

            osg::MatrixTransform* mat = NULL;
            // get geode drawable is attached too
            if( (mat = it->second->getMatrixParent()) != NULL)
            {
                _root->removeElement(mat);
            }

            // remove old shape from map and scenegraph
            _shapes.erase(it);
       }
}


void CommandParser::remove(std::string elementName)
{
       // create a new object if it doesnt exist or override old one
       std::map<std::string, BasicShape* >::iterator it = _shapes.find(elementName);
       if( it != _shapes.end() )
       {
            osg::MatrixTransform* mat = NULL;
            // get geode drawable is attached too
            if( (mat = it->second->getMatrixParent()) != NULL)
            {
                _root->removeElement(mat);
            }

            // remove old shape from map and scenegraph
            _shapes.erase(it);
       }
}

std::string CommandParser::getParameter(std::string command, std::string param)
{
    std::string substring, value, searchParam;
    searchParam.append(param).append("=");
    size_t found = command.find(searchParam);
    if(found != std::string::npos)
    {
        // extract value
        substring = command.substr(found + searchParam.size());
        std::stringstream ss(substring);
        ss >> value;
    }
    return value;
}

CommandParser::~CommandParser() 
{
	_mkill = true;
    _condition->signal();

	join();
}

void CommandParser::rotate(std::string elementName, std::string command)
{
	
	std::string value;
	float head, pitch, roll;
	osg::MatrixTransform* matNode;
	BasicShape* geoNode;
	osg::Matrixd* rotationMat;
	osg::Matrixd tempMat;
	osg::Vec3d scaleVec;
	osg::Vec3d transVec;
	float degtorad = osg::PI/180.0;

	//find MatrixTransform parent
	geoNode = _shapes[elementName];
	matNode = geoNode->getMatrixParent();

	//search for pitch
	value = getParameter(command, "pitch");
	if( !value.empty())
	{
		//extract value from parameter
		std::stringstream ss(value);
		ss >> pitch;
	}
	else
		pitch = 0.0;
	
	//search for roll
	value = getParameter(command, "roll");
	if( !value.empty() )
	{
		//extract value from parameter
		std::stringstream ss(value);
		ss >> roll;
	}
	else
		roll = 0.0;

	//search for heading
	value = getParameter(command, "head");
	if( !value.empty() )
	{
		//extract value from parameter
		std::stringstream ss(value);
		ss >> head;
	}
	else
		head = 0.0;

	//create new matrix to set
	rotationMat = new osg::Matrixd();
	rotationMat->makeRotate(pitch*degtorad, osg::Vec3f(1, 0, 0), roll*degtorad, osg::Vec3f(0, 1, 0), head*degtorad, osg::Vec3f(0, 0, 1));
	tempMat = matNode->getMatrix();

	scaleVec = tempMat.getScale();
	transVec = tempMat.getTrans();
	rotationMat->preMultScale(scaleVec);
	rotationMat->postMultTranslate(transVec);
	matNode->setMatrix(*rotationMat);

}

void CommandParser::scale(std::string elementName, std::string command)
{
	
	std::string value;
	float xscale, yscale, zscale;
	osg::MatrixTransform* matNode;
	BasicShape* geoNode;
	osg::Matrixd* scaleMat;
	osg::Matrixd tempMat;
	osg::Quat rotateQuat;
	osg::Vec3d transVec;
	
	//find MatrixTransform parent
	geoNode = _shapes[elementName];
	matNode = geoNode->getMatrixParent();

	//check whether it is uniform scale or not
	value = getParameter(command, "factor");
	if( !value.empty() )
	{
		//extract value from parameter
		std::stringstream ss(value);
		ss >> xscale;
		yscale = xscale;
		zscale = xscale;
	}
	else
	{
		//search for xscale
		value = getParameter(command, "x");
		if( !value.empty() )
		{
			//extract value from parameter
			std::stringstream ss(value);
			ss >> xscale;
		}
		else
			xscale = 1.0;

		//search for yscale
		value = getParameter(command, "y");
		if( !value.empty() )
		{
			//extract value from parameter
			std::stringstream ss(value);
			ss >> yscale;
		}
		else
			yscale = 1.0;

		//search for zscale
		value = getParameter(command, "z");
		if( !value.empty() )
		{
			//extract value from parameter
			std::stringstream ss(value);
			ss >> zscale;
		}
		else
			zscale = 1.0;
	}

	//create new matrix to set
	scaleMat = new osg::Matrixd();
	scaleMat->makeScale(xscale, yscale, zscale);
	tempMat = matNode->getMatrix();
	rotateQuat = tempMat.getRotate();
	transVec = tempMat.getTrans();
	scaleMat->postMultRotate(rotateQuat);
	scaleMat->postMultTranslate(transVec);
	matNode->setMatrix(*scaleMat);

}

void CommandParser::translate(std::string elementName, std::string command)
{

	std::string value;
	float xtrans, ytrans, ztrans;
	osg::MatrixTransform* matNode;
	BasicShape* geoNode;
	osg::Matrixd* transMat;
	osg::Matrixd rotateMat;
	osg::Matrixd scaleMat;
	osg::Matrixd tempMat;
	osg::Quat rotateQuat;
	osg::Vec3d scaleVec;

	//find MatrixTransform parent
	geoNode = _shapes[elementName];
	matNode = geoNode->getMatrixParent();

	//search for xtrans
	value = getParameter(command, "x");
	if( !value.empty() )
	{
		//extract value from parameter
		std::stringstream ss(value);
		ss >> xtrans;
	}
	else
		xtrans = 0.0;

	//search for ytrans
	value = getParameter(command, "y");
	if( !value.empty() )
	{
		//extract value from parameter
		std::stringstream ss(value);
		ss >> ytrans;
	}
	else
		ytrans = 0.0;

	//search for ztrans
	value = getParameter(command, "z");
	if( !value.empty() )
	{
		//extract value from parameter
		std::stringstream ss(value);
		ss >> ztrans;
	}
	else
		ztrans = 0.0;

	//create new matrix to set
	transMat = new osg::Matrixd();
	transMat->makeTranslate(xtrans, ytrans, ztrans);
	tempMat = matNode->getMatrix();
	rotateQuat = tempMat.getRotate();
	scaleVec = tempMat.getScale();
	transMat->preMultRotate(rotateQuat);
	transMat->preMultScale(scaleVec);
	matNode->setMatrix(*transMat);

}



