#include "CommandParser.h"

#include "shapes/PointShape.h"
#include "shapes/TriangleShape.h"
#include "shapes/QuadShape.h"
#include "shapes/CircleShape.h"
#include "shapes/LineShape.h"
#include "shapes/RectangleShape.h"
#include "shapes/TextShape.h"
#include "shapes/ScalableLineShape.h"

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
    _shapeDefinitions["scalableline"] = new Factory<ScalableLineShape>();
    _shapeDefinitions["rectangle"] = new Factory<RectangleShape>();
    _shapeDefinitions["text"] = new Factory<TextShape>();

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
            osg::Geode* geode = new osg::Geode();
            geode->setDataVariance(osg::Object::DYNAMIC);
            geode->addDrawable(newShape->asDrawable());
            _root->addElement(geode);

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

            osg::Geode* geode = NULL;
            // get geode drawable is attached too
            if( (geode = it->second->getParent()) != NULL)
            {
                _root->removeElement(geode);
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
            osg::Geode* geode = NULL;
            // get geode drawable is attached too
            if( (geode = it->second->getParent()) != NULL)
            {
                _root->removeElement(geode);
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
