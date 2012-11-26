#include "CommandParser.h"

#include "shapes/PointShape.h"
#include "shapes/TriangleShape.h"
#include "shapes/QuadShape.h"
#include "shapes/CircleShape.h"
#include "shapes/LineShape.h"
#include "shapes/RectangleShape.h"

#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>

// command thread constructor
CommandParser::CommandParser(ThreadQueue<std::string>* queue, osg::Group* root) : _queue(queue), _root(root)
{
	_mkill = false;

    // create shape definitions
    _shapeDefinitions["point"] = new Factory<PointShape>();
    _shapeDefinitions["triangle"] = new Factory<TriangleShape>();
    _shapeDefinitions["quad"] = new Factory<QuadShape>();
    _shapeDefinitions["circle"] = new Factory<CircleShape>();
    _shapeDefinitions["line"] = new Factory<LineShape>();
    _shapeDefinitions["rectangle"] = new Factory<RectangleShape>();
        
	start(); //starts the thread
}

void CommandParser::run() 
{
	while ( ! _mkill ) 
	{
        std::string command;
        if( _queue->get(command) )
            parseCommand(command);
        //else
        //    sleep(1);
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

   // smallest number of params needed is 1
   if(tokens.size() < 1)
       return;

   // check initial command
   std::transform(tokens[0].begin(), tokens[0].end(), tokens[0].begin(), ::tolower);

   std::string commandType = tokens[0]; 
   std::string elementName;
   if( tokens.size() > 1 )
       elementName.append(tokens[1]);
  
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
/*
   else if( commandType.compare("var") == 0 && !elementName.empty()) // check for variable
   {
        Variables::getInstance()->add(elementName, getParameter(command, "value"));
   }
*/
   else // new object defined need to create
   {
       // create a new object if it doesnt exist or override old one
       std::map<std::string, BasicShape* >::iterator it = _shapes.find(elementName);
       if( it != _shapes.end() )
       {
            osg::Geode* geode = NULL;
            // get geode drawable is attached too
            while( (geode = it->second->getParent(0)->asGeode()) != NULL)
            {
                osg::Group* parent = NULL;

                // access all the parents of the attached geode
                while( (parent = geode->getParent(0)->asGroup()) != NULL)
                {
                    parent->removeChild(geode);
                }
            }
            
            // remove old shape from map and scenegraph
            _shapes.erase(it);
       }

       // make sure shape exists in table to create
       if( _shapeDefinitions.find(commandType) != _shapeDefinitions.end() )
       {
            // object created
            BasicShape* newShape = _shapeDefinitions[commandType]->createInstance(command, elementName);

            // get name from object created and add to table 
            _shapes[newShape->getName()] = newShape;

            // create a new geode to add to scenegraph
            osg::ref_ptr<osg::Geode> geode = new osg::Geode();
            geode->setDataVariance(osg::Object::DYNAMIC);
            geode->addDrawable(newShape);
            _root->addChild(geode.get());
       }
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
	join();
}
