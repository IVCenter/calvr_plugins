#ifndef _COMMANDPARSER_H
#define _COMMANDPARSER_H

#include <OpenThreads/Thread>
#include <OpenThreads/Mutex>
#include <OpenThreads/Block>
#include <osg/Node>
#include <osg/Geode>
#include <osg/MatrixTransform>
#include <osg/Matrixd>

#include "ThreadQueue.h"
#include "MainNode.h"
//#include "Variables.h"
#include "shapes/BasicShape.h"
#include "shapes/ShapeFactory.h"
#include "shapes/Factory.h"


class CommandParser : public OpenThreads::Thread
{
    protected:
        bool _mkill;
	    OpenThreads::Mutex _mutex;
		OpenThreads::Condition * _condition;
        virtual void run();

        ThreadQueue<std::string> * _queue;
        //osg::Group* _root;
        MainNode* _root;

        // shape definitions
        //ShapeRegistry<ShapeFactory*, std::string > _shapeDefinitions;
        std::map<std::string, ShapeFactory* > _shapeDefinitions;

        // map for object creation
        std::map<std::string, BasicShape* > _shapes;

        // parse incoming command
        void parseCommand(std::string command);
        std::string getParameter(std::string command, std::string param);
        void remove(std::string elementName);
        void removeAll();

	//MatrixTransform manipulation commands
	void rotate(std::string elementName, std::string command);
	void scale(std::string elementName, std::string command);
	void translate(std::string elementName, std::string command);

    public:
    	CommandParser(ThreadQueue<std::string>* queue, MainNode* root);
	    ~CommandParser();
};
#endif
