#ifndef _COMMANDPARSER_H
#define _COMMANDPARSER_H

#include <OpenThreads/Thread>
#include <OpenThreads/Mutex>
#include <OpenThreads/Block>
#include <osg/Node>
#include <osg/Geode>

#include "ThreadQueue.h"
//#include "Variables.h"
#include "shapes/BasicShape.h"
#include "shapes/Factory.h"


class CommandParser : public OpenThreads::Thread
{
    protected:
        bool _mkill;
	    OpenThreads::Mutex _mutex;
		OpenThreads::Condition * _condition;
        virtual void run();

        ThreadQueue<std::string> * _queue;
        osg::Group* _root;

        // shape definitions
        std::map<std::string, ShapeFactory* > _shapeDefinitions;

        // map for object creation
        std::map<std::string, BasicShape* > _shapes;

        // parse incoming command
        void parseCommand(std::string command);
        std::string getParameter(std::string command, std::string param);
        void remove(std::string elementName);

    public:
    	CommandParser(ThreadQueue<std::string>* queue, osg::Group* root);
	    ~CommandParser();
};
#endif
