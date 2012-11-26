#ifndef _SOCKETTHREAD_H
#define _SOCKETTHREAD_H

#include <string>
#include <vector>
#include <map>
#include <list>
#include <queue>
#include <OpenThreads/Thread>
#include <OpenThreads/Mutex>
#include <osg/MatrixTransform>

#include <osgDB/ReadFile>
#include <osgDB/WriteFile>
#include <osgDB/ReaderWriter>
#include <osgDB/Registry>

#include "ThreadQueue.h"
#include "zhelpers.h"

using namespace osg;
using namespace std;
using namespace OpenThreads;

class SocketThread : public OpenThreads::Thread
{
private:
	bool _mkill;
	virtual void run();
	OpenThreads::Mutex _mutex;

    ThreadQueue<std::string>* _commands;
    void * _context;
    void * _subscriber;

protected:
	SocketThread();
    void seperateCommands(char *);

public:
	SocketThread(ThreadQueue<std::string>* commands);
	~SocketThread();

};
#endif
