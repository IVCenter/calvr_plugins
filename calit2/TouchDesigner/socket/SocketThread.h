#ifndef _SOCKETTHREAD_H
#define _SOCKETTHREAD_H

#include <string>
#include <vector>
#include <map>
#include <list>
#include <OpenThreads/Thread>
#include <OpenThreads/Mutex>
#include <strings.h>
#include <osg/MatrixTransform>

#include <osgDB/ReadFile>
#include <osgDB/WriteFile>
#include <osgDB/ReaderWriter>
#include <osgDB/Registry>

#include <sstream>
#include <fstream>
#include <iostream>


#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include <cvrUtil/CVRSocket.h>

#include "../util/ShapeHelper.h"

using namespace osg;
using namespace osgDB;

using namespace std;
using namespace OpenThreads;

class SocketThread : public OpenThreads::Thread
{
	private:
		bool _mkill;
		virtual void run();
		string _serverName;
		list< string > _serializedScenes;
		OpenThreads::Mutex _mutex;
		ShapeHelper * sh;
		Geode * geoWorker;
		string sceneData;

	protected:
		SocketThread();
		
		struct sockaddr_in _serverAddr;
        	struct sockaddr_in _clientAddr;
        
        	string _port;
        	int _sockID; ///< socket descriptor
        	socklen_t _addrLen;
        	char* readSocket();

	public:
		SocketThread( string& server);
		~SocketThread();
		string getSerializedScene(); 
		Geode * getTestNode();
		ShapeHelper * getTestSH();
};
#endif
