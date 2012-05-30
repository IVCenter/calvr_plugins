#include "SocketThread.h"

#include <iostream>
#include <sstream>
#include <sys/time.h>
#include <unistd.h>
#include <string>

using namespace osg;
using namespace osgDB;

using namespace std;
using namespace OpenThreads;


#define PORT 19997
#define PACKETLEN 500  // # bytes in a packet

char recvData[PACKETLEN];
fd_set readfds;



// socket thread constructor
SocketThread::SocketThread(string & serverName) : _serverName(serverName)
{
   _mkill = false;

   // create connection to server
   	if ((_sockID = socket(AF_INET, SOCK_DGRAM, 0)) == -1)
	{
		cerr<<"Socket Error!" << endl;
	} 
	else
	{
		_serverAddr.sin_family = AF_INET;
		_serverAddr.sin_port = htons(PORT);
		_serverAddr.sin_addr.s_addr = INADDR_ANY;
		bzero(&(_serverAddr.sin_zero), 8);
	}
	
	if (bind(_sockID, (struct sockaddr *)&_serverAddr, sizeof(struct sockaddr)) == -1)
	{
		cerr << "Bind Error!" << endl;
	}
	else 
	{
		_addrLen = sizeof(struct sockaddr);
		cerr<<"Server waiting for client on port: "<<PORT<<endl;
	
		FD_ZERO(&readfds);
		FD_SET(_sockID, &readfds);
	}
	
	// a bunch of stuff needed for threads?
	sh = new ShapeHelper();
//	_mu
	
   	start(); //starts the thread
   	
   	cerr << "SOCKET THREAD INITIALIZED" << endl;
}



void SocketThread::run() 
{

	ReaderWriter * readerwriter =  Registry::instance()->getReaderWriterForExtension("ive");

	//serialize
	stringstream ss;
	

	//printf("%s\n", ss.str().c_str());  // send this via master slave

   	 while ( ! _mkill ) 
    	{
	// check server for info
	
	
	
	sh->processData(readSocket());
	
	//if (sh->processedAll)
	//{
		_mutex.lock();
		// GOTTA TURN THIS TO BINARY
		geoWorker = sh->getGeode();
		readerwriter->writeNode(*geoWorker, ss);
		_serializedScenes.push_back(ss.str());
		
		//cerr
		//cerr << "pushing data\t" << endl;
		//cerr << ss.str() << endl;
		_mutex.unlock();
	//	sh->processedAll=false;
	//}
		
	
	// send result to parser 
	// place mutex around adding ive string to vector e.g _mutex.lock() and _mutex.unlock();
    }
}

string SocketThread::getSerializedScene(void)
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);

    string ret = "";

    if (!_serializedScenes.empty())
    {
        ret = _serializedScenes.front();
        _serializedScenes.pop_front();
    }
    
    cerr << "size " << _serializedScenes.size() << endl;
    
    return ret;
}


SocketThread::~SocketThread() 
{
      _mkill = true;
      join();
}

char* SocketThread::readSocket()
{
	int bytes_read;
	int rs;

	rs = select(_sockID + 1, &readfds, 0, 0, 0);
	if(rs > 0){
		bytes_read = recvfrom(_sockID, recvData, PACKETLEN , 0, (struct sockaddr *)&_clientAddr, &_addrLen);
		if(bytes_read <= 0){
			cerr<<"No data read."<<endl;
		}
		// Prepares data for processing...
		recvData[bytes_read]='\0';	
		//cerr << &recvData[0] << endl;	
		return &recvData[0];
	}
}

Geode * SocketThread::getTestNode()
{
	return geoWorker;
}
