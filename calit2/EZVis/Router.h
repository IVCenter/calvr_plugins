#ifndef _ROUTER_H
#define _ROUTER_H

#include <string>
#include <vector>
#include <map>
#include <list>
#include <queue>
#include <OpenThreads/Thread>
#include <OpenThreads/Mutex>
#include <iostream>

#include "zhelpers.h"

#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <time.h> 
#include <sys/fcntl.h>

using namespace std;
using namespace OpenThreads;

class Router : public OpenThreads::Thread
{
private:
	bool _mkill;
	virtual void run();
	OpenThreads::Mutex _mutex;
	size_t _buffSize;
    char* _buffer;
	struct sockaddr _client_addr; 
    socklen_t _addr_len;
	void *_context;
	void *_publisher;
	int _listenfd;
    std::string _killMsg;
    int _port;

protected:
	size_t readLine(int sockd, char *bptr, size_t maxlen); 
	Router() {};

public:
	Router(int port);
	~Router();

};
#endif
