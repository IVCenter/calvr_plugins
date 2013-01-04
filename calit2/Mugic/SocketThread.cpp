#include "SocketThread.h"

#include <iostream>
#include <sstream>
#include <algorithm> 
#include <functional> 
#include <cctype>
#include <locale>
#include <sys/time.h>
#include <unistd.h>
#include <string>
#include <zmq.h>

const std::string newLine("\n");

static inline std::string &ltrim(std::string &s) 
{
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
    return s;
}

static inline std::string &rtrim(std::string &s)
{
    s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
    return s;
}

static inline std::string &trim(std::string &s) 
{
    return ltrim(rtrim(s));
}


// socket thread constructor
SocketThread::SocketThread(ThreadQueue<std::string>* commands, std::string address) : _commands(commands)
{
	_mkill = false;

    std::string addressport("tcp://");
    addressport.append(address);
    addressport.append(":5560");
        
    _context = zmq_ctx_new();

    // outside connection
    _subscriber = zmq_socket( _context, ZMQ_SUB);
    zmq_connect( _subscriber, addressport.c_str());
    zmq_setsockopt (_subscriber, ZMQ_SUBSCRIBE, NULL, NULL);

	start(); //starts the thread
}



void SocketThread::run() 
{
	while ( ! _mkill ) 
	{
        char *commands = s_recv(_subscriber);

        if( commands != NULL )
        {
            // process string
            seperateCommands(commands); 
        }
        free(commands);
	}
}

// parse commands and update nodequeue
void SocketThread::seperateCommands(char* commands)
{
   std::string element;
   std::stringstream ss(commands);
   while( getline(ss, element, ';') )
   {
        if(element.compare(newLine) != 0 )
        {
            _commands->add(ltrim(element));
        }
   }
}

SocketThread::~SocketThread() 
{
	_mkill = true;

    // clean up connection
    zmq_close(_subscriber);
    zmq_ctx_destroy (_context);

	join();
}
