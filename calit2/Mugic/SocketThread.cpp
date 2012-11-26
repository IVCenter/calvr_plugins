#include "SocketThread.h"

#include <iostream>
#include <sstream>
#include <sys/time.h>
#include <unistd.h>
#include <string>
#include <zmq.h>


// socket thread constructor
SocketThread::SocketThread(ThreadQueue<std::string>* commands) : _commands(commands)
{
	_mkill = false;
        
    _context = zmq_ctx_new();

    // outside connection
    _subscriber = zmq_socket( _context, ZMQ_SUB);
    zmq_connect( _subscriber, "tcp://127.0.0.1:5560");
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

        // no activity
        //s_sleep(1);
	}

    // clean up connection
    zmq_close (_subscriber);
    zmq_ctx_destroy (_context);
}

// parse commands and update nodequeue
void SocketThread::seperateCommands(char* commands)
{
   std::string element;
   std::stringstream ss(commands);
   while( getline(ss, element, ';') )
   {
        _commands->add(element);
   }
}

SocketThread::~SocketThread() 
{
	_mkill = true;
	join();
}
