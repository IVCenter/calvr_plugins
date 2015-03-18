#include "Router.h"

size_t Router::readLine(int sockd, char *bptr, size_t maxlen) 
{
    size_t n, rc;
    char    c, *buffer;

    buffer = bptr;

    for ( n = 1; n < maxlen; n++ ) {
	
	if ( (rc = read(sockd, &c, 1)) == 1 ) {
	    *buffer++ = c;
	    if ( c == '\n' )
		break;
	}
	else if ( rc == 0 ) {
	    if ( n == 1 )
		return 0;
	    else
		break;
	}
	else {
	    if ( errno == EINTR )
		continue;
	    return -1;
	}
    }

    *buffer = 0;
    return n;
}

// socket thread constructor
Router::Router(int port) : _port(port), _buffSize(65536), _mkill(false)
{
	// incoming socket setup
    struct sockaddr_in serv_addr; 
    _addr_len = sizeof(_client_addr);
    _buffer = new char[_buffSize];
    int connfd = 0, n = 0;
    _listenfd = 0;
    int buffsize = 65536;

    _listenfd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);

    memset(&serv_addr, '0', sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    serv_addr.sin_port = htons(_port); 
    setsockopt(_listenfd, SOL_SOCKET, SO_RCVBUF, (void*)&buffsize, sizeof(buffsize));
    bind(_listenfd, (struct sockaddr*)&serv_addr, sizeof(serv_addr)); 
    listen(_listenfd, 5); 

    // Prepare our context and sockets
    _context = zmq_ctx_new ();
    _publisher = zmq_socket (_context, ZMQ_PUB);
    
    zmq_bind (_publisher, "tcp://*:6660");


	start(); //starts the thread
}



void Router::run() 
{
	while ( ! _mkill ) 
	{
		bzero(_buffer, _buffSize);
        if (recvfrom(_listenfd, _buffer, _buffSize, 0, &_client_addr, &_addr_len) == -1)
            printf("Got error\n");

        s_send(_publisher, _buffer);
	}
}


Router::~Router() 
{
    // need to close the listener
	_mkill = true;

    // called sent empty msg to exit router
   	struct sockaddr_in si_other;
    int s, i, slen=sizeof(si_other);
             
    if ( (s=socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) == -1)
    {
        std::cerr << "Failed to create close socket msg\n";
        return;
    }
                                          
    memset((char *) &si_other, 0, sizeof(si_other));
    si_other.sin_family = AF_INET;
    si_other.sin_port = htons(_port);
                                                           
    if (inet_aton("127.0.0.1", &si_other.sin_addr) == 0)
    {
        std::cerr << "Failed to connect to headNode router\n";
        return;
    }	

    std::string closeMsg("delete all"); 
    if (sendto(s, closeMsg.c_str(), closeMsg.size() , 0 , (struct sockaddr *) &si_other, slen)==-1)
    {
        std::cerr << "Error Sending message\n";
    }

    close(s);

    // IDEA send the socket a special packet so it knows to close itself TODO
    close(_listenfd);

    // clean up connection
    zmq_close(_publisher);
    zmq_ctx_destroy (_context);

	join();
}
