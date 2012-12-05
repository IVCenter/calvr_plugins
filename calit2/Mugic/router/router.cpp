// Simple request-reply broker
//
#include "../zhelpers.h"

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

size_t readLine(int sockd, char *bptr, size_t maxlen) {
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

int main (void)
{
    // incoming socket setup
    struct sockaddr_in serv_addr; 
    struct sockaddr client_addr; 
    socklen_t addr_len = sizeof(client_addr);
    size_t buffSize = 1500;
    char buffer[buffSize];
    int listenfd = 0, connfd = 0, n = 0;

    listenfd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    memset(&serv_addr, '0', sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    serv_addr.sin_port = htons(19997); 
    bind(listenfd, (struct sockaddr*)&serv_addr, sizeof(serv_addr)); 
    listen(listenfd, 5); 

    // Prepare our context and sockets
    void *context = zmq_ctx_new ();

    void *publisher = zmq_socket (context, ZMQ_PUB);
    zmq_bind (publisher, "tcp://*:5560");

    while( true )
    {
        bzero(buffer, buffSize);
        if (recvfrom(listenfd, buffer, buffSize, 0, &client_addr, &addr_len)==-1)
             printf("Got error\n");
    
		//printf("%s", buffer);
        
        s_send(publisher, buffer);    

        // dont use up all cycles
        //sleep(1);
    }


    // We never get here but clean up anyhow
    zmq_close (publisher);
    zmq_ctx_destroy (context);
    return 0;
}

