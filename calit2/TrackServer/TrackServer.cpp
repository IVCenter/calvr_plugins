#include "TrackServer.h"
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <osg/Matrix>
#include <sys/time.h>
#include <unistd.h>
using namespace std;
using namespace cvr;
using namespace osg;

#define CLIENTS 1000
#define IDSIZE  10
#define TIMEOUT_THRESHOLD 1000

Matrixd mat0;
Matrixd mat1;
char clients[CLIENTS][IDSIZE];
int  timeout[CLIENTS];
char status1[6] = "1not\0";
char status2[6] = "2not\0";
CVRPLUGIN(TrackServer)

bool TrackServer::init() {
   ServerThread *thread = new ServerThread();
   memset((void*)&clients,'\0',sizeof(clients));
   thread->start();
}

Matrixd getStubMat(int seed){
    Matrixd *mat = new Matrixd(
        seed, seed, seed, seed,
        seed, seed, seed, seed,
        seed, seed, seed, seed,
        seed, seed, seed, seed);
    return *mat;
}

/*
Return: client index in list if it exists. -1 on failure.
*/
int getClient(char* id){
    for(int i = 0; i < CLIENTS; i++){
        if(strncmp(clients[i], id, IDSIZE) == 0)
            return i;
    }
    return -1;
}

/*
Return: client position in list on success, -1 on failure (ie. list is full).
*/
int insertClient(char* id){
    for(int i = 0; i < CLIENTS; i++){
        if(clients[i][0] == '\0'){
            strncpy(clients[i], id, IDSIZE);
            return i;
        }
    }
    return -1;
}

/*
Return: 1 on successful removal, -1 on failute (ie. no such client)
*/
int removeClient(int index){
    if(clients[index][0] == '\0')
        return -1;
    memset((char *) clients[index], '\0', IDSIZE);
    return 0;
}

void updateTimeoutList(){
    for(int i = 0; i < CLIENTS; i++){
        timeout[i]++;
    }
}

/*
Return: number of clients removed due to timeout
*/
int removeTimeouts(){
    int clientsRemoved = 0;
    for(int i = 0; i < CLIENTS; i++){
        if(timeout[i] > TIMEOUT_THRESHOLD){
            removeClient(i);
            clientsRemoved++;
        }
    }
    return clientsRemoved;
}

void updateDrawingStatus(char *buf){
    switch(buf[0]){
    case '1':
        if(buf[1] == 'd' & status1[1] != 'd')
            strncpy(status1,buf,5);
        else if(buf[1] == 'n' & status1[1] != 'n')
            strncpy(status1,buf,4);
        break;
    case '2':
        if(buf[1] == 'd' & status2[1] != 'd')
            strncpy(status2,buf,5);
        else if(buf[1] == 'n' & status2[1] != 'n')
            strncpy(status2,buf,4);
        break;
    }
}
ServerThread::ServerThread() : OpenThreads::Thread(){ 
    slen = sizeof(sockaddr_in);
    location = (char*)malloc(LOCLEN * sizeof(char*));

    // Create socket
	if ((s = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) == -1)
	    diep("Socket could not be created");

    // Bind socket to PORT
	memset((char *) &si_me, 0, sizeof(si_me));
	si_me.sin_family = AF_INET;
	si_me.sin_port = htons(PORT);
	si_me.sin_addr.s_addr = htonl(INADDR_ANY);
	if (bind(s, (const sockaddr*)&si_me, sizeof(si_me)) == -1)
		diep("Could not bind socket");

    // Initialize if drawing or not

	printf("Server started...\n");
}
inline long getMilliSecs(){
    struct timeval t;
    gettimeofday(&t, NULL);
    return ((t.tv_sec) * 1000 + t.tv_usec/1000.0) + 0.5;
}


void ServerThread::run(){
    cerr << "Thread init\n";
    int i = 0;
    int currentClientId = -1;
    int timeoutsRemoved = -1;
    unsigned long start, stop, useconds;
    while(true){
        // Clear buffers
        memset((char *) buf, 0, BUFLEN);
        memset((char *) location, 0, LOCLEN);

        // Receive packet
		if (recvfrom(s, buf, BUFLEN, 0, (struct sockaddr *)&si_other, &slen) == -1)
			diep("recvfrom()");
//		printf("Received packet from %s:%d\nData: %s",
//				inet_ntoa(si_other.sin_addr), ntohs(si_other.sin_port), buf);

        buf[BUFLEN-1] = '\0';
        // Update if either device is drawing or not
        start = getMilliSecs();
        updateDrawingStatus(buf);




        // Maintain clients list
        currentClientId = getClient(buf);
        if(currentClientId == -1)
            currentClientId = insertClient(buf);
        updateTimeoutList();
        timeoutsRemoved = removeTimeouts();

        // Get tracker info
        if(currentClientId < 2 && currentClientId > -1){ // Range of physical trackers
            mat0 = PluginHelper::getHandMat(currentClientId);
            mat1 = PluginHelper::getHeadMat(0);
        }else{
            mat0 = getStubMat(currentClientId);
            mat1 = getStubMat(currentClientId);
        }


        // ** DEBUG **
       // printf("|%f, %f, %f, %f|\n", mat0(0,0), mat0(1,0), mat0(2,0), mat0(3,0));
       // printf("|%f, %f, %f, %f|\n", mat0(0,1), mat0(1,1), mat0(2,1), mat0(3,1));
       // printf("|%f, %f, %f, %f|\n", mat0(0,2), mat0(1,2), mat0(2,2), mat0(3,2));
       // printf("|%f, %f, %f, %f|\n\n", mat0(0,3), mat0(1,3), mat0(2,3), mat0(3,3));            
        // <END>** DEBUG **

        sprintf(location, "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,|%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,|%s,%s\0",
            mat0(0,0),mat0(0,1),mat0(0,2),mat0(0,3),mat0(1,0),mat0(1,1),mat0(1,2),mat0(1,3),
            mat0(2,0),mat0(2,1),mat0(2,2),mat0(2,3),mat0(3,0),mat0(3,1),mat0(3,2),mat0(3,3),
            mat1(0,0),mat1(0,1),mat1(0,2),mat1(0,3),mat1(1,0),mat1(1,1),mat1(1,2),mat1(1,3),
            mat1(2,0),mat1(2,1),mat1(2,2),mat1(2,3),mat1(3,0),mat1(3,1),mat1(3,2),mat1(3,3),
            status1, status2);

        // Send packet
		if (sendto(s, location, LOCLEN, 0, (struct sockaddr *)&si_other, slen) == -1)
			diep("sendto()");
//		printf("Sent back: %s\n\n", location);

        stop = getMilliSecs();
        useconds = stop - start;
        cout << "Time elapsed: " << useconds << endl;
    }
}

void ServerThread::diep(const char *s){
	perror(s);
	exit(1);
}
