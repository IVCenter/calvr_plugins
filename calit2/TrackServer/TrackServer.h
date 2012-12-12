#ifndef _TRACKSERVER_
#define _TRACKSERVER_

#include <cvrConfig/ConfigManager.h>
#include <cvrKernel/CVRPlugin.h>
#include <cvrKernel/PluginHelper.h>
#include <OpenThreads/Thread>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <unistd.h>

#define BUFLEN 512
#define LOCLEN 512
#define NPACK 10
#define PORT 44444

class TrackServer : public cvr::CVRPlugin {
  public:
    bool init();
};

class ServerThread : public OpenThreads::Thread {
  public:
	struct sockaddr_in si_me, si_other;
	int s, i; 
    socklen_t slen;
	char buf[BUFLEN], *location;

    ServerThread();
    virtual void run();
    void diep(const char *s);
};

#endif
