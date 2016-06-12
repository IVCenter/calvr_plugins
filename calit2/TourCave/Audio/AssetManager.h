#ifndef _ASSET_MANAGER_H_
#define _ASSET_MANAGER_H_

#include <string>
#include <netdb.h>

#include <stdint.h>

#define AM_TCP_PORT 15002
#define AM_UDP_PORT 15003
#define MAX_PACKET_SIZE 4096

class AssetManager {
public:
	AssetManager(const std::string& baseAddress, const std::string& ip);
	AssetManager(const std::string& baseAddress, const std::string& ip, 
		     int tcp_port, int udp_port);
	~AssetManager(); 

	// Must call Init() to load project in Asset Manager
	bool Init();
	// Must call Destroy() to unload project in Asset Manager
	bool Destroy();

	void Mute(bool onoff);
	void SetVolume(float level);
	bool SendTCP(const char* url, const char* format, ...);
	bool SendUDP(const char* url, const char* format, ...);

	static bool SendUDP(const std::string& ip, int portnum, const void* buf, size_t len, int flags=0);

	void SetMasterVolume(float level);
private:
	bool SetupTCPConnection();
	bool SetupUDPSocket();
	bool SendTCPMessage(const char* url, const char* format, ...);
	bool SendUDPMessage(const char* url, const char* format, ...);

	bool inited;

	std::string baseAddress; // Base OSC address
	std::string ip;
	int tcp_port;
	int udp_port;

	int tcp_socket;
	int udp_socket;
	struct sockaddr ai_addr;
	int ai_addrlen;

	uint8_t buf[MAX_PACKET_SIZE];
};

#endif // _ASSET_MANAGER_H_
