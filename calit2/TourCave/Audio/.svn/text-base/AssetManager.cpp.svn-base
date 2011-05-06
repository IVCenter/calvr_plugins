#include "AssetManager.h"

#include <stdarg.h> // va_start, va_end
#include <cstring> // memcpy
#include <cstdio> // sprintf
#include <cmath> // log10

#include <iostream> // std::cerr, std::endl, std::cout

// network
#include <netinet/in.h>
#include <sys/socket.h> 
#include <arpa/inet.h> // htonl

#include "oscpack.h"

AssetManager::AssetManager(const std::string& baseAddress, const std::string& ip)
: inited(false) 
, baseAddress(baseAddress)
, ip(ip)
, tcp_port(AM_TCP_PORT)
, udp_port(AM_UDP_PORT)
{
}

AssetManager::AssetManager(const std::string& baseAddress, const std::string& ip,
			   int tcp_port, int udp_port)
: inited(false) 
, baseAddress(baseAddress)
, ip(ip)
, tcp_port(tcp_port)
, udp_port(udp_port)
{
}

AssetManager::~AssetManager()
{
}

bool AssetManager::Init()
{
	bool ret;

	// Setup the UDP socket
	ret = SetupUDPSocket();
	if (!ret) {
		std::cerr << "Error AssetManager::Init - cannot setup UDP socket" << std::endl;
		return false;
	}

	// Need to work out a better way to initialize...
	inited = true;

	// Send a load message to Asset manager
	ret = SendTCPMessage("/AM/Load", "s", baseAddress.c_str());
//	ret = SendUDPMessage("/AM/Load", "s", baseAddress.c_str());
	if (!ret) {
		std::cerr << "Error AssetManager::Init - cannot send Load message to AM" << std::endl;
		close(udp_socket);
		inited = false;
		return false;
	}
	

	return true;
}

bool AssetManager::Destroy()
{
//	SendTCPMessage("/AM/Unload", "s", baseAddress.c_str());
	SendUDPMessage("/AM/Unload", "s", baseAddress.c_str());
	close(udp_socket);
	inited = false;
	return true;
}

void AssetManager::Mute(bool onoff)
{
//	SendTCPMessage("/AM/Mute", "i", onoff ? 1 : 0);
	SendUDPMessage("/AM/Mute", "i", onoff ? 1 : 0);
}

void AssetManager::SetVolume(float level)
{
	float dB = 20.0 * log10(level);
        std::cerr << "DB: " << dB << std::endl;
	SendTCPMessage("/AM/Project/Volume", "sd", baseAddress.c_str(), dB);
//	SendUDPMessage("/AM/Project/Volume", "sf", baseAddress.c_str(), dB);
}

void AssetManager::SetMasterVolume(float level)
{
	float dB = 20.0 * log10(level);
//	SendTCPMessage("/AM/Volume", "sf", baseAddress.c_str(), dB);
	SendUDPMessage("/AM/Volume", "sf", baseAddress.c_str(), dB);
}

bool AssetManager::SendTCP(const char* url, const char* format, ...)
{
	va_list ap;
	int32_t size;
	std::string str = baseAddress + const_cast<char*>(url);

	va_start(ap, format);
	size = voscpack(buf+4, str.c_str(), format, ap);
	va_end(ap);

	if (size <= 0) {
		return false;
	}
	
	if (!SetupTCPConnection()) {
		return false;
	}

	int32_t bigsize;
	int sent;

	bigsize = htonl(size);
	memcpy(buf, &bigsize, 4);
	sent = send(tcp_socket, buf, size+4, 0);

	close(tcp_socket);
	if (sent == -1) {
		return false;
	}

	return true;
}

bool AssetManager::SendUDP(const char* url, const char* format, ...)
{
	if (!inited) {
		return false;
	}

	va_list ap;
	int32_t size;
	std::string str = baseAddress + const_cast<char*>(url);

	va_start(ap, format);
	size = voscpack(buf, str.c_str(), format, ap);
	va_end(ap);

	sendto(udp_socket, buf, size, 0, &ai_addr, ai_addrlen);

	return true;
}

bool AssetManager::SetupTCPConnection()
{
	char port[7];
	int sockfd, rv;
	struct addrinfo hints, *servinfo, *p;

	// Setup sockets for TCP
	memset(&hints, 0, sizeof hints);
	hints.ai_family = AF_UNSPEC;
	hints.ai_socktype = SOCK_STREAM;
	sprintf(port, "%d", AM_TCP_PORT);
	if ((rv = getaddrinfo(ip.c_str(), port, &hints, &servinfo)) != 0) {
		return false;
	}

	// loop through all the results and connect to the first we can
	for(p = servinfo; p != NULL; p = p->ai_next) {
		if ((sockfd = socket(p->ai_family, p->ai_socktype,
			p->ai_protocol)) == -1) {
			continue;
		}

		if (connect(sockfd, p->ai_addr, p->ai_addrlen) == -1) {
			close(sockfd);
			continue;
		}

		break;
	}

	freeaddrinfo(servinfo);

	if (p == NULL) {
		return false;
	}
	
	tcp_socket = sockfd;
	return true;
}

bool AssetManager::SetupUDPSocket()
{
	char port[7];
	int sockfd, rv;
	struct addrinfo hints, *servinfo, *p;

	// Setup sockets for UDP
	memset(&hints, 0, sizeof hints);
	hints.ai_family = AF_UNSPEC;
	hints.ai_socktype = SOCK_DGRAM;
	sprintf(port, "%d", AM_UDP_PORT);
	if ((rv = getaddrinfo(ip.c_str(), port, &hints, &servinfo)) != 0) {
		return false;
	}

	// loop through all the results and connect to the first we can
	for(p = servinfo; p != NULL; p = p->ai_next) {
		if ((sockfd = socket(p->ai_family, p->ai_socktype,
			p->ai_protocol)) == -1) {
			continue;
		}

		break;
	}

	if (p == NULL) {
		freeaddrinfo(servinfo);
		return false;
	}

	memcpy(&ai_addr, p->ai_addr, sizeof ai_addr);
	ai_addrlen = p->ai_addrlen;
	freeaddrinfo(servinfo);

	udp_socket = sockfd;

	return true;
}

bool AssetManager::SendTCPMessage(const char* url, const char* format, ...)
{
	bool ret = SetupTCPConnection();
	if (ret) {
		int32_t size, bigsize, sent;
		va_list ap;
		va_start(ap, format);
		size = voscpack(buf+4, url, format, ap);
		va_end(ap);
		if (size > 0) {
			bigsize = htonl(size);
			memcpy(buf, &bigsize, 4);
			sent = send(tcp_socket, buf, size+4, 0);
		}
		else {
			close(tcp_socket);
			return false;
		}
		close(tcp_socket);
	}
	else {
		return false;
	}

	return true;
}

bool AssetManager::SendUDPMessage(const char* url, const char* format, ...)
{
	if (inited) {
		int32_t size, sent;
		va_list ap;
		va_start(ap, format);
		size = voscpack(buf, url, format, ap);
		va_end(ap);
		if (size > 0) {
			sent = sendto(udp_socket, buf, size, 0, &ai_addr, ai_addrlen);
		}
		else {
			return false;
		}
	}
	else {
		return false;
	}

	return true;
}

bool AssetManager::SendUDP(const std::string& ip, int portnum, const void* buf, size_t len, int flags)
{
	char port[7];
	int sockfd, rv;
	struct addrinfo hints, *servinfo, *p;
	
	// Setup sockets for UDP
	memset(&hints, 0, sizeof hints);
	hints.ai_family = AF_UNSPEC;
	hints.ai_socktype = SOCK_DGRAM;
	sprintf(port, "%d", portnum);
	if ((rv = getaddrinfo(ip.c_str(), port, &hints, &servinfo)) != 0) {
		return false;
	}

	// loop through all the results and connect to the first we can
	for(p = servinfo; p != NULL; p = p->ai_next) {
		if ((sockfd = socket(p->ai_family, p->ai_socktype,
			p->ai_protocol)) == -1) {
			continue;
		}

		break;
	}

	if (p == NULL) {
		return false;
	}

	int sent, total = 0;
	while (total < len) {
		sent = sendto(sockfd, buf, len, flags, p->ai_addr, p->ai_addrlen);
		if (sent == -1) {
			return false;
		}
		total += sent;
	}

	close(sockfd);
	freeaddrinfo(servinfo);
	return true;
}

