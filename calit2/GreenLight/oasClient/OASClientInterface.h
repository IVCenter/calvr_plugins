/**
 * @file    OASClientInterface.h
 * @author  Shreenidhi Chowkwale
 */

#ifndef _OAS_CLIENT_INTERFACE_H_
#define _OAS_CLIENT_INTERFACE_H_

#include <iostream>
#include <fstream>
#include <string>
#include <cstdarg>
#include <cstring>
#include <cstdlib>
#include <sys/socket.h>
#include <sys/stat.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

namespace oasclient
{

class OASClientInterface
{
public:
    enum { MESSAGE_PACKET_SIZE = 256, DATA_PACKET_SIZE = 512};

    /**
     * Initialize the connection to the audio server with the specified host location and port.
     */
    static bool initialize(const std::string &host, const unsigned short port);

    /**
     * Write data to the server, using a format similar to the printf() family of functions.
     */
    static bool writeToServer(const char *format, ...);

    /**
     * Read data from the server. Data and number of bytes are returned by reference via the
     * function parameters.
     */
    static bool readFromServer(char *&data, size_t &count);

    /**
     * Transfer the file with the given path and filename to the server.
     */
    static bool sendFile(const std::string &sPath, const std::string &sFilename);

    /**
     * Shutdown the connection to the server and clean up any allocated resources.
     */
    static bool shutdown();

    /**
     * Modify the global (listener's) gain level. The default is 1, and a value of 0 will mute all
     * sounds completely.
     */
    static bool setListenerGain(float gain);

    /**
     * Modify the listener's position. Default is <0, 0, 0>
     */
    static bool setListenerPosition(float x, float y, float z);

    /**
     * Modify the listener's velocity. Default is <0, 0, 0>. Note that this is ONLY used for
     * doppler effect calculations, and does not cause the position to be updated.
     * If the velocity is NOT set, then doppler effect simulation will not occur.
     */
    static bool setListenerVelocity(float x, float y, float z);

    /**
     * Modify the listener's orientation, in terms of a "look-at" vector and "up" vector.
     * Defaults are <0, 0, -1> and <0, 1, 0>, respectively.
     */
    static bool setListenerOrientation(float atX, float atY, float atZ,
                                       float upX, float upY, float upZ);

private:
    static int _socketFD;

};

}
#endif // _OAS_CLIENT_INTERFACE_H_

