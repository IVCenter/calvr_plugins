/**
 * @file    OASClientInterface.cpp
 * @author  Shreenidhi Chowkwale
 */

#include "OASClientInterface.h"

// statics
int oasclient::OASClientInterface::_socketFD = -1;

bool oasclient::OASClientInterface::initialize(const std::string &host, unsigned short port)
{
    struct sockaddr_in stSockAddr;
    int socketFD;

    // Set the address info struct to 0
    memset(&stSockAddr, 0, sizeof(stSockAddr));

    // Set the values for the address struct
    stSockAddr.sin_family = AF_INET;
    stSockAddr.sin_port = htons(port);

    // Convert address from text to binary
    if (0 >= inet_pton(AF_INET, host.c_str(), &stSockAddr.sin_addr))
    {
        return false;
    }

    // Create the socket
    socketFD = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);

    if (-1 == socketFD)
    {
        return false;
    }

    // Connect to the established socket
    if (-1 == connect(socketFD, (struct sockaddr *) &stSockAddr, sizeof(stSockAddr)))
    {
        close(socketFD);
        return false;
    }

    oasclient::OASClientInterface::_socketFD = socketFD;
    return true;
}

bool oasclient::OASClientInterface::shutdown()
{
    if (oasclient::OASClientInterface::writeToServer("QUIT")
        && 0 == close(oasclient::OASClientInterface::_socketFD))
    {
        return true;
    }
    else
    {
        return false;
    }
}
bool oasclient::OASClientInterface::writeToServer(const char *format, ...)
{
    char buf[MESSAGE_PACKET_SIZE * 4] = {0};
    va_list args;

    if (!format || -1 == oasclient::OASClientInterface::_socketFD)
    {
        return false;
    }

    va_start(args, format);
    vsprintf(buf, format, args);
    va_end(args);

    // Check if the formatted message is too long
    if (MESSAGE_PACKET_SIZE < strlen(buf))
    {
        return false;
    }

    if (-1 == write(oasclient::OASClientInterface::_socketFD, buf, MESSAGE_PACKET_SIZE))
    {
        return false;
    }

    return true;
}

bool oasclient::OASClientInterface::readFromServer(char *&data, size_t &count)
{
    char buf[MESSAGE_PACKET_SIZE] = {0};
    int retval;

    if (-1 == oasclient::OASClientInterface::_socketFD)
    {
        return false;
    }

    retval = read(oasclient::OASClientInterface::_socketFD,
                  buf,
                  oasclient::OASClientInterface::MESSAGE_PACKET_SIZE);

    if (-1 == retval || 0 == retval)
    {
        return false;
    }

    count = retval;

    char *newData = new char[count];
    memcpy(newData, buf, count);
    
    data = newData;
    return true;
}

bool oasclient::OASClientInterface::sendFile(const std::string &sPath, const std::string &sFilename)
{
    int fileSize;
    struct stat fileInfo;
    char *data;
    const char *filePath;

    // Append the path and filename together, and assign the filePath pointer to the appended string
    std::string sFilePath = sPath + "/" + sFilename;
    filePath = sFilePath.c_str();

    // Retrieve file information. Note: stat() returns 0 on success
    if (0 != stat(filePath, &fileInfo))
    {
        return false;
    }

    fileSize = fileInfo.st_size;

    // Send the PTFI message to the server
    char buf[MESSAGE_PACKET_SIZE] = {0};
    sprintf(buf, "PTFI %s %d", sFilename.c_str(), fileSize);
    
    if (!oasclient::OASClientInterface::writeToServer(buf))
    {
        return false;
    }

    // Read file from disk
    data = new char[fileSize];
    std::ifstream fileIn(filePath, std::ios::in | std::ios::binary);
    fileIn.read(data, fileInfo.st_size);
    
    // Send the file over the socket
    int bytesLeft, bytesWritten;
    char *dataPtr;
    
    bytesLeft = fileSize;
    dataPtr = data;

    // While we still have bytes/data to write...
    while (bytesLeft > 0)
    {
        // Write a chunk of data out to the socket
        bytesWritten = write(oasclient::OASClientInterface::_socketFD, dataPtr, bytesLeft);

        // If an error occurred, return failure
        if (bytesWritten == 0 || bytesWritten == -1)
        {
            return false;
        }

        // Move the data pointer up by the number of bytes written
        dataPtr += bytesWritten;

        // Reduce the number of bytes remaining by the number of bytes written
        bytesLeft -= bytesWritten;
    }
   
    return true;
}

bool oasclient::OASClientInterface::setListenerGain(float gain)
{
    return writeToServer("GAIN %f", gain);
}


bool oasclient::OASClientInterface::setListenerPosition(float x, float y, float z)
{
    return writeToServer("SLPO %f %f %f", x, y, z);
}


bool oasclient::OASClientInterface::setListenerVelocity(float x, float y, float z)
{
    return writeToServer("SLVE %f %f %f", x, y, z);
}

bool oasclient::OASClientInterface::setListenerOrientation(float atX, float atY, float atZ,
                                                           float upX, float upY, float upZ)
{
    return writeToServer("SLOR %f %f %f %f %f %f", atX, atY, atZ, upX, upY, upZ);
}

