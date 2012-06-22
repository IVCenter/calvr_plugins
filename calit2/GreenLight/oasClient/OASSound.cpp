/**
 * @file    OASSound.cpp
 * @author  Shreenidhi Chowkwale
 */

#include "OASSound.h"

oasclient::OASSound::OASSound(const std::string &sPath, const std::string &sFilename)
{
    _init();
    _path = sPath;
    _filename = sFilename;

    _handle = _getHandleFromServer();
    
    if (-1 == _handle)
    {
        if (oasclient::OASClientInterface::sendFile(_path, _filename))
        {
            _handle = _getHandleFromServer();
        }
    }
}

oasclient::OASSound::OASSound(WaveformType waveType, float frequency, float phaseShift, float durationInSeconds)
{
    _init();

    if (oasclient::OASClientInterface::writeToServer("WAVE %d, %f, %f, %f", waveType,
                                                                         frequency,
                                                                         phaseShift,
                                                                         durationInSeconds))
    {
        char *handleString;
        size_t length;

        if (oasclient::OASClientInterface::readFromServer(handleString, length))
        {
            _handle = atol(handleString);
        }
    }

}

oasclient::OASSound::~OASSound()
{
    if (isValid())
    {
        oasclient::OASClientInterface::writeToServer("RHDL %ld", _handle);
    }
    _path.clear();
    _filename.clear();
}

void oasclient::OASSound::_init()
{
    _handle = -1;
}

long oasclient::OASSound::_getHandleFromServer()
{
    if (!oasclient::OASClientInterface::writeToServer("GHDL %s", _filename.c_str()))
    {
        return -1;
    }

    char *handleString;
    size_t length;

    if (!oasclient::OASClientInterface::readFromServer(handleString, length))
    {
        return -1;
    }

    return atol(handleString);
}

bool oasclient::OASSound::isValid()
{
    return (_handle >= 0);
}

long oasclient::OASSound::getHandle()
{
    return _handle;
}

bool oasclient::OASSound::play()
{
    return oasclient::OASClientInterface::writeToServer("PLAY %ld", _handle);
}

bool oasclient::OASSound::stop()
{
    return oasclient::OASClientInterface::writeToServer("STOP %ld", _handle);
}

bool oasclient::OASSound::setLoop(bool loop)
{
    return oasclient::OASClientInterface::writeToServer("SSLP %ld %ld", _handle, loop ? 1 : 0);
}

bool oasclient::OASSound::setGain(float gain)
{
    return oasclient::OASClientInterface::writeToServer("SSVO %ld %f", _handle, gain);
}

bool oasclient::OASSound::setPosition(float x, float y, float z)
{
    return oasclient::OASClientInterface::writeToServer("SSPO %ld %f %f %f", _handle, x, y, z);
}

bool oasclient::OASSound::setDirection(float angle)
{
    return oasclient::OASClientInterface::writeToServer("SSDI %ld %f", _handle, angle);
}

bool oasclient::OASSound::setDirection(float x, float y, float z)
{
    return oasclient::OASClientInterface::writeToServer("SSDI %ld %f %f %f", _handle, x, y, z);
}

bool oasclient::OASSound::setVelocity(float x, float y, float z)
{   
    return oasclient::OASClientInterface::writeToServer("SSVE %ld %f %f %f", _handle, x, y, z);
}

bool oasclient::OASSound::setPitch(float pitchFactor)
{
    return oasclient::OASClientInterface::writeToServer("SPIT %ld %f", _handle, pitchFactor);
}
