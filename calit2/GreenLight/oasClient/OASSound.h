/**
 * @file    OASSound.h
 * @author  Shreenidhi Chowkwale
 */

#ifndef _OAS_SOUND_H_
#define _OAS_SOUND_H_

#include "OASClientInterface.h"

namespace oasclient
{

class OASSound
{
public:

    /**
     * These are the waveform types supported by the server for sound sources generated based
     * on simple waves. The sine wave is the most commonly requested type.
     */
    enum WaveformType
    {
        SINE = 1,
        SQUARE = 2,
        SAWTOOTH = 3,
        WHITENOISE = 4,
        IMPULSE = 5
    };

    /**
     * Create a new sound source based on a file with the given path and filename.
     */
    OASSound(const std::string &sPath, const std::string &sFilename);

    /**
     * Create a new sound source based on the specified wavetype, frequency and phaseshift.
     */
    OASSound(WaveformType waveType, float frequency, float phaseShift, float durationInSeconds);
    ~OASSound();

    bool isValid();

    long getHandle();

    /**
     * Play the sound source. Play always resumes from the beginning.
     */
    bool play();

    /**
     * Stop playing the sound source.
     */
    bool stop();

    /**
     * Set the sound source to loop or stop looping.
     */
    bool setLoop(bool loop);

    /**
     * Set the gain (volume) of the sound source. The default is 1.
     */
    bool setGain(float gain);

    /**
     * Set the position of the sound source.
     */
    bool setPosition(float x, float y, float z);

    /**
     * Set the direction of the sound source by specifying an angle in the X-Z plane.
     */
    bool setDirection(float angle);

    /**
     * Set the direction of the sound source by specifying a directional vector. I
     */
    bool setDirection(float x, float y, float z);

    /**
     * Set the velocity of the sound source. The velocity is used ONLY for the doppler effect
     * calculations. The server does not internally update the position based on the specify the
     * velocity.
     */
    bool setVelocity(float x, float y, float z);

    /**
     * Set the pitch of the sound. This works by changing the rate of playback of the sound source.
     * @param pitchFactor The default pitchFactor is 1. Multiplying the pitchFactor by 2 will
     *                    increase the pitch by one octave, and dividing the pitchFactor by 2 will
     *                    decrease the pitch by one octave.
     */
    bool setPitch(float pitchFactor);


private:
    void _init();
    long _getHandleFromServer();

    long _handle;
    std::string _filename;
    std::string _path;

};

}

#endif // _OAS_SOUND_H_

