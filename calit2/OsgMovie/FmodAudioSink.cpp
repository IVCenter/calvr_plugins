/*
 * FmodAudioSink.cpp
 *
 *  Created on: Feb 16, 2012
 *     Authors: Philip Weber <pweber@eng.ucsd.edu>,
 *              John Mangan <jmangan@eng.ucsd.edu>
 */

#include "FmodAudioSink.h"
#include <string.h>

using namespace osg;

/*static*/ FMOD::System *  FmodAudioSink::mFsystem = NULL;

static FMOD_RESULT F_CALLBACK soundReadCallback(FMOD_SOUND *in_sound, void *in_data, unsigned int datalen)
{
    FMOD::Sound* sound = reinterpret_cast<FMOD::Sound*>(in_sound);
    FmodAudioSink* sink  = 0;
    sound->getUserData(reinterpret_cast<void**>(&sink));

    osg::observer_ptr<osg::AudioStream> as = sink->AudioStream();
    if (as.valid())
        as->consumeAudioBuffer(in_data, datalen);

    return FMOD_OK;
}

FmodAudioSink::FmodAudioSink(osg::AudioStream* audioStream):
    mStarted(false),
    mPaused(false),
    mAudioStream(audioStream)
{
    // Thread-safe singleton creation of mFsystem
    if (!mFsystem)
    {
        OpenThreads::ScopedLock<OpenThreads::Mutex> lock(mMutexSingleton);
        if (!mFsystem)
        {
            FMOD::System_Create(&mFsystem);
            mFsystem->init(32, FMOD_INIT_NORMAL, 0);
            unsigned int ui = 0;
        }
    }
}

FmodAudioSink::~FmodAudioSink()
{
    stop();
}

void FmodAudioSink::play()
{
    if (mStarted)
    {
        if (mPaused)
        {
            mChannel->setPaused(false);
            mPaused = false;
            mFsystem->update();
        }
        return;
    }


    mStarted = true;
    mPaused = false;

    osg::notify(osg::NOTICE)<<"FmodAudioSink()::play()"<<std::endl;
    osg::notify(osg::NOTICE)<<"  audioFrequency()="<<mAudioStream->audioFrequency()<<std::endl;
    osg::notify(osg::NOTICE)<<"  audioNbChannels()="<<mAudioStream->audioNbChannels()<<std::endl;
    osg::notify(osg::NOTICE)<<"  audioSampleFormat()="<<mAudioStream->audioSampleFormat()<<std::endl;

    FMOD_CREATESOUNDEXINFO exinfo;
    memset(&exinfo, 0, sizeof(FMOD_CREATESOUNDEXINFO));
    exinfo.cbsize = sizeof(FMOD_CREATESOUNDEXINFO);              /* required. */
    exinfo.decodebuffersize = 2048;//mAudioStream->audioFrequency();              /* Chunk size of stream update in samples.  This will be the amount of data passed to the user callback. */
    exinfo.length = mAudioStream->audioFrequency() * mAudioStream->audioNbChannels() * sizeof(signed short); /* Length of PCM data in bytes of whole song (for Sound::getLength) */

    exinfo.numchannels = mAudioStream->audioNbChannels();
    exinfo.defaultfrequency = mAudioStream->audioFrequency();              /* Default playback rate of sound. */
    exinfo.format = FMOD_SOUND_FORMAT_PCM16;                     /* Data format of sound. */
    exinfo.pcmreadcallback = soundReadCallback;                           /* User callback for reading. */
    exinfo.userdata = this;

    FMOD_MODE mode = FMOD_2D | FMOD_OPENUSER  | FMOD_HARDWARE | FMOD_LOOP_NORMAL| FMOD_CREATESTREAM;
    FMOD_RESULT result = mFsystem->createStream(NULL, mode, &exinfo, &mFsound);

    mFsystem->update();

    if (result == FMOD_OK)
    {
        FMOD_RESULT result2 = mFsystem->playSound(FMOD_CHANNEL_FREE, mFsound, false, &mChannel);
        if (result2 == FMOD_OK)
               mChannel->setVolume(1000);
        else
        {
           printf("Error found\n");
               std::string errorstr(FMOD_ErrorString(result2));
               throw "FMOD_playSound() failed (" + errorstr + ")";
               mChannel = NULL;
        }
    }
    else
    {
        throw "FMOD_createStream() failed (" + std::string(FMOD_ErrorString(result)) + ")";
         mFsound = NULL;
         mChannel = NULL;
    }
}

void FmodAudioSink::pause()
{
    if (mStarted)
    {
        mChannel->setPaused(true);
        mPaused = true;
        mFsystem->update();
    }
}

void FmodAudioSink::stop()
{
    if (mStarted)
    {
         if(mChannel)
         {
               if (!mPaused)
                    mChannel->setPaused(true);
               mChannel->stop();
               mChannel = NULL;
         }

         if(mFsound)
         {
              mFsound->release();
              mFsound = NULL;
         }

         mFsystem->update();

         osg::notify(osg::NOTICE)<<"~FmodAudioSink() destructor, but still playing"<<std::endl;
    }
}
