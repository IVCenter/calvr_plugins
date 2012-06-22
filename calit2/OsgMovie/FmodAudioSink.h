/*
 * FmodAudioSink.h
 *
 *  Created on: Feb 16, 2012
 *     Authors: Philip Weber <pweber@eng.ucsd.edu>,
 *              John Mangan <jmangan@eng.ucsd.edu>
 */

#ifndef FMODAUDIOSINK_H_
#define FMODAUDIOSINK_H_

#include <osg/AudioStream>
#include <osgDB/ReadFile>

#include <fmodex/fmod.hpp>
#include <fmodex/fmod_errors.h>

class FmodAudioSink : public osg::AudioSink
{
    public:

        FmodAudioSink(osg::AudioStream* audioStream);
        ~FmodAudioSink();

        virtual void play();
        virtual void pause();
        virtual void stop();

        virtual bool playing() const { return mStarted && !mPaused; }

        osg::observer_ptr<osg::AudioStream>
        AudioStream() const
        { return mAudioStream; }

    private:

        bool                                mStarted;
        bool                                mPaused;
        osg::observer_ptr<osg::AudioStream> mAudioStream;
        FMOD::Sound *                       mFsound;
        FMOD::Channel*                      mChannel;

        static FMOD::System*                mFsystem;
        OpenThreads::Mutex                  mMutexSingleton;
};


#endif /* FMODAUDIOSINK_H_ */
