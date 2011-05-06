#ifndef _FETCHQUEUE_H
#define _FETCHQUEUE_H

#include <iostream>           // For cerr and cout
#include <cstdlib>            // For atoi()

#include <map>
#include <queue>
#include <OpenThreads/Thread>
#include <OpenThreads/Mutex>
#include <OpenThreads/Block>
#include "Geometry.h"

/** @file FetchQuene.h
 *  Simple request object that contains frame number, Geometry pointer
 *  and a char pointer.
 */
struct Request
{
	int frame;  /// frame number for geometry that is requested.
	Geometry* geom; /// pointer to requested object which contains file offset info.
        char * buffer; /// pointer where data should be transfered.
};

/** FetchQueue class is a threadsafe queue classes used for 
 *  loading parts from disk in a background thread.
 */ 
class FetchQueue : public OpenThreads::Thread
{
	private:
		bool _mkill;
		virtual void run();
		OpenThreads::Mutex _mutex;
                OpenThreads::Mutex _readLock;
		OpenThreads::Block _blocked;
		std::queue< Request > _incoming;
		int _maxframes;
		static FetchQueue* fq;

	protected:
		FetchQueue();
		~FetchQueue();

	public:
                // Single instance accessor.
		static FetchQueue* getInstance();
                /// Add a part/Geometry request to the queue.
		void addRequest(int frameNum, Geometry* geom, char * buffer = NULL);
                /// set the maximum number of frames in the sequence.
		void setMaxFrames(int frame) { _maxframes = frame; };
                void lock();
                void unlock();
};
#endif
