#include "FetchQueue.h"

FetchQueue* FetchQueue::fq = NULL;

// thread constructor
FetchQueue::FetchQueue()
{
   _mkill = false;
   _maxframes = 0;
   start(); //starts the thread
}

FetchQueue* FetchQueue::getInstance()
{
	if( fq == NULL)
		fq = new FetchQueue();

	return fq;
}

void FetchQueue::addRequest(int framenum, Geometry* geom, char * buffer)
{
	OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);

	// make sure frame is not outside scope
	if( framenum >= _maxframes )
		framenum = 0;

	Request request;
	request.frame = framenum;
	request.geom = geom;
	request.buffer = buffer;

	_incoming.push(request);
	_blocked.release();
}

void FetchQueue::run() 
{
    while ( ! _mkill ) 
    {

	// check if request queue has data to process
	if(_incoming.size() )
	{

		// mutex lock queue
		_mutex.lock();
		Request req = _incoming.front();
		_incoming.pop();
		_mutex.unlock();

		// lock disk read during post traversal
		_readLock.lock();
		//open correct file and reference correct drawable and load into _data
		//check if data needs to be updated
		req.geom->SetBuffer(req.frame, req.buffer);
		_readLock.unlock();
	}
	else
	{
		//printf("Fetch thread blocked\n");
		// block until a new request is added
		_blocked.block();
            	_blocked.reset();
	} 		
    }
}

FetchQueue::~FetchQueue() 
{
      _mkill = true;
      join();
}

void FetchQueue::lock()
{
    _readLock.lock();
}

void FetchQueue::unlock()
{
    _readLock.unlock();
}

