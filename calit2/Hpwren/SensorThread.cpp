#include "SensorThread.h"

#include <iostream>
#include <sstream>
#include <sys/time.h>
#include <unistd.h>
#include <string.h>

using namespace std;

// sensor thread constructor
SensorThread::SensorThread(map< string, Sensor > & sensors)
{
   _mkill = false;

   _timeout_ms = 10000L;

   curl_global_init(CURL_GLOBAL_ALL);

   // create curl connection to the servers
   std::map<std::string, Sensor>::iterator it = sensors.begin();
   for(; it != sensors.end(); it++)
   {
   	_http_handles.push_back(curl_easy_init());

	// add look up to map
	SensorData data;
	data.direction = 0.0f;
	data.velocity = 0.0f;
	data.temperature = 0.0f;
	data.pressure = 0.0f;
    data.humidity = 0.0f;

	strcpy(data.name, it->first.c_str());
	_lookup.insert( std::pair< CURL*, SensorData >( _http_handles.back(), data));

       	printf("creating %s connection\n", data.name);

  	/* set the options (I left out a few, you'll get the point anyway) */
  	curl_easy_setopt(_http_handles.back(), CURLOPT_URL, data.name);
	curl_easy_setopt(_http_handles.back(), CURLOPT_CONNECT_ONLY, 1L);

  	curl_easy_perform(_http_handles.back());
   }

   start(); //starts the thread
}

void SensorThread::getData(std::vector< SensorData > & sensors)
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
  
    int index = 0; 
    std::map<CURL*, SensorData >::iterator it = _lookup.begin();
    for( ; it != _lookup.end(); ++it)
    {
	sensors[index] = it->second;
	index++;
    }
}

void SensorThread::parseDirection(SensorData & data, std::string& msg)
{
   std::string velocity("Dm=");
   float value;
   if( findSubString(velocity, msg, value) )
   {
       data.direction = value;
   }
}

void SensorThread::parseTemperature(SensorData & data, std::string& msg)
{
   std::string temperature("Ta=");
   float value;
   if( findSubString(temperature, msg, value) )
   {
       data.temperature = value;
   }
}

void SensorThread::parseVelocity(SensorData & data, std::string& msg)
{
   std::string velocity("Sm=");
   float value;
   if( findSubString(velocity, msg, value) )
   {
	data.velocity = value;
   }
}

void SensorThread::parsePressure(SensorData & data, std::string& msg)
{
   std::string pressure("Pa=");
   float value;
   if( findSubString(pressure, msg, value) )
   {
	data.pressure = value;
   }
}

void SensorThread::parseHumidity(SensorData & data, std::string& msg)
{
   std::string humidity("Ua=");
   float value;
   if( findSubString(humidity, msg, value) )
   {
	data.humidity = value;
   }
}

bool SensorThread::findSubString(std::string substr, std::string& basestring, float & value)
{
   size_t found = basestring.find(substr);
   if( found != string::npos )
   {
	std::string substring = basestring.substr(found + substr.size(), 4);
	std::stringstream ss(substring);
	ss >> value;
	return true;
   }
   return false;
}

void SensorThread::run() 
{
    while ( ! _mkill ) 
    {
	// iterate through list looking for data to read
	std::map< CURL*, SensorData >:: iterator it = _lookup.begin();
	for(;it != _lookup.end(); it++)
	{
		CURLcode res = curl_easy_getinfo(it->first, CURLINFO_LASTSOCKET, &_sockextr);

       		if( res == CURLE_OK )
               	{
               		wait_on_socket((curl_socket_t)_sockextr, 1000L);
               		curl_easy_recv( it->first, _buffer, sizeof(_buffer), &_size);
			std::string info(_buffer, _size);
   
			_mutex.lock();            		
			parseDirection(it->second, info);
			parseVelocity(it->second, info);
			parseTemperature(it->second, info);
			parsePressure(it->second, info);
			parseHumidity(it->second, info);
			_mutex.unlock();            		
	        }
	}
    }
    _block.release();
}

int SensorThread::wait_on_socket(curl_socket_t sockfd, long timeout_ms)
{
   struct timeval tv;
   fd_set infd, outfd, errfd;
   int res;

   tv.tv_sec = timeout_ms / 1000;
   tv.tv_usec= (timeout_ms % 1000) * 1000;

   FD_ZERO(&infd);
   FD_ZERO(&outfd);
   FD_ZERO(&errfd);

   FD_SET(sockfd, &errfd); // always check for error
   FD_SET(sockfd, &infd);

   // select() returns the number of signalled sockets or -1
   res = select(sockfd + 1, &infd, &outfd, &errfd, &tv);
   return res;
} 


SensorThread::~SensorThread() 
{
      _mkill = true;

      _block.block();

      for(int i = 0; i < (int)_http_handles.size(); i++)
      {
          curl_easy_cleanup(_http_handles[i]);
      }

      curl_global_cleanup();

      join();
}
