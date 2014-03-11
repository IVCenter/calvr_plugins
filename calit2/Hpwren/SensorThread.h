#ifndef _SENSORTHREAD_H
#define _SENSORTHREAD_H

#include <string>
#include <vector>
#include <map>
#include <queue>
#include <OpenThreads/Thread>
#include <OpenThreads/Mutex>
#include <OpenThreads/Block>
#include <osg/MatrixTransform>
#include <curl/curl.h>
#include "Sensor.h"

struct SensorData {
	char name[255];
	float direction;
	float velocity;
	float temperature;
	float pressure;
	float humidity;
};

class SensorThread : public OpenThreads::Thread
{
	private:
		bool _mkill;
		virtual void run();
		OpenThreads::Mutex _mutex;
		OpenThreads::Block _block;

		// curl
		std::vector< CURL* > _http_handles;
		int _queue;
		char _buffer[4096];
		long _timeout_ms;
		size_t _size;
        	long _sockextr;
		std::map <CURL* ,SensorData > _lookup;
		int wait_on_socket(curl_socket_t sockfd, long timeout_ms);

	        bool findSubString(std::string substr, std::string& basestring, float & value);
		void parseDirection(SensorData& data, std::string& msg);
		void parseVelocity(SensorData& data, std::string& msg);
		void parseTemperature(SensorData& data, std::string& msg);
		void parseHumidity(SensorData& data, std::string& msg);
		void parsePressure(SensorData& data, std::string& msg);
		

	protected:
		SensorThread();

	public:
		SensorThread( std::map<std::string, Sensor > &);
		~SensorThread();
		void getData(std::vector <SensorData> & sensors);
};
#endif
