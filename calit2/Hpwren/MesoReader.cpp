#include "MesoReader.h"

#include<fstream>
#include<iostream>
#include <sstream>

#include <string.h>

static size_t http_write(void* buf, size_t size, size_t nmemb, void* userp)
{
	if(userp)
	{
		std::ostringstream* oss = static_cast<std::ostringstream*>(userp);
		std::streamsize len = size * nmemb;
		oss->write(static_cast<char*>(buf), len);
		return nmemb;
	}

	return 0;
}

MesoReader::MesoReader(std::string url, std::map<std::string, Sensor > & sensors, osg::ref_ptr<osgText::Font> font, osg::ref_ptr<osgText::Style> style, std::string fileName, bool rotate)
{
    CURL* curl = curl_easy_init();

    std::stringstream data;
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, &http_write);
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 1L);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_FILE, &data);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 2);
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());

    curl_easy_perform(curl);
    curl_easy_cleanup(curl);

    std::string test = data.str();

    // if empty assign redundant
    if( test.empty() )
    {
        // read in the local file and use that data
        std::ifstream myfile (fileName.c_str());
        data << myfile.rdbuf();
        myfile.close();

        // re-assign to test
        test = data.str();
    }

    //std::cerr << test << std::endl;

    // parse json
    Json::Reader reader;
    Json::Value jsonRoot;

    // parse mounted json data 
    if( !reader.parse(data, jsonRoot,false))
    {
	std::cerr << "Error parsing Json\n";
        return;
    }

    Json::Value & currentArray = jsonRoot["features"];
    
    for(int i = 0; i < currentArray.size(); i++)
    {
	Json::Value prop = currentArray[i]["properties"];

	// only add active towers
	if( prop["description"]["status"].asString().compare("ACTIVE") == 0 && !(prop["description"]["stid"].asString().compare(0,2, "HP") == 0) )
	{
	    Sensor sensor(true, font, style, rotate);
	    sensor.setVelocity(prop["wind_speed"]["value"].asFloat());
	    sensor.setDirection(prop["wind_direction"]["value"].asFloat());
	    sensor.setTemperature(5.0/9.0 * (prop["temperature"]["value"].asFloat() -32.0));
	    sensor.setCoord( currentArray[i]["geometry"]["coordinates"][0].asFloat(),  currentArray[i]["geometry"]["coordinates"][1].asFloat());
    
	    sensors.insert(std::pair<std::string, Sensor> (prop["description"]["name"].asString(), sensor));
	}
    }
}
