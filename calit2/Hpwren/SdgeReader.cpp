#include "SdgeReader.h"

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

SdgeReader::SdgeReader(std::string url, std::map<std::string, Sensor > & sensors, osg::ref_ptr<osgText::Font> font, osg::ref_ptr<osgText::Style> style, std::string fileName, bool rotate)
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

    //std::cerr << "Data: " << test << std::endl;

    size_t foundRecord;
    do
    {
    	foundRecord = test.find("<tr><td>");
    	if(foundRecord != std::string::npos )
    	{
		test = test.substr(foundRecord + 8); 

		// look for closing tag
		size_t found = test.find("</td>");
		if(found != std::string::npos)
		{
	    	std::string name = test.substr(0, found);
	    	//std::cerr << "Name: "  << name << std::endl;

			// TODO is string not empty create Sensor record
			if( !name.empty() && (name.find('#') == std::string::npos))
			{
                // Coords
                float lat, lon;

                // data to be added to list
                Sensor data(true, font, style, rotate);

				// get wind speed
				test = test.substr(5 + name.size());
        
				// need to look for elements to add to each sensor
				size_t foundsub = test.find("</td><td>");
				if( foundsub != std::string::npos )
				{
                    float value = 0.0;
					test = test.substr(foundsub + 9);
					std::stringstream ss (test);
					ss >> value;

                    data.setVelocity(value);
					//std::cerr << "Wind speed: " << data.velocity << std::endl;
				}

				// get direction
				foundsub = test.find("</td><td>");
				if( foundsub != std::string::npos )
				{
                    float value = 0.0;
					test = test.substr(foundsub + 9);
					std::stringstream ss (test);
					ss >> value;
                    
                    data.setDirection(value);
					//std::cerr << "Wind direction: " << data.direction << std::endl;
				}

				// get temperature (not imediately after 3rd element along)
				for(int i = 0; i < 2; i++)
				{
					foundsub = test.find("</td><td>");
					if( foundsub != std::string::npos )
					{
						test = test.substr(foundsub + 9);
					}
				}

				// get temperature
				foundsub = test.find("</td><td>");
				if( foundsub != std::string::npos )
				{
                    float value = 0.0;
					test = test.substr(foundsub + 9);
                    //std::cerr << "5: " << test << std::endl;
					std::stringstream ss (test);
					ss >> value;

                    data.setTemperature(5.0/9.0 * (value -32.0));
					//std::cerr << "Temperature: " << data.temperature << std::endl;
				}
				
                // get lat (8th element along)
				for(int i = 0; i < 7; i++)
				{
					foundsub = test.find("</td><td>");
					if( foundsub != std::string::npos )
					{
						test = test.substr(foundsub + 9);
					}
				}
				
                // get lat
                foundsub = test.find("</td><td>");
				if( foundsub != std::string::npos )
				{
					test = test.substr(foundsub + 9);
					std::stringstream ss (test);
					ss >> lat;
					//std::cerr << "Lat: " << data.lat << std::endl;
				}
                
                // get lon
                foundsub = test.find("</td><td>");
				if( foundsub != std::string::npos )
				{
					test = test.substr(foundsub + 9);
                    
                    std::stringstream ss (test);
					ss >> lon;
                    //std::cerr << "Lon: " << data.lon << std::endl;
				}
            
                // get full name
                foundsub = test.find("</td><td>");
                if( foundsub != std::string::npos )
                {
                    test = test.substr(foundsub + 9);

                    // find end tag
                    foundsub = test.find("</td></tr>");

                    // check to make sure there is some length
                    if( foundsub != std::string::npos )
                    {
                        std::string fullName = test.substr(0, foundsub);
                    
                        // copy short name
        		        //strcpy(data.fullname, fullName.c_str());
					    //std::cerr << "Full name: " << data.fullname << std::endl;
                    }
                }

                // set coord data
                data.setCoord(lon, lat);

                // add sensor
                sensors.insert(std::pair<std::string, Sensor> (name, data));
			}
		}
    	}
    }
    while( foundRecord != std::string::npos );
}
