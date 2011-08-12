#include "Utility.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <kernel/ComController.h>
#include <osgDB/FileUtils>

int utl::intFromString(std::string str)
{
   int i;
   std::stringstream ss;

   ss << str;
   ss >> i;

   return i;
}

std::string utl::stringFromInt(int i)
{
   std::stringstream ss;

   ss << i;

   return ss.str();
}

// Fetch data from server file at url, save at filename, and store contents in std::string given
void utl::downloadFile(std::string downloadUrl, std::string fileName, std::string &content)
{
    if (cvr::ComController::instance()->isMaster())
    {
        // Execute Linux command
        system ( ("curl --retry 1 --connect-timeout 4 --output " + fileName + " \"" + downloadUrl + "\"").c_str() );

        std::ifstream file;
        file.open(fileName.c_str());
        int fileSize = 0;

        if (!file)
        {
            std::cerr << "Error: downloadFile() failed to open:" << fileName << std::endl;
        }
        else
        {
            /*Read in file */
            content = ""; // Just incase
            while(!file.eof())
            {
                content += file.get();
            }
            fileSize = content.length();
        }
        file.close(); 

        cvr::ComController::instance()->sendSlaves(&fileSize, sizeof(fileSize));

        if (fileSize > 0)
        {
            char * cArray = new char[fileSize];
            memcpy(cArray, content.c_str(), fileSize); 
            cvr::ComController::instance()->sendSlaves(cArray, sizeof(char)*fileSize);
            delete[] cArray;
        }
    }
    else //slave nodes
    {
        int fileSize;
        cvr::ComController::instance()->readMaster(&fileSize, sizeof(fileSize));

        if (fileSize > 0)
        {
            char * cArray = new char[fileSize];
            cvr::ComController::instance()->readMaster(cArray, sizeof(char)*fileSize);
            content = cArray;
            delete[] cArray;
        }
    }
}

bool utl::loadShaderSource(osg::Shader * obj, const std::string& fileName)
{
    std::string fqFileName = osgDB::findDataFile(fileName);
    if( fqFileName.length() == 0 )
    {
        std::cout << "File \"" << fileName << "\" not found." << std::endl;
        return false;
    }
    bool success = obj->loadShaderSourceFromFile( fqFileName.c_str());
    if ( !success  )
    {
        std::cout << "Couldn't load file: " << fileName << std::endl;
        return false;
    }
    else
    {
        return true;
    }
}
