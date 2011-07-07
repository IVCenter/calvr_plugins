#include "GreenLight.h"
#include <iostream>
#include <sstream>
#include <mxml.h>

// local functions
int intFromString(string str);
Vec3 wattColor(float watt, int minWatt, int maxWatt);

void GreenLight::setPowerColors(bool displayPower)
{
    int maxWattage = 0;
    int minWattage = 0;
    std::map< string, int> entityWattsMap;

    if (!displayPower)
    {
        map<string,Entity *>::iterator mit;
        for (mit = _components.begin(); mit != _components.end(); mit++)
            mit->second->setColor(Vec3(.7,.7,.7));
        return;
    }

    // Display power per component
    FILE *fp = fopen(ConfigManager::getEntry("local", "Plugin.GreenLight.Power", "").c_str(), "r");
    if (!fp)
    {
        cerr << "Error (setComponentColors): Cannot open \"" << ConfigManager::getEntry("local", "Plugin.GreenLight.Power", "") << "\"." << endl;
        _displayPowerCheckbox->setValue(false);
        return;
    }

    mxml_node_t * xmlTree = mxmlLoadFile(NULL, fp, MXML_TEXT_CALLBACK);

    mxml_node_t * measurements = mxmlFindElement(xmlTree,xmlTree,"measurements",NULL,NULL,MXML_DESCEND_FIRST);

    if (measurements == NULL)
    {
        std::cerr << "Warning: No <measurements> tag in xml power file. Aborting. ";
        return;
    }

    mxml_node_t * sensor = mxmlFindElement(measurements,measurements,"sensor",NULL,NULL,MXML_DESCEND_FIRST);

    if (sensor)
    {
        do
        {
            mxml_node_t * nameNode = mxmlFindElement(sensor,sensor,"name",NULL,NULL,MXML_DESCEND_FIRST);
            mxml_node_t * valueNode = mxmlFindElement(sensor,sensor,"value",NULL,NULL,MXML_DESCEND_FIRST);

            if (nameNode == NULL || nameNode->child->value.text.whitespace == 1)
            {
                std::cerr << "Error parsing power xml file (bad name on sensor)." << endl;
                continue;
            }

            string name = nameNode->child->value.text.string;

            if (valueNode == NULL || valueNode->child->value.text.whitespace == 1)
            {
                std::cerr << "Error parsing power xml file value for \"" << name <<"\"." << endl;
                continue;
            }
         
            string value = valueNode->child->value.text.string;

            int wattage = intFromString(value);
            entityWattsMap[name] = wattage;

            if (wattage > maxWattage)
                maxWattage = wattage;

            if (wattage > 0 && wattage < minWattage)
                minWattage = wattage;

        } while ((sensor = mxmlWalkNext(sensor,measurements,MXML_NO_DESCEND)) != NULL);
    }

    map<string,Entity *>::iterator mit;
    for (mit = _components.begin(); mit != _components.end(); mit++)
    {
        float wattage = entityWattsMap[mit->first];
        mit->second->setColor(wattColor(wattage,minWattage,maxWattage));
    }
}

int intFromString(string str)
{
   int i;

   std::stringstream ss;

   ss << str;
   ss >> i;

   return i;
}

Vec3 wattColor(float watt, int minWatt, int maxWatt)
{
    if (watt == 0)
        return Vec3(.2,.2,.2);

    float interpolate = (watt-minWatt)/(maxWatt-minWatt);

    float red = (interpolate-.5)/.25;
    if (red < 0) red = 0;
    if (red > 1) red = 1;

    float green;
    if (interpolate < .25)
        green = 1 - interpolate/.25;
    else if (interpolate < .5)
        green = (interpolate-.25)/.25;
    else if (interpolate < .75)
        green = 1;
    else
        green = 1 - (interpolate-.75)/.25;

    float blue = 1-(interpolate-.25)/.25;
    if (blue < 0) blue = 0;
    if (blue > 1) blue = 1;

    return Vec3(red, green, blue);
}
