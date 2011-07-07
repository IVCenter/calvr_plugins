#include "GreenLight.h"
#include <iostream>
#include <sstream>
#include <mxml.h>

// local functions
int intFromString(string str);
Vec3 wattColor(float watt, int minWatt, int maxWatt);

void GreenLight::setPowerColors(bool displayPower)
{
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
        } while ((sensor = mxmlWalkNext(sensor,measurements,MXML_NO_DESCEND)) != NULL);
    }

    map<string,Entity *>::iterator mit;
    Entity * ent;
    for (mit = _components.begin(); mit != _components.end(); mit++)
    {
        float wattage = entityWattsMap[mit->first];
        ent = mit->second;
        ent->setColor( wattColor(wattage, ent->minWattage, ent->maxWattage));
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

    if (minWatt == 0 || maxWatt == 0 || maxWatt < minWatt)
        return Vec3(1,1,1);

    if (watt < minWatt)
        return Vec3(.9,1,1);

    if (watt > maxWatt)
        return Vec3(1,0,0);

    // Watt-Weight:  R,  G,  B
    // minWatt (0):  0,  0,  1
    // low   (.33):  0,  1,  0
    // high  (.67):  1,  1,  0
    // maxWatt (1):  1, .4,  0

    float interpolate = (watt-minWatt)/(maxWatt-minWatt);

    float red = (interpolate-.33)/.34;
    if (red > 1) red = 1;

    float green;
    if (interpolate < .33)
        green = interpolate/.33;
    else if (interpolate < .67)
        green = 1;
    else
        green = 1 - .6*(interpolate-.67)/.33;

    float blue = 1-(interpolate-.33)/.33;
    if (blue < 0) blue = 0;

    return Vec3(red, green, blue);
}
