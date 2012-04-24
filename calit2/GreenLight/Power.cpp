#include "GreenLight.h"
#include <iostream>
#include <sstream>
#include <mxml.h>

/***
 * TODO: LOOK AT THIS SECTION 2/21/12
 * Parses through an XML file and sets the watt color of individual components?
 * Called on Button Action Event....
 */
void GreenLight::setPowerColors(bool displayPower)
{
    std::map< std::string, std::map< std::string, int > > componentWattsMap;

    /***
     * If the switch is off, revert all components colors back to their default color(?)
     */
    if (!displayPower)
    {
        std::set< Component *>::iterator sit;
        for (sit = _components.begin(); sit != _components.end(); sit++)
            (*sit)->defaultColor();
        return;
    }

    // Display power per component
    FILE *fp = fopen(cvr::ConfigManager::getEntry("local",
                      "Plugin.GreenLight.Power", "").c_str(), "r");
    if (!fp)
    { //If there is no valid entry? file?
        std::cerr << "Error (setComponentColors): Cannot open \""
                  << cvr::ConfigManager::getEntry("local", "Plugin.GreenLight.Power", "")
                  << "\"." << std::endl;

        _displayPowerCheckbox->setValue(false);
        return;
    }

    mxml_node_t * xmlTree = mxmlLoadFile(NULL, fp, MXML_TEXT_CALLBACK);

    mxml_node_t * measurements = 
        mxmlFindElement(xmlTree,xmlTree,"measurements",NULL,NULL,MXML_DESCEND_FIRST);

    if (measurements == NULL)
    {
        std::cerr << "Warning: No <measurements> tag in xml power file. Aborting." << std::endl;
        return;
    }

    mxml_node_t * sensor = mxmlFindElement(measurements,measurements,"sensor",
                                           NULL,NULL,MXML_DESCEND_FIRST);

    if (sensor)
    { // interpret xml
        do
        {
            mxml_node_t * nameNode = mxmlFindElement(sensor,sensor,"name",
                                         NULL,NULL,MXML_DESCEND_FIRST);
            mxml_node_t * timeNode = mxmlFindElement(sensor,sensor,"time",
                                         NULL,NULL,MXML_DESCEND_FIRST);
            mxml_node_t * valueNode = mxmlFindElement(sensor,sensor,"value",
                                         NULL,NULL,MXML_DESCEND_FIRST);

            if (nameNode == NULL || nameNode->child->value.text.whitespace == 1)
            {
                std::cerr << "Error parsing power xml file (bad name on sensor)." << std::endl;
                continue;
            }

            std::string name = nameNode->child->value.text.string;

            if (timeNode == NULL || timeNode->child == NULL || timeNode->child->next == NULL)
            {
                std::cerr << "Error parsing power xml file for \"" << name <<"\" time." << std::endl;
                continue;
            }

            std::string time = timeNode->child->value.text.string;
            time += " ";
            time += timeNode->child->next->value.text.string;

            if (valueNode == NULL || valueNode->child->value.text.whitespace == 1)
            {
                std::cerr << "Error parsing power xml file for \"" << name <<"\" value." << std::endl;
                continue;
            }

            std::string value = valueNode->child->value.text.string;
            int wattage = utl::intFromString(value);

            std::map< std::string, std::map< std::string, int > >::iterator mit;
            if ((mit = componentWattsMap.find(name)) == componentWattsMap.end())
            {
                std::map<std::string, int> newMap;
                newMap[time] = wattage;
                componentWattsMap[name] = newMap;
            }
            else
            {
                std::map< std::string, int >::iterator tit;
                if ((tit = mit->second.find(time)) == mit->second.end())
                    mit->second[time] = wattage;
                else
                    tit->second += wattage;                
            }
        } while ((sensor = mxmlWalkNext(sensor,measurements,MXML_NO_DESCEND)) != NULL);
    }

    std::set< Component * >::iterator sit;
    for (sit = _components.begin(); sit != _components.end(); sit++)
    {
        std::list< osg::Vec3 > colors;
        std::map< std::string, std::map< std::string, int > >::iterator cit;
        if ((cit = componentWattsMap.find((*sit)->name)) != componentWattsMap.end())
        {
            std::list< int > watts;
            int minWatt = 0, maxWatt = 0;
            std::map< std::string, int >::iterator mit;
            for (mit = cit->second.begin(); mit!= cit->second.end(); mit++)
            {
                 watts.push_back( mit->second );
                 if (mit->second != 0 && (mit->second < minWatt || minWatt == 0))
                     minWatt = mit->second;
                 if (mit->second > maxWatt)
                     maxWatt = mit->second;
            }

            std::list< int >::iterator lit;
            for (lit = watts.begin(); lit != watts.end(); lit++)
            {
              // if the magnifyRange checkbox is set to true.. use minWatt/maxWatt as the color,
              //  otherwise, use the current iteration's (sit) Component's color value..
                 if (_magnifyRangeCheckbox->getValue())
                     colors.push_back( wattColor(*lit, minWatt, maxWatt) );
                 else
                     colors.push_back( wattColor(*lit, (*sit)->minWattage, (*sit)->maxWattage) );
              }
        }
        else
        {
            colors.push_back( wattColor(0, (*sit)->minWattage, (*sit)->maxWattage) );
        }
        (*sit)->setColor(colors);
    }
}

osg::Vec3 GreenLight::wattColor(float watt, int minWatt, int maxWatt)
{
    if (watt == 0)  // If it is off, grey?
        return osg::Vec3(.2,.2,.2);

    // White if any of the min/max constraints are zero
    // OR if max is less than min.
    if (minWatt == 0 || maxWatt == 0 || maxWatt < minWatt)
        return osg::Vec3(1,1,1);

    // Less than the minimum watt reqs.
    if (watt < minWatt)
        return osg::Vec3(.8,.8,1);

    // Over the maximum watt reqs.
    if (watt > maxWatt)
        return osg::Vec3(1,0,0);

    // Watt-Weight:  R,  G,  B
    // minWatt (0):  0,  0,  1
    // low   (.33):  0,  1,  0
    // high  (.67):  1,  1,  0
    // maxWatt (1):  1, .4,  0

    float interpolate = (watt-minWatt)/(maxWatt-minWatt);

    if (minWatt == maxWatt)
        interpolate = .5;

    float red = (interpolate-.33)/.34;
    if (red < 0) red = 0;
    if (red > 1) red = 1;

    float green;
    if (interpolate < .33)
        green = interpolate/.33;
    else if (interpolate < .67)
        green = 1;
    else
    {
        float maxGreenSub = 1.0; // .6
        green = 1 - maxGreenSub * (interpolate-.67)/.33;

        if (green < (1 - maxGreenSub) ) green = ( 1 - maxGreenSub ) ;
        if (green > 1) green = 1;
    }


    float blue = 1 - (interpolate-.33)/.33;
    if (blue < 0) blue = 0;
    if (blue > 1) blue = 1;

    return osg::Vec3(red, green, blue);
}

void GreenLight::createTimestampMenus()
{
    std::vector<std::string> years;
    years.push_back("2011");

    std::vector<std::string> months;
    months.push_back("January");
    months.push_back("Febuary");
    months.push_back("March");
    months.push_back("April");
    months.push_back("May");
    months.push_back("June");
    months.push_back("July");
    months.push_back("August");
    months.push_back("September");
    months.push_back("October");
    months.push_back("November");
    months.push_back("December");

    std::vector<std::string> days;
    for (int i = 1; i <= 31; i++)
        days.push_back(std::string(i < 10 ? "0" : "") + utl::stringFromInt(i));

    std::vector<std::string> hours;
    for (int i = 0; i < 24; i++)
        hours.push_back(std::string(i < 10 ? "0" : "") + utl::stringFromInt(i));

    std::vector<std::string> minutes;
    for (int i = 0; i < 60; i++)
        minutes.push_back(std::string(i < 10 ? "0" : "") + utl::stringFromInt(i));

    _timeFrom = new cvr::SubMenu("Start Time","Start-of-Range Timestamp");
    _timeFrom->setCallback(this);
    _powerMenu->addItem(_timeFrom);

    _timeTo = new cvr::SubMenu("End  Time","End-of-Range Timestamp");
    _timeTo->setCallback(this);
    _powerMenu->addItem(_timeTo);

    _yearText = new cvr::MenuText("Year:");
    _yearText->setCallback(this);
    _monthText = new cvr::MenuText("Month:");
    _monthText->setCallback(this);
    _dayText = new cvr::MenuText("Day:");
    _dayText->setCallback(this);
    _hourText = new cvr::MenuText("Hour:");
    _hourText->setCallback(this);
    _minuteText = new cvr::MenuText("Minute:");
    _minuteText->setCallback(this);

    _timeFrom->addItem(_yearText);
    _yearFrom = new cvr::MenuList();
    _yearFrom->setCallback(this);
    _yearFrom->setValues(years);
    _timeFrom->addItem(_yearFrom);

    _timeFrom->addItem(_monthText);
    _monthFrom = new cvr::MenuList();
    _monthFrom->setCallback(this);
    _monthFrom->setValues(months);
    _timeFrom->addItem(_monthFrom);

    _timeFrom->addItem(_dayText);
    _dayFrom = new cvr::MenuList();
    _dayFrom->setCallback(this);
    _dayFrom->setValues(days);
    _timeFrom->addItem(_dayFrom);    

    _timeFrom->addItem(_hourText);
    _hourFrom = new cvr::MenuList();
    _hourFrom->setCallback(this);
    _hourFrom->setValues(hours);
    _timeFrom->addItem(_hourFrom);

    _timeFrom->addItem(_minuteText);
    _minuteFrom = new cvr::MenuList();
    _minuteFrom->setCallback(this);
    _minuteFrom->setValues(minutes);
    _timeFrom->addItem(_minuteFrom);

    _timeTo->addItem(_yearText);
    _yearTo = new cvr::MenuList();
    _yearTo->setCallback(this);
    _yearTo->setValues(years);
    _timeTo->addItem(_yearTo);

    _timeTo->addItem(_monthText);
    _monthTo = new cvr::MenuList();
    _monthTo->setCallback(this);
    _monthTo->setValues(months);
    _timeTo->addItem(_monthTo);

    _timeTo->addItem(_dayText);
    _dayTo = new cvr::MenuList();
    _dayTo->setCallback(this);
    _dayTo->setValues(days);
    _timeTo->addItem(_dayTo);    

    _timeTo->addItem(_hourText);
    _hourTo = new cvr::MenuList();
    _hourTo->setCallback(this);
    _hourTo->setValues(hours);
    _timeTo->addItem(_hourTo);

    _timeTo->addItem(_minuteText);
    _minuteTo = new cvr::MenuList();
    _minuteTo->setCallback(this);
    _minuteTo->setValues(minutes);
    _timeTo->addItem(_minuteTo);
}

void GreenLight::animatePower()
{
    std::set< Component * >::iterator sit;
    for (sit = _components.begin(); sit != _components.end(); sit++)
    {
        if ( (*sit)-> animating ){
            if( ++((*sit)->animationPosition) > 100  )
            { // End Animation
                (*sit) -> animationPosition = 0;
                (*sit) -> animating = false;
            }else
            {
//              std::cout << "Animation Position: " << (*sit) -> animationPosition << std::endl;
                setPowerColors(true); // update texture.
            }
        }
    }
}
