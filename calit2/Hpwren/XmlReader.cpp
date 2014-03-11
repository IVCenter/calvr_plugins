#include "XmlReader.h"

#include<iostream>
#include <sstream>

#include <mxml.h>
#include <cstdio>
#include <stack>
#include <algorithm>

#include <mxml.h>

XmlReader::XmlReader(std::string file)
{
    _configRootList = std::vector<mxml_node_t *>();
    _debugOutput = false;
    loadFile(file);
}

XmlReader::~XmlReader()
{
    for(int i = 0; i < _configRootList.size(); i++)
    {
        mxmlDelete(_configRootList[i]);
        _configRootList.clear();
    }
}

bool XmlReader::loadFile(std::string file)
{

    FILE *fp;
    mxml_node_t * tree;

    fp = fopen(file.c_str(), "r");
    if(fp == NULL)
    {
        std::cerr << "Unable to open file: " << file << std::endl;
        return false;
    }
    tree = mxmlLoadFile(NULL, fp, MXML_TEXT_CALLBACK);
    fclose(fp);

    if(tree == NULL)
    {
        std::cerr << "Unable to parse XML file: " << file << std::endl;
        return false;
    }

    _configRootList.push_back(tree);

    return true;
}

std::string XmlReader::getEntry(std::string path, std::string def,
                                    bool * found)
{
    return getEntry("value", path, def, found);
}

std::string XmlReader::getEntry(std::string attribute, std::string path,
                                    std::string def, bool * found)
{
    if(path.empty())
    {
        if(found)
        {
            *found = false;
        }
        return def;
    }

    for(int i = 0; i < _configRootList.size(); i++)
    {
        std::string pathFrag;
        std::string pathRemainder = path;
        mxml_node_t * xmlNode = _configRootList[i]->child;

        size_t location = pathRemainder.find_first_of('.');
        if(location == std::string::npos)
        {
            pathFrag = pathRemainder;
            pathRemainder = "";
        }
        else
        {
            pathFrag = pathRemainder.substr(0, location);
            if(location + 1 < pathRemainder.size())
            {
                pathRemainder = pathRemainder.substr(location + 1);
            }
            else
            {
                pathRemainder = "";
            }
        }

        //std::cerr << "Looking for fragment: " << pathFrag << std::endl;
        //std::cerr << "with remainder: " << pathRemainder << std::endl;

        std::stack<std::pair<mxml_node_t *,std::string> > parentStack;

        do
        {
            if(!parentStack.empty())
            {
                xmlNode = parentStack.top().first->next;
                if(!pathRemainder.empty())
                {
                    pathRemainder = pathFrag + "." + pathRemainder;
                }
                else
                {
                    pathRemainder = pathFrag;
                }
                pathFrag = parentStack.top().second;
                parentStack.pop();
            }
            while(xmlNode)
            {
                if(xmlNode->type != MXML_ELEMENT)
                {
                    /*std::cerr << "Not elememnt node. type: " << xmlNode->type << std::endl;
                     if(xmlNode->type == MXML_OPAQUE)
                     {
                     std::cerr << "Opaque node value: " << xmlNode->value.opaque << std::endl;
                     }
                     if(xmlNode->type == MXML_TEXT)
                     {
                     std::cerr << "Text node value: " << xmlNode->value.text.string << std::endl;
                     }*/
                    xmlNode = xmlNode->next;
                    continue;
                }
                std::string nodeName = xmlNode->value.element.name;
                const char * nameAtt = mxmlElementGetAttr(xmlNode, "name");
                std::string suffix = nameAtt ? nameAtt : "";

                //std::cerr << "Looking at node: " << nodeName << " with suffix " << suffix << std::endl;

                location = pathFrag.find_first_of(':');
                if((location == std::string::npos && pathFrag == nodeName)
                        || (location != std::string::npos && pathFrag
                                == nodeName + ":" + suffix))
                {
                    //std::cerr << "Found Fragment." << std::endl;
                    if(pathRemainder.empty())
                    {
                        //found node
                        const char * attr =
                                mxmlElementGetAttr(xmlNode, attribute.c_str());
                        if(attr)
                        {
                            if(found)
                            {
                                *found = true;
                            }
                            if(_debugOutput)
                            {
                                std::cerr << "Path: " << path << " Attr: "
                                        << attribute << " value: " << attr
                                        << std::endl;
                            }
                            return attr;
                        }
                        else
                        {
                            /*if(found)
                             {
                             *found = false;
                             }
                             std::cerr << "Path: " << path << " Attr: " << attribute << " value: " << def << " (default)" << std::endl;
                             return def;*/
                            xmlNode = xmlNode->next;
                        }
                    }
                    else
                    {
                        parentStack.push(
                                         std::pair<mxml_node_t *,std::string>(
                                                                              xmlNode,
                                                                              pathFrag));
                        location = pathRemainder.find_first_of('.');
                        if(location == std::string::npos)
                        {
                            pathFrag = pathRemainder;
                            pathRemainder = "";
                        }
                        else
                        {
                            pathFrag = pathRemainder.substr(0, location);
                            if(location + 1 < pathRemainder.size())
                            {
                                pathRemainder = pathRemainder.substr(location
                                        + 1);
                            }
                            else
                            {
                                pathRemainder = "";
                            }
                        }
                        //std::cerr << "Looking for fragment: " << pathFrag << std::endl;
                        //std::cerr << "with remainder: " << pathRemainder << std::endl;
                        xmlNode = xmlNode->child;
                    }
                }
                else
                {
                    xmlNode = xmlNode->next;
                }
            }
        } while(!parentStack.empty());
    }
    if(found)
    {
        *found = false;
    }
    if(_debugOutput)
    {
        std::cerr << "Path: " << path << " Attr: " << attribute << " value: "
                << def << " (default)" << std::endl;
    }
    return def;
}

float XmlReader::getFloat(std::string path, float def, bool * found)
{
    return getFloat("value", path, def, found);
}

float XmlReader::getFloat(std::string attribute, std::string path,
                              float def, bool * found)
{
    bool hasEntry = false;
    std::stringstream ss;
    ss << def;
    std::string result = getEntry(attribute, path, ss.str(), &hasEntry);
    if(hasEntry)
    {
        if(found)
        {
            *found = true;
        }
        return atof(result.c_str());
    }
    if(found)
    {
        *found = false;
    }
    return def;
}

double XmlReader::getDouble(std::string path, double def, bool * found)
{
    return getDouble("value", path, def, found);
}

double XmlReader::getDouble(std::string attribute, std::string path,
                              double def, bool * found)
{
    bool hasEntry = false;
    std::stringstream ss;
    ss << def;
    std::string result = getEntry(attribute, path, ss.str(), &hasEntry);
    if(hasEntry)
    {
        if(found)
        {
            *found = true;
        }
        return atof(result.c_str());
    }
    if(found)
    {
        *found = false;
    }
    return def;
}

int XmlReader::getInt(std::string path, int def, bool * found)
{
    return getInt("value", path, def, found);
}

int XmlReader::getInt(std::string attribute, std::string path, int def,
                          bool * found)
{
    bool hasEntry = false;
    std::stringstream ss;
    ss << def;
    std::string result = getEntry(attribute, path, ss.str(), &hasEntry);
    if(hasEntry)
    {
        if(found)
        {
            *found = true;
        }
        return atoi(result.c_str());
    }
    if(found)
    {
        *found = false;
    }
    return def;
}

bool XmlReader::getBool(std::string path, bool def, bool * found)
{
    return getBool("value", path, def, found);
}

bool XmlReader::getBool(std::string attribute, std::string path, bool def,
                            bool * found)
{
    bool hasEntry = false;
    std::stringstream ss;
    ss << def;
    std::string result = getEntry(attribute, path, ss.str(), &hasEntry);
    if(hasEntry)
    {
        if(found)
        {
            *found = true;
        }
        std::transform(result.begin(), result.end(), result.begin(), ::tolower);
        if(result == "on" || result == "true")
        {
            return true;
        }
        else
        {
            return false;
        }
    }
    if(found)
    {
        *found = false;
    }
    return def;
}

osg::Vec3 XmlReader::getVec3(std::string path, osg::Vec3 def, bool * found)
{
    return getVec3("x","y","z",path,def,found);
}

osg::Vec3 XmlReader::getVec3(std::string attributeX, std::string attributeY, 
	std::string attributeZ, std::string path, osg::Vec3 def,
	bool * found)
{
    bool hasEntry = false;
    bool isFound;

    osg::Vec3 result;
    result.x() = getFloat(attributeX,path,def.x(),&isFound);
    if(isFound)
    {
	hasEntry = true;
    }
    result.y() = getFloat(attributeY,path,def.y(),&isFound);
    if(isFound)
    {
	hasEntry = true;
    }
    result.z() = getFloat(attributeZ,path,def.z(),&isFound);
    if(isFound)
    {
	hasEntry = true;
    }

    if(found)
    {
	*found = hasEntry;
    }
    return result;
}

osg::Vec4 XmlReader::getVec4(std::string path, osg::Vec4 def, 
	bool * found)
{
    return getVec4("x","y","z","w",path,def,found);
}

osg::Vec4 XmlReader::getVec4(std::string attributeX, std::string attributeY, 
	std::string attributeZ, std::string attributeW, std::string path, 
	osg::Vec4 def, bool * found)
{
    bool hasEntry = false;
    bool isFound;

    osg::Vec4 result;
    result.x() = getFloat(attributeX,path,def.x(),&isFound);
    if(isFound)
    {
	hasEntry = true;
    }
    result.y() = getFloat(attributeY,path,def.y(),&isFound);
    if(isFound)
    {
	hasEntry = true;
    }
    result.z() = getFloat(attributeZ,path,def.z(),&isFound);
    if(isFound)
    {
	hasEntry = true;
    }
    result.w() = getFloat(attributeW,path,def.w(),&isFound);
    if(isFound)
    {
	hasEntry = true;
    }

    if(found)
    {
	*found = hasEntry;
    }
    return result;
}

osg::Vec3d XmlReader::getVec3d(std::string path, osg::Vec3d def, bool * found)
{
    return getVec3d("x","y","z",path,def,found);
}

osg::Vec3d XmlReader::getVec3d(std::string attributeX, std::string attributeY, 
	std::string attributeZ, std::string path, osg::Vec3d def,
	bool * found)
{
    bool hasEntry = false;
    bool isFound;

    osg::Vec3d result;
    result.x() = getDouble(attributeX,path,def.x(),&isFound);
    if(isFound)
    {
	hasEntry = true;
    }
    result.y() = getDouble(attributeY,path,def.y(),&isFound);
    if(isFound)
    {
	hasEntry = true;
    }
    result.z() = getDouble(attributeZ,path,def.z(),&isFound);
    if(isFound)
    {
	hasEntry = true;
    }

    if(found)
    {
	*found = hasEntry;
    }
    return result;
}

osg::Vec4d XmlReader::getVec4d(std::string path, osg::Vec4d def, 
	bool * found)
{
    return getVec4d("x","y","z","w",path,def,found);
}

osg::Vec4d XmlReader::getVec4d(std::string attributeX, std::string attributeY, 
	std::string attributeZ, std::string attributeW, std::string path, 
	osg::Vec4d def, bool * found)
{
    bool hasEntry = false;
    bool isFound;

    osg::Vec4d result;
    result.x() = getDouble(attributeX,path,def.x(),&isFound);
    if(isFound)
    {
	hasEntry = true;
    }
    result.y() = getDouble(attributeY,path,def.y(),&isFound);
    if(isFound)
    {
	hasEntry = true;
    }
    result.z() = getDouble(attributeZ,path,def.z(),&isFound);
    if(isFound)
    {
	hasEntry = true;
    }
    result.w() = getDouble(attributeW,path,def.w(),&isFound);
    if(isFound)
    {
	hasEntry = true;
    }

    if(found)
    {
	*found = hasEntry;
    }
    return result;
}

osg::Vec4 XmlReader::getColor(std::string path, osg::Vec4 def, bool * found)
{
    return getVec4("r","g","b","a",path,def,found);
}

void XmlReader::getChildren(std::string path,
                                std::vector<std::string> & destList)
{
    if(path.empty())
    {
        return;
    }

    for(int i = 0; i < _configRootList.size(); i++)
    {
        std::string pathFrag;
        std::string pathRemainder = path;
        mxml_node_t * xmlNode = _configRootList[i]->child;

        size_t location = pathRemainder.find_first_of('.');
        if(location == std::string::npos)
        {
            pathFrag = pathRemainder;
            pathRemainder = "";
        }
        else
        {
            pathFrag = pathRemainder.substr(0, location);
            if(location + 1 < pathRemainder.size())
            {
                pathRemainder = pathRemainder.substr(location + 1);
            }
            else
            {
                pathRemainder = "";
            }
        }

        std::stack<std::pair<mxml_node_t *,std::string> > parentStack;

        do
        {
            if(!parentStack.empty())
            {
                xmlNode = parentStack.top().first->next;
                if(!pathRemainder.empty())
                {
                    pathRemainder = pathFrag + "." + pathRemainder;
                }
                else
                {
                    pathRemainder = pathFrag;
                }
                pathFrag = parentStack.top().second;
                parentStack.pop();
            }
            while(xmlNode)
            {
                if(xmlNode->type != MXML_ELEMENT)
                {
                    xmlNode = xmlNode->next;
                    continue;
                }
                std::string nodeName = xmlNode->value.element.name;
                const char * nameAtt = mxmlElementGetAttr(xmlNode, "name");
                std::string suffix = nameAtt ? nameAtt : "";

                //std::cerr << "Looking at node: " << nodeName << " with suffix " << suffix << std::endl;

                location = pathFrag.find_first_of(':');
                if((location == std::string::npos && pathFrag == nodeName)
                        || (location != std::string::npos && pathFrag
                                == nodeName + ":" + suffix))
                {
                    //std::cerr << "Found Fragment." << std::endl;
                    if(pathRemainder.empty())
                    {
                        if(xmlNode->child)
                        {
                            mxml_node_t * cnode = xmlNode->child;
                            while(cnode)
                            {
                                if(cnode->type != MXML_ELEMENT)
                                {
                                    cnode = cnode->next;
                                    continue;
                                }
                                // ignore comment tags
                                if(strncmp(cnode->value.element.name, "!--", 3))
                                {
                                    destList.push_back(
                                                       cnode->value.element.name);
                                }
                                cnode = cnode->next;
                            }
                        }
                        xmlNode = xmlNode->next;
                    }
                    else
                    {
                        parentStack.push(
                                         std::pair<mxml_node_t *,std::string>(
                                                                              xmlNode,
                                                                              pathFrag));
                        location = pathRemainder.find_first_of('.');
                        if(location == std::string::npos)
                        {
                            pathFrag = pathRemainder;
                            pathRemainder = "";
                        }
                        else
                        {
                            pathFrag = pathRemainder.substr(0, location);
                            if(location + 1 < pathRemainder.size())
                            {
                                pathRemainder = pathRemainder.substr(location
                                        + 1);
                            }
                            else
                            {
                                pathRemainder = "";
                            }
                        }
                        //std::cerr << "Looking for fragment: " << pathFrag << std::endl;
                        //std::cerr << "with remainder: " << pathRemainder << std::endl;
                        xmlNode = xmlNode->child;
                    }
                }
                else
                {
                    xmlNode = xmlNode->next;
                }
            }
        } while(!parentStack.empty());
    }
    return;
}
