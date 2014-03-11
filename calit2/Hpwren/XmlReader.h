#ifndef XML_READER_H
#define XML_READER_H

#include <osg/Vec3>
#include <osg/Vec4>
#include <osg/Vec3d>
#include <osg/Vec4d>

#include <string>
#include <map>
#include <vector>

struct mxml_node_s;
typedef struct mxml_node_s mxml_node_t;

class XmlReader
{
    public:
        XmlReader(std::string file);


        std::string getEntry(std::string path, std::string def = "",
                                    bool * found = NULL);

        std::string getEntry(std::string attribute, std::string path,
                                    std::string def = "", bool * found = NULL);

        float getFloat(std::string path, float def = 0.0, bool * found =
                NULL);

        float getFloat(std::string attribute, std::string path,
                             float def = 0.0, bool * found = NULL);
        
        double getDouble(std::string path, double def = 0.0, bool * found =
                NULL);

        double getDouble(std::string attribute, std::string path,
                              double def = 0.0, bool * found = NULL);

        int getInt(std::string path, int def = 0, bool * found = NULL);

        int getInt(std::string attribute, std::string path, int def = 0,
                          bool * found = NULL);

        bool getBool(std::string path, bool def = false, bool * found =
                NULL);

        bool getBool(std::string attribute, std::string path, bool def =
                false, bool * found = NULL);

        osg::Vec3 getVec3(std::string path, osg::Vec3 def = osg::Vec3(0,0,0), 
                bool * found = NULL);

        osg::Vec3 getVec3(std::string attributeX, std::string attributeY, 
                std::string attributeZ, std::string path, osg::Vec3 def = osg::Vec3(0,0,0),
                bool * found = NULL);

        osg::Vec4 getVec4(std::string path, osg::Vec4 def = osg::Vec4(0,0,0,1), 
                bool * found = NULL);

        osg::Vec4 getVec4(std::string attributeX, std::string attributeY, 
                std::string attributeZ, std::string attributeW, std::string path, 
                osg::Vec4 def = osg::Vec4(0,0,0,1), bool * found = NULL);

        osg::Vec3d getVec3d(std::string path, osg::Vec3d def = osg::Vec3d(0,0,0), 
                bool * found = NULL);

        osg::Vec3d getVec3d(std::string attributeX, std::string attributeY, 
                std::string attributeZ, std::string path, osg::Vec3d def = osg::Vec3d(0,0,0),
                bool * found = NULL);

        osg::Vec4d getVec4d(std::string path, osg::Vec4d def = osg::Vec4d(0,0,0,1), 
                bool * found = NULL);

        osg::Vec4d getVec4d(std::string attributeX, std::string attributeY, 
                std::string attributeZ, std::string attributeW, std::string path, 
                osg::Vec4d def = osg::Vec4d(0,0,0,1), bool * found = NULL);

        osg::Vec4 getColor(std::string path, osg::Vec4 def = osg::Vec4(1,1,1,1), 
                bool * found = NULL);

        void getChildren(std::string path,
                                std::vector<std::string> & destList);

    protected:
        XmlReader();
        virtual ~XmlReader();
        bool loadFile(std::string file);

        std::vector<mxml_node_t *> _configRootList; ///< List of the roots of all loaded xml files

        bool _debugOutput; ///< should config debug statements be printed
};
#endif
