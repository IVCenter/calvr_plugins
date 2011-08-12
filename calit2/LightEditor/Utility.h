#include <string>
#include <osg/Shader>

namespace utl
{
    int intFromString(std::string str);
    std::string stringFromInt(int i);
    void downloadFile(std::string downloadUrl, std::string fileName, std::string &content);
    bool loadShaderSource(osg::Shader * obj, const std::string& fileName);
};
