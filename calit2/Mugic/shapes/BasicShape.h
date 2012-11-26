#ifndef _BASICSHAPE_
#define _BASICSHAPE_

#include <osg/Geometry>

#include <string>
#include <vector>
#include <OpenThreads/Mutex>

enum SimpleShape {POINT, TRIANGLE, QUAD, RECTANGLE, CIRCLE, LINE};


class BasicShape : public osg::Geometry, public osg::Drawable::UpdateCallback
{
public:        

    virtual void update(std::string command) = 0;
    virtual void update(osg::NodeVisitor*, osg::Drawable* drawable) = 0;
	SimpleShape getType() { return _type; };
    std::string getName();
    bool isDirty();

protected:
    BasicShape();
	virtual ~BasicShape();
	SimpleShape _type;
    osg::Vec3Array* _vertices;
    osg::Vec4Array* _colors;

    // counter used to naming objects
    static unsigned int counter;
    bool _dirty;
    std::string _name;
    mutable OpenThreads::Mutex _mutex;

    // keep a map of parameters used (quick way to find out in update if data needs to be recomputed)
    // key var name, vector holds name of attributes mapping is applied to
    std::map<std::string, std::string > _localParams;

    // access the parameter
    void addParameter(std::string command, std::string param);
    void addLocalParam(std::string varName, std::string param);
    void setParameter(std::string varName, float& value);
    void setName(std::string);
};

#endif
