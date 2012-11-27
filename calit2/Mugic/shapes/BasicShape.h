#ifndef _BASICSHAPE_
#define _BASICSHAPE_

#include <osg/Geometry>

#include <string>
#include <vector>
#include <OpenThreads/Mutex>

namespace SimpleShape 
{
    enum ShapeType {POINT, TRIANGLE, QUAD, RECTANGLE, CIRCLE, LINE, TEXT};
}
    
class BasicShape : public osg::Geometry
{
    public:        

        virtual void update(std::string command) = 0;
	    SimpleShape::ShapeType getType() { return _type; };
        std::string getName();
        bool isDirty();

        struct ShapeUpdateCallback : public osg::Drawable::UpdateCallback
        {
            virtual void update(osg::NodeVisitor*, osg::Drawable* drawable)
            {
                BasicShape* shape = dynamic_cast<BasicShape*> (drawable);
                shape->update();
            }
        };


    protected:
        BasicShape();
	    virtual ~BasicShape();
	    SimpleShape::ShapeType _type;
        osg::Vec3Array* _vertices;
        osg::Vec4Array* _colors;

        virtual void update() = 0;

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
        void setParameter(std::string varName, std::string& value);
        void setName(std::string);
};

#endif
