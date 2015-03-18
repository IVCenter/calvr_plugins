#ifndef _BASICSHAPE_
#define _BASICSHAPE_

#include <osg/Geometry>

#include <string>
#include <vector>
#include <OpenThreads/Mutex>
#include "../Type.h"
#include "../Vec3Type.h"
#include "../Vec4Type.h"
#include "../FloatType.h"
#include "../StringType.h"
#include "../BoolType.h"

namespace SimpleShape 
{
    enum ShapeType {POINT, TRIANGLE, QUAD, RECTANGLE, CIRCLE, LINE, SCALABLELINE, TEXT};
}

class BasicShape
{
    public:        

        virtual void update(std::string command) = 0;
        virtual osg::Geode* getParent() = 0;
        virtual osg::Drawable* asDrawable() = 0;
	    SimpleShape::ShapeType getType() { return _type; };
        std::string getName();
        bool isDirty();

    protected:
        BasicShape();
	    virtual ~BasicShape();
	    SimpleShape::ShapeType _type;

        virtual void update() = 0;

        // counter used to naming objects
        bool _dirty;
        std::string _name;
        mutable OpenThreads::Mutex _mutex;

        // map for local params
        std::map<std::string, Type* > _paramMapping;

        // add a param
        void createParameter(std::string paramName, Type *type);
        void setParameter(std::string paramName, std::string param);
        Type* getParameter(std::string paramName);

        void setName(std::string);
};

#endif
