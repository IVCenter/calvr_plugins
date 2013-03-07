#ifndef _MODELSHAPE_
#define _MODELSHAPE_

#include "GeometryShape.h"

#include <osg/Geometry>
#include <osg/Node>
#include <osg/MatrixTransform>

class ModelShape : public GeometryShape
{
public:
	ModelShape(std::string command = "", std::string name = "");
	virtual ~ModelShape();
	void update(std::string);
	osg::MatrixTransform* getMatrixParent();
	osg::Node* getModelNode();

protected:
	ModelShape();
	void setModel(std::string);
	void setShaders(std::string, std::string);
	void update();
	osg::Node* _modelNode; //actual model node


};

#endif
