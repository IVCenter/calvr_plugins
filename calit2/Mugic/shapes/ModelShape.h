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
	void setModel(std::string);;
	void update();
	std::string _model_name; //name of model file
	osg::Node* _modelNode; //actual model node


};

#endif
