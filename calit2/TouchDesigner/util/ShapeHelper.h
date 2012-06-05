#ifndef _ShapeHelper_
#define _ShapeHelper_

#include <osg/MatrixTransform>
#include <osg/Vec3d>
#include <osg/Vec4>

#include "vvtokenizer.h"
#include "../shapes/BasicShape.h"
#include "../shapes/TriangleShape.h"
#include "../shapes/CircleShape.h"
#include "../shapes/RectShape.h"

#include <string>
#include <vector>
#include <math.h>
#include <limits>

using namespace std;
using namespace osg;


class ShapeHelper  
{
public:        
	ShapeHelper(Geode *);
	ShapeHelper();


	virtual ~ShapeHelper();

	// returns a shape randomly generated
	BasicShape * genRandomShape();


	// determines whether to add or update a shape to current geode
	// this method is the ideal way to handle most data packets
	void processData(char*);

	// get the number of shapes managed by this shape helper
	int getShapeCount();

	Geode * getGeode();

	bool processedAll;

	bool debugOn;

protected:

	// getting parameters from the data
	double getDoubleParam(string);
	int getIntParam(string);
	char* getCharParam(string);
	double getDoubleParamC(char*);
	int getIntParamC(char*);
	char* getCharParamC(char*);	

	vvTokenizer * tok;

	
	bool genAll;


	// the geode we're managing
	Geode * geode;

	double random();
	double random(double,double);


	// shape generating
	CircleShape * genCircle();
	RectShape * genRectP();
	RectShape * genRectC();
	TriangleShape * genTriangleP();
	TriangleShape * genTriangleC();


	// shape updating
	void updateCircle();
	void updateCircle(int);
	void updateRectP();
	void updateRectC();
	void updateTriangleP();
	void updateTriangleC();
	void updateTriangleC(int);

	void updateCircle(CircleShape*);
	void updateRectP(RectShape*);
	void updateRectC(RectShape*);
	void updateTriangleP(TriangleShape*);
	void updateTriangleC(TriangleShape*);

	// a cleaner update method
	void updateShape(int);

	// other handlers
	int shapeCount;
	int updateType;

};

#endif
