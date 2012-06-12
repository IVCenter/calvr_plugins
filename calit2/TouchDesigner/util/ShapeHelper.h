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

<<<<<<< HEAD

=======
>>>>>>> e7aefd8ba4f234f3d166f0a934e5cf6ea045a343
	// get the number of shapes managed by this shape helper
	int getShapeCount();

	Geode * getGeode();

	bool processedAll;

<<<<<<< HEAD

	bool debugOn;

=======
	bool debugOn;
>>>>>>> e7aefd8ba4f234f3d166f0a934e5cf6ea045a343

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
<<<<<<< HEAD

=======
>>>>>>> e7aefd8ba4f234f3d166f0a934e5cf6ea045a343


	// the geode we're managing
	Geode * geode;

	double random();
	double random(double,double);


	// shape generating
	BasicShape* genShape(char*);
	BasicShape* genShape();
	
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

	void updateTriangleC(int);

 
	void updateCircle(CircleShape*);
	void updateRectP(RectShape*);
	void updateRectC(RectShape*);
	void updateTriangleP(TriangleShape*);
	void updateTriangleC(TriangleShape*);

	// a cleaner update method
	void updateShape(int);
<<<<<<< HEAD
  void updateShape(BasicShape*);
  void updateShape();
  void updateShape(char*);
=======
>>>>>>> e7aefd8ba4f234f3d166f0a934e5cf6ea045a343

	// other handlers
	int shapeCount;
	int updateType;

};

#endif
