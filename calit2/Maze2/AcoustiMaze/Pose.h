#ifndef MAZE_POSE_H
#define MAZE_POSE_H


#include <math.h>

class Pose 
{

protected:

	
public:	
	//Color indcol=Color.red;
    Pose();
	Pose(int i);

    // TODO
	void set(/*Atom[] args*/);
	//void paint(Graphics2D g, CoordMatrix m);
	void set(float x,float y,float z,float yaw, float pitch,float roll);

	static double gr, gn;//=20,gn=5;
	double x,y,z,yaw,pitch,roll;
	int index;
	int px,py,pnx,pny,ps,pns;

};

#endif
