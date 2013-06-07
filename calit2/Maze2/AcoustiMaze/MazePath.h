#ifndef MAZE_MAZEPATH_H
#define MAZE_MAZEPATH_H


#include <stdlib.h>
#include <math.h>
#include <vector>

#include "CoordMatrix.h"
#include "MazeSquare.h"


class MazePath 
{

public:
	int pl;
    std::vector<int> _dirs;
    std::vector<int> _bend;
    int _bends;

	MazePath(int x, int y);
	void reset();
	void compile();

    // TODO
/*    void draw(Graphics2D g,CoordMatrix m, MazeSquare qr, int w);*/

	static int pathLength;
//	int pts[100][2];// = new int[pathLength][2];
    std::vector< std::vector<int> > pts;

protected:               
	int** _chk;// = {{0,1},{1,0},{0,-1},{-1,0}};

	static const double volume = 0; //sabine-relevant volume of path
	double* filter;// = {1.,1.,1.,1.,1.,1.}; //savine-relevant filter of path
	
	int x1,x2,y1,y2;
    

};
    
#endif
