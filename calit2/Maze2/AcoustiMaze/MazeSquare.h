#ifndef MAZE_MAZESQUARE_H
#define MAZE_MAZESQUARE_H

#include "CoordMatrix.h"
#include "Pose.h"

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>

class AcoustiMaze;
class MazePath;

class MazeSquare 
{

protected:

	 int directNeighbors;//=0;
	 int directFrom;//=5;

	 double value;//=1; //what is the shortest path to the source from here?
	 bool hasSource;//=false; //is the source in this square?
	 bool hasObserver;//=false; //is the observer in this square?
	 bool isEntrance;//=false;

	 
	 std::vector< std::vector<double> > diffsrc;//={{0,0},{0,0},{0,0},{0,0}};
	 std::vector< std::vector<int> > outSq;//={{1,1},{-1,1},{-1,-1},{1,-1}};
	 
	 
	 double p1x,p1y,p2x,p2y,p3x,p3y,dx,dy;
	 double pdis,dis1,dis2,dis3;
	 
	 double gain;
	 
	 double strx,stry;
	 double stox,stoy;
	 
	 int topoNr;//=0;
	 std::vector<double> inGain;//={0,0,0,0};
	 
	 
	 double Nu,Eu,Su,Wu;
	 double No,Eo,So,Wo;
	 double lN,lE,lS,lW;
	 std::vector<double> inDelay;//[4];//={0,0,0,0};
	 

	 double obsAngl;//=0;
	 double direct;//=0;
    

     std::vector< std::vector<double> > inMaterial;
/*	 double inMaterial[6][4];=
	 			{ //array storing the accumulated material filter of the path
	 			  {0.6,0.3,0.1,0.1,0.1,0.1},
	                {0.6,0.3,0.1,0.1,0.1,0.1},
	                {0.6,0.3,0.1,0.1,0.1,0.1},
	                {0.6,0.3,0.1,0.1,0.1,0.1}
	 			};*/
	 int x,y; //grid X of this square
	 
	 double wx;  //world X of this square
	 double wy;  //world Y of this square
	 double ws;
	 double wbx;  //bounding world X of this square
	 double wby;  //bounding world Y of this square
	 double wbs;
	 
	 double cx;
	 double cy;

	 
	 double _X,_Y;
	 double X_,Y_;
	 
	 int gx,gy,gs;


public:	

    MazeSquare(const AcoustiMaze * maz, int arx, int ary);
    void reset();
    void compile();
    void wallAngles(Pose pos);
    void eval(Pose pos, int layer);
    void getDelays();
    double norm(double px, double py);
    void dodiffsrc();
    void checkEntrance(Pose pos, int layer);
    void notEntrance();
    void gainer();
    double observerAngle(Pose pos);

    // TODO
/*    void drawplane(Graphics2D g, CoordMatrix m);
    void drawwalls(Graphics2D g, CoordMatrix m);
    void drawpaths(Graphics2D g, CoordMatrix m);
*/
    
    AcoustiMaze *maz;
    bool seesDirect;//=false; //is the direct sound audible in this square?
    std::vector<std::string> walls;//={"void","void","void","void"}; //materials for walls N,W,S,E
    std::vector<MazePath*> inPath;// = new MazePath[4];
    std::vector<bool> reached;// ={false,false,false,false}; //array storing if a square wall (NESW) has been reached by a sound path
};

#endif

