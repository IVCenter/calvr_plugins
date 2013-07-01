#ifndef ACOUSTIMAZE_PLUGIN_H
#define ACOUSTIMAZE_PLUGIN_H

#include <cvrKernel/CVRPlugin.h>
#include <cvrKernel/PluginHelper.h>
#include <cvrMenu/SubMenu.h>
#include <cvrMenu/MenuRangeValue.h>
#include <cvrMenu/MenuCheckbox.h>
#include <cvrMenu/MenuButton.h>



#include "MazeSquare.h"
#include "MazePath.h"
#include <stdio.h>
#include <stdlib.h>


class AcoustiMaze : public cvr::MenuCallback, public cvr::CVRPlugin
{

public:
    AcoustiMaze(/*MaxObject mo*/);
    virtual ~AcoustiMaze();   
    //void paint(Graphics2D g, CoordMatrix m);

	bool init();
    void menuCallback(cvr::MenuItem * item);
    void preFrame();
    bool processEvent(cvr::InteractionEvent * event);

    void soundUpdate(MazeSquare *sq);
    void entered(MazeSquare sq);
    void FloorPlan(int arguments[]);
    void eval(Pose pos, int layer);
    void reset();
    void check_test(int squareX, int squareY, int enterW, int gen);
    void check(int squareX, int squareY, int enterW, int gen); //check the square, entering from wall enterW, step gen.
    bool isStraightLine(std::vector< std::vector<int> >& line, int l);      
    
    // TODO

    int xo,yo,xs,ys;
    double gunit;
    double srcX,srcY,srcR; //world coordiantes of source
    double obsX,obsY,obsR; //world coordinates of observer, X,Y,P (angle)
    double srcLX,srcLY; // coordinates of source within current sqare.
    double obsLX,obsLY;
    bool debug;// = false;
    std::vector< std::vector<int> > chk;//[2][4];//={{0,1},{1,0},{0,-1},{-1,0}};
    std::vector< std::vector<MazeSquare*> > maze; //= new Array(new Array(new Square(1,1))); //two dimensional array of squares

protected:

    int xsum,ysum;
    bool isInPathAlready;
    int intVar;
    int opposite[4];// ={3,2,1,0};
    int maxlen;//=100;

    cvr::SubMenu * _mainMenu;
    cvr::MenuButton * _loadBtn, * _clearBtn;

    // TODO
/*    MaxObject max;
    Atom[] arg=new Atom[3];
    Atom[] args=new Atom[4];
    Atom[] args2=new Atom[1];
    Atom[] args3=new Atom[4];
    Atom[] args8=new Atom[8];   
*/
};


#endif
