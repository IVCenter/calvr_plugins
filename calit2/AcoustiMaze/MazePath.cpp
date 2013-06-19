
#include "MazePath.h"

int MazePath::pathLength = 100;;

MazePath::MazePath(int x, int y)
{
    pl = 0;
    _bends = 0;

    _chk[0][0] = 0;
    _chk[0][1] = 1;
    _chk[1][0] = 1;
    _chk[1][1] = 0;
    _chk[2][0] = 0;
    _chk[2][1] = -1;
    _chk[3][0] = -1;
    _chk[3][1] = 0;


    //pts = new int[pathLength][2];

    filter[0] = 1;
    filter[1] = 1;
    filter[2] = 1;
    filter[3] = 1;
    filter[4] = 1;
    filter[5] = 1;
}

void MazePath::reset()
{
    pl = 0;
    for (int i = 0; i < pathLength; i++)
    {
        pts[i][0] = 0;
        pts[i][1] = 0;
    }
}

void MazePath::compile()
{
    _bend[0] = 0;
    _dirs[0] = 0;

    for (int i=0; i<pl-2; i++)
    {
        for (int j=0; j<4; j++)
        {
            if ((pts[i+1][0] - pts[i][0] == _chk[j][0]) && (pts[i+1][1] - pts[i][1] == _chk[j][1])) 
            {
                _dirs[i]=j;
            }
        }
        if (i>0)
        {
            _bend[i] = _dirs[i] - _dirs[i-1];
            if (_bend[i]>1) 
            {
                _bend[i]=-1;
            }
            if (_bend[i]<-1) 
            {
                _bend[i]=1;
            }
            _bends += abs(_bend[i]);
        }
    }
}

// TODO
/*
void MazePath::draw(Graphics2D g,CoordMatrix m, MazeSquare qr, int w)
{
    int o,p,q,r;
    o = pts[0][0];
    p = pts[0][1];
    q = pts[0][0];
    r = pts[0][1];
    //.print("path "+w+" ");
    g.setColor(new Color((int)(255*(sin(w)/2.0+0.5)),(int)(255*(cos(w)/2.0+0.5)),(int)(255*((-.sin(w)/2.0)+0.5)))); //determine color
    for (int i=0;i<pl-1;i++) //go through path of observer-square (obsmx/obsmy maze coords)
    {
        double ox,oy;
        o = pts[i][0];
        p = pts[i][1];
        q = pts[i+1][0];
        r = pts[i+1][1];
        //.print(""+o+"/"+p+"-");
        ox = 0.01*w;
        oy = 0.01*w;
        x1 = m.xi(qr.maz.maze[o][p].wx + qr.maz.maze[o][p].ws*0.5 + ox);
        y1 = m.yi(qr.maz.maze[o][p].wy + qr.maz.maze[o][p].ws*0.5 + oy);
        x2 = m.xi(qr.maz.maze[q][r].wx + qr.maz.maze[q][r].ws*0.5 + ox);
        y2 = m.yi(qr.maz.maze[q][r].wy + qr.maz.maze[q][r].ws*0.5 + oy);
        g.drawLine(x1,y1,x2,y2); 
       //g.drawLine((int)(400*0.9*(qr.maz.maze[o][p].x)*2.0/qr.maz.xs+ox),(int)(400*0.9*(qr.maz.maze[o][p].y)*2.0/qr.maz.ys+oy),(int)(400*0.9*(qr.maz.maze[q][r].x)*2.0/qr.maz.xs+ox), (int)(400*0.9*(qr.maz.maze[q][r].y)*2.0/qr.maz.ys+oy));
    }
   // System.out.println(""+q+"/"+r+"-");
}
 */              
