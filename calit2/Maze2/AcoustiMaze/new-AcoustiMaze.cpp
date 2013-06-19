#include "AcoustiMaze.h"


CVRPLUGIN(AcoustiMaze)



// TODO
AcoustiMaze::AcoustiMaze(/*MaxObject mo*/)
{
    debug = false;
    xsum=0;
    ysum=0;
    gunit=4.8;
    opposite[0] = 3;
    opposite[1] = 2;
    opposite[2] = 1;
    opposite[3] = 0;

    srcX=0;
    srcY=0;
    srcR=0;
    obsX=1;
    obsY=1;
    obsR=0;

    srcLX=0;
    srcLY=0;
    obsLX=0;
    obsLY=0;

    chk[0][0] = 0;
    chk[0][1] = 1;
    chk[1][0] = 1;
    chk[1][1] = 0;
    chk[2][0] = 0;
    chk[2][1] = -1;
    chk[3][0] = -1;
    chk[3][1] = 0;


    maxlen=100;

    //max=mo;
    //maze = new MazeSquare[20][20];
    for(int i=0;i<20;i++)
    {
        for(int j=0;j<20;j++)
        {
            maze[i][j] = new MazeSquare(this, i, j);
        }
    }
    
}

~AcoustiMaze::AcoustiMaze()
{

}

bool AcoustiMaze::AcoustiMaze::init()
{
    return true;
}

void AcoustiMaze::menuCallback(cvr::MenuItem * item)
{
    if (item == _loadButton)
    {
        std::string filename = "/home/jgoss/EasyModel.MAZ";//_dataDir + "EasyMaze.MAZ";
        ifstream inFile;
        inFile.open(filename.c_str());
        if (!inFile) 
        {
            cout << "Unable to open maze model file " << filename << endl;
            return;
        }

        /* keep iterating through model file searching for all valid entries */
        string entryNameStr;
        while (inFile >> entryNameStr)
        {
            // Floorplan Close_Wall Open_Wall Door
            if (!strcmp(entryNameStr.c_str(), "Floorplan"))
            {
                int args[4];
                inFile >> args[0];
                inFile >> args[1];
                inFile >> args[2];
                inFile >> args[3];
                FloorPlan(args);
            }
            else if (!strcmp(entryNameStr.c_str(), "Close_Wall"))
            {
                int sqx, sqy, wallNum;
                std::string wall;

                inFile >> sqx;
                inFile >> sqy;
                inFile >> wall;

                if (!strcmp(wall.c_str(), "N"))
                    wall = 0;
                if (!strcmp(wall.c_str(), "E"))
                    wall = 1;
                if (!strcmp(wall.c_str(), "S"))
                    wall = 2;
                if (!strcmp(wall.c_str(), "W"))
                    wall = 3;

                maze[sqx - xo][sqy - yo].walls[wall] = "Close_Wall";
            }
            else if (!strcmp(entryNameStr.c_str(), "Open_Wall"))
            {
                int sqx, sqy, wallNum;
                std::string wall;

                inFile >> sqx;
                inFile >> sqy;
                inFile >> wall;

                if (!strcmp(wall.c_str(), "N"))
                    wall = 0;
                if (!strcmp(wall.c_str(), "E"))
                    wall = 1;
                if (!strcmp(wall.c_str(), "S"))
                    wall = 2;
                if (!strcmp(wall.c_str(), "W"))
                    wall = 3;

                maze[sqx - xo][sqy - yo].walls[wall] = "Open_Wall";
            }
            else if (!strcmp(entryNameStr.c_str(), "Door"))
            {
                int sqx, sqy, wallNum;
                std::string wall, color;

                inFile >> sqx;
                inFile >> sqy;
                inFile >> wall;
                inFile >> color;

                if (!strcmp(wall.c_str(), "N"))
                    wall = 0;
                if (!strcmp(wall.c_str(), "E"))
                    wall = 1;
                if (!strcmp(wall.c_str(), "S"))
                    wall = 2;
                if (!strcmp(wall.c_str(), "W"))
                    wall = 3;

                maze[sqx - xo][sqy - yo].walls[wall] = "Open_Wall";
            }
        }
    }
}

void AcoustiMaze::preFrame()
{

}

bool AcoustiMaze::processEvent(cvr::InteractionEvent * event)
{

}
   
void AcoustiMaze::soundUpdate(MazeSquare sq)
{   
    // TODO
    /*
    args8[0]=Atom.newAtom(sq.diffsrc[0][0]);
    args8[1]=Atom.newAtom(sq.diffsrc[0][1]);
    args8[2]=Atom.newAtom(sq.diffsrc[1][0]);
    args8[3]=Atom.newAtom(sq.diffsrc[1][1]);
    args8[4]=Atom.newAtom(sq.diffsrc[2][0]);
    args8[5]=Atom.newAtom(sq.diffsrc[2][1]);
    args8[6]=Atom.newAtom(sq.diffsrc[3][0]);
    args8[7]=Atom.newAtom(sq.diffsrc[3][1]);
    max.outlet(0,"/"+sq.topoNr+"/diffsrc", args8);

    args2[0]=Atom.newAtom((sq.inPath[0].pl-1)*sq.ws);
    max.outlet(0,"/"+sq.topoNr+"/N/distance", args2);
    args2[0]=Atom.newAtom((sq.inPath[1].pl-1)*sq.ws);
    max.outlet(0,"/"+sq.topoNr+"/E/distance", args2);
    args2[0]=Atom.newAtom((sq.inPath[2].pl-1)*sq.ws);
    max.outlet(0,"/"+sq.topoNr+"/S/distance", args2);
    args2[0]=Atom.newAtom((sq.inPath[3].pl-1)*sq.ws);
    max.outlet(0,"/"+sq.topoNr+"/W/distance", args2);

    args2[0]=Atom.newAtom(sq.gain);
    max.outlet(0,"/"+sq.topoNr+"/gain", args2); 

    if (sq.hasObserver)
    {
        if (sq.seesDirect || sq.hasSource) args2[0]=Atom.newAtom(1);
        else args2[0]=Atom.newAtom(0);
        max.outlet(0,"/direct", args2); 
    }
    */
}

void AcoustiMaze::entered(MazeSquare sq)
{
    // TODO
    /*
    arg[0]=Atom.newAtom(0);
    arg[1]=Atom.newAtom(0);
    arg[2]=Atom.newAtom(0);
    max.outlet(0,"/diffuse/sqverb"+opposite[sq.topoNr]+"/N", arg);
    max.outlet(0,"/diffuse/sqverb"+opposite[sq.topoNr]+"/W", arg);
    max.outlet(0,"/diffuse/sqverb"+opposite[sq.topoNr]+"/S", arg);
    max.outlet(0,"/diffuse/sqverb"+opposite[sq.topoNr]+"/E", arg);
    */
}

void AcoustiMaze::FloorPlan(int arguments[])
{
    xo=arguments[0];
    yo=arguments[1];
    xs=1+arguments[2]-arguments[0];
    ys=1+arguments[3]-arguments[1];
    maze=new MazeSquare[xs][ys];
    for(int i=0;i<xs;i++)
    {
        for(int j=0;j<ys;j++)
        {
            maze[i][j]=new MazeSquare(this, i,j);
        }
    }
}


// TODO osg
void AcoustiMaze::paint(/*Graphics2D g,*/ CoordMatrix m)
{
    for(int i=0;i<xs;i++)
    {
        for(int j=0;j<ys;j++)
        {
            maze[i][j].drawplane(g, m);
        }
    }
    for(int i=0;i<xs;i++)
    {
        for(int j=0;j<ys;j++)
        {
            maze[i][j].drawwalls(g, m);
        }
    }
    for(int i=0;i<xs;i++)
    {
        for(int j=0;j<ys;j++)
        {
            maze[i][j].drawpaths(g, m);
        }
    }
}

void AcoustiMaze::eval(Pose pos, int layer)
{
    for(int i=0;i<xs;i++)
    {
        for(int j=0;j<ys;j++)
        {
            maze[i][j].eval(pos,layer);
        }
    }
}

void AcoustiMaze::reset()
{
    for(int i=0;i<xs;i++)
    {
        for(int j=0;j<ys;j++)
        {
            maze[i][j].reset();
        }
    }


void check_test(int squareX,int squareY,int enterW,int gen)
{
    if (debug)  std::cout << "check_" << std::endl;

    int wall=0;
    for (int ix=0;ix<this.xs;ix++)
    for (int iy=0;iy<this.ys;iy++)
    {
        if (debug) std::cout << "square " << ix << " " << iy << std::endl;

        for (int w=0;w<4;w++) //check wall w (N,E,S,W)
        {
        int jx=ix+chk[w][0]; //determine maze coordinates of adjacent square the wall connects to
        int jy=iy+chk[w][1];
        int iw=(w+2)%4; 
        maze[ix][iy].inPath[w].pl=0;
        
        if(debug) std::cout << "" << w << " " << maze[ix][iy].walls[w] << std::endl;

        if (jx>-1&&jx<xs&&jy>-1&&jy<ys)
        {
             wall=0;
            
             if (!(maze[ix][iy].walls[w]=="void" ) ) wall++;
             if (!(maze[jx][jy].walls[iw]=="void") ) wall++;
             if(debug) std::cout << " ->" << wall << " " << std::endl;
             if (wall==0)
             {
                if(debug) std::cout << " thru ";
                maze[ix][iy].inPath[w].pl=2;
                maze[ix][iy].inPath[w].pts[0][0]=ix;
                maze[ix][iy].inPath[w].pts[0][1]=iy;
                maze[ix][iy].inPath[w].pts[1][0]=jx;
                maze[ix][iy].inPath[w].pts[1][1]=jy;
                if(debug) std::cout << "" << ix << "/" << iy << "-" << jx << "/" << jy << std::endl;
             }
                else
                {   
                    if(debug) std::cout << " blocked";
                    maze[ix][iy].inPath[w].pl=1;
                    maze[ix][iy].inPath[w].pts[0][0]=ix;
                    maze[ix][iy].inPath[w].pts[0][1]=iy;
                }
            }
            else
            {
                if(debug) std::cout << " to outside";
            }
            if(debug) std::cout << std::endl;
        }
        
        if(debug) std::cout << "*" << std::endl;
    }
    
    for(int i=0;i<xs;i++)
    {
        for(int j=0;j<xs;j++)
        {
            maze[i][j].compile();
        }
    }
}

void check(int squareX,int squareY,int enterW,int gen) //check the square, entering from wall enterW, step gen.
{
    int wall=0;
    for (int w=0;w<4;w++) //check wall w (N,E,S,W)
    {
        int iw=(w+2)%4; //the wall opposing w (S,W,N,E)
        if ((!(enterW==w))||(gen==0)) //if w is not where we came in, and this is not the square with the src
        {
            int inX=squareX+chk[w][0]; //determine maze coordinates of adjacent square the wall connects to
            int inY=squareY+chk[w][1];
            isInPathAlready=false;
            for (int op=0;op<maze[squareX][squareY].inPath[enterW].pl-1;op++)
            {
                isInPathAlready=isInPathAlready||((maze[squareX][squareY].inPath[enterW].pts[op][0]==inX) &&(maze[squareX][squareY].inPath[enterW].pts[op][1]==inY)); 
            }    

             if (inX>=0 && inX<xs && inY>=0 && inY<ys && !isInPathAlready)
             {
                wall=0;
                if (!(maze[squareX][squareY].walls[w]=="void" || maze[squareX][squareY].walls[w]=="Dor") ) wall++;
                if (!(maze[inX][inY].walls[iw]=="void" || maze[inX][inY].walls[iw]=="Dor")) wall++;
                if (wall==0) 
                {
                
                    // first, check if we are startig here and if yes, start the recursive mapping process
                    if (gen==0)
                    {
                        maze[inX][inY].inPath[iw].pl=2;
                        maze[inX][inY].inPath[iw].pts[0][0]=squareX;
                        maze[inX][inY].inPath[iw].pts[0][1]=squareY;
                        maze[inX][inY].inPath[iw].pts[1][0]=inX;
                        maze[inX][inY].inPath[iw].pts[1][1]=inY;
                
                        maze[inX][inY].seesDirect=maze[inX][inY].seesDirect||isStraightLine(maze[inX][inY].inPath[iw].pts,maze[inX][inY].inPath[iw].pl);
                        for (int dirN=0;dirN<4;dirN++)
                        {
                          
                        }
                  
                  
                      // post("isStraight: ",maze[inX][inY].seesDirect,inX,inY);post();
                       if (maze[inX][inY].inPath[iw].pl<MazePath.pathLength) //if maximum length of path has not been reached, start recursion.
                       {
                           check(inX,inY,iw,1); 
                       }                                   
                    }

                    //if this is not the first square, extend the path:
                    //first, check if the way we currently arrive at the wall is shorter than we have previously arrived
                    //or if the wall has been reached by a path yet at all.
                    else if (maze[inX][inY].inPath[iw].pl>maze[squareX][squareY].inPath[enterW].pl || maze[inX][inY].reached[iw]==false) 
                    {
                        intVar=maze[squareX][squareY].inPath[enterW].pl;
                        maze[inX][inY].reached[iw]=true;
                        maze[inX][inY].inPath[iw].pl=intVar+1;
                        for (int op=0;op<intVar;op++)
                        {
                            maze[inX][inY].inPath[iw].pts[op][0]=maze[squareX][squareY].inPath[enterW].pts[op][0];
                            maze[inX][inY].inPath[iw].pts[op][1]=maze[squareX][squareY].inPath[enterW].pts[op][1];
                        }
                        maze[inX][inY].inPath[iw].pts[intVar][0]=inX;
                        maze[inX][inY].inPath[iw].pts[intVar][1]=inY;

                        // if(debug){System.out.print("square "+inX+"/"+inY+" "+iw+" ");
                        maze[inX][inY].seesDirect=maze[inX][inY].seesDirect||isStraightLine(maze[inX][inY].inPath[iw].pts,maze[inX][inY].inPath[iw].pl);
                        if (maze[inX][inY].inPath[iw].pl<MazePath.pathLength) 
                        {
                            check(inX,inY,iw,gen+1); 
                        }                                   
                    } 
                }
            }
        }
    }
}


bool AcoustiMaze::isStraightLine(int** line, int l)
{
    xsum=0;
    ysum=0;
    bool result=false;
    for (int i=0;i<l-1;i++)
    {
        xsum+=line[i+1][0]-line[i][0];
        ysum+=line[i+1][1]-line[i][1];
    }
    result=(std::abs(xsum)==l-1)||(std::abs(ysum)==l-1);
    //if(debug){System.out.println("has straight line "+result+" "+xsum+" "+ysum+" "+line.length);
    return result;
}

