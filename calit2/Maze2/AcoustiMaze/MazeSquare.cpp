
#include "MazeSquare.h"
#include "AcoustiMaze.h"

MazeSquare::MazeSquare(const AcoustiMaze *maz, int arx, int ary)
{
    bool arr[] = { false,false,false,false };

    //reached.assign(arr, arr + sizeof(arr) / sizeof(bool));

    directNeighbors = 0;
	directFrom = 5;
    seesDirect=false; //is the direct sound audible in this square?
    value=1; //what is the shortest path to the source from here?
    hasSource=false; //is the source in this square?
    hasObserver=false; //is the observer in this square?
    isEntrance=false;

    std::string strarr[] = { "void", "void", "void", "void" };
    walls.assign(strarr, strarr + sizeof(strarr) / sizeof(std::string));

    double diffarr[][2] = { {0,0}, {0,0}, {0,0}, {0,0} };
    //diffsrc.assign(diffarr, diffarr + sizeof(diffarr) / sizeof(double)); 

    int intarr[][2] = { {1,1},{-1,1},{-1,-1},{1,-1} };
	//outSq.assign(intarr, intarr + sizeof(intarr) / sizeof(int));
    gain = 0;
    topoNr = 0;
    
    double darr[] = {0,0,0,0};
    inGain.assign(darr, darr + sizeof(darr) / sizeof(double));
    inDelay.assign(darr, darr + sizeof(darr) / sizeof(double));

    obsAngl=0;
	direct=0;

    double floatarr[][6] =
    { //array storing the accumulated material filter of the path
      {0.6,0.3,0.1,0.1,0.1,0.1},
        {0.6,0.3,0.1,0.1,0.1,0.1},
        {0.6,0.3,0.1,0.1,0.1,0.1},
        {0.6,0.3,0.1,0.1,0.1,0.1}
    };
    //inMaterial.assign(floatarr, floatarr + sizeof(floatarr) / sizeof(double));

    x = 0;
    y = 0;
	 
    wx = 0;  //world X of this square
    wy = 0;  //world Y of this square
    ws = 0;
    wbx = 0;  //bounding world X of this square
    wby = 0;  //bounding world Y of this square
    wbs = 0;

    cx = 0;
    cy = 0;

    _X = 0;
    _Y = 0;
    X_ = 0;
    Y_ = 0;



    x=arx; // grid X of this square
    y=ary; // grid Y of this square
    ws=maz->gunit;
    wbs=2*maz->gunit;
    wx=(x-(0.5*maz->xs))*ws;
    wy=(y-(0.5*maz->ys))*ws;
    wbx=(x-(0.5*maz->xs)-0.5)*ws;
    wby=(y-(0.5*maz->ys)-0.5)*ws;

    cx=(x-(0.5*maz->xs)+0.5)*ws;
    cy=(y-(0.5*maz->ys)+0.5)*ws;
    maz=maz;

    for (int i=0; i < inPath.size(); i++)
    {
        inPath[i] = new MazePath(x,y);
    }
    topoNr = arx%3 + (3*(ary%3));
    //std::cout << "topoNr " << topoNr << std::endl;
}

void MazeSquare::reset()
{
    value=0;
    seesDirect=false;
    direct=0;
    directNeighbors=0;
    directFrom=5;
    for (int i=0; i<reached.size(); i++)
    {
        reached[i]=false;
    }
    for (int i=0;i< inPath.size(); i++)
    {
        inPath[i]->reset();
    }
}
   
void MazeSquare::compile()
{
    for (int w=0;w<this->inPath.size();w++) //for all walls N,E,S,W
    {
        this->inPath[w]->compile();
    }
}   

void MazeSquare::wallAngles(Pose pos)
{	
    X_=(pos.x-wx)/ws;
    Y_=(pos.y-wy)/ws;
    _X=1-X_;
    _Y=1-Y_;

    Nu=-atan2(X_,_Y);
    No=atan2(_X,_Y);

    Eu=(M_PI/2)-atan2(_Y,_X);
    Eo=(M_PI/2)+atan2(Y_,_X);

    Su=(M_PI)-atan2(_X,Y_);
    So=(M_PI)+atan2(X_,Y_);

    Wu=(3*M_PI/2)-atan2(Y_,X_);
    Wo=(3*M_PI/2)+atan2(_Y,X_);

    if (Nu<0){Nu+=(2.0*M_PI);No+=(2.0*M_PI);}
    if (Eu<0){Eu+=(2.0*M_PI);Eo+=(2.0*M_PI);}
    if (Su<0){Su+=(2.0*M_PI);So+=(2.0*M_PI);}
    if (Wu<0){Wu+=(2.0*M_PI);Wo+=(2.0*M_PI);}
}

void MazeSquare::eval(Pose pos, int layer)
{
    if (layer==1)
    {
        if (pos.x>wx && pos.y>wy && (pos.x-wx)<ws && (pos.y-wy)<ws) //check if within core square
        {
            this->hasSource=true;
            //if(maz.debug){System.out.println("about to check: "+this.x+" "+this.y);
            maz->reset();
            maz->check(this->x,this->y,0,0);
            maz->srcLX=pos.x-wx;
            maz->srcLY=pos.y-wy;
        }
        else 
        {
            this->hasSource=false;
        }
    }
	   		
    if (layer==0)
    {
        if (pos.x>wx && pos.y>wy && (pos.x-wx)<ws && (pos.y-wy)<ws) //check if within core square
        {
            this->hasObserver=true; 
            dodiffsrc();
            maz->obsLX=pos.x-wx;
            maz->obsLY=pos.y-wy;
            //maz.entered(this);
            if(maz->debug) std::cout << "in: " << x << "/" << y << std::endl;
            wallAngles(pos);

        
            for (int di=0; di < maz->chk.size(); di++)
            {
                inGain[di]=0;
                //if(maz.debug){System.out.print("chk "+(x+maz.chk[di][0])+" "+(y+maz.chk[di][1])+" "+maz.xs+" "+maz.ys);
                if (x+maz->chk[di][0]>=0 && x+maz->chk[di][0]<maz->xs && y+maz->chk[di][1]>=0 
                    && y+maz->chk[di][1]<maz->ys)
                {
                    //	if(maz.debug){System.out.print(" yes ");
                    if(walls[di]=="void" && maz->maze[x+maz->chk[di][0]][y+maz->chk[di][1]]->walls[(di+2)%4]=="void")
                    {
                        maz->maze[x+maz->chk[di][0]][y+maz->chk[di][1]]->checkEntrance(pos, layer);
                        inGain[di]=maz->maze[x+maz->chk[di][0]][y+maz->chk[di][1]]->gain;
                    }
                    else
                    {
                        inGain[di]=0;
                        maz->maze[x+maz->chk[di][0]][y+maz->chk[di][1]]->gain=0;
                        maz->maze[x+maz->chk[di][0]][y+maz->chk[di][1]]->isEntrance=false;
                        maz->maze[x+maz->chk[di][0]][y+maz->chk[di][1]]->notEntrance();
                    }
                    if (inGain[di]<0) inGain[di]=0;
                }
                for (int ou = 0; ou < outSq.size(); ou++)
                {
                    if (x+outSq[ou][0]>0 && y+outSq[ou][1]>0 && x+outSq[ou][0]<maz->xs && y+outSq[ou][1]<maz->ys)
                    {
                        maz->maze[x+outSq[ou][0]][y+outSq[ou][1]]->notEntrance();
                    }
                }
            //else if(maz.debug){System.out.print(" no?!?");
            //if(maz.debug){System.out.println();
            }
            
            lN=cos(inGain[0]*M_PI*0.5);
            lW=cos(inGain[1]*M_PI*0.5);
            lS=cos(inGain[2]*M_PI*0.5);
            lE=cos(inGain[3]*M_PI*0.5);

            getDelays();
            
            gain=1;
            
            //if (this.seesDirect)if(maz.debug){System.out.print("direct_on ");
            //if(maz.debug){System.out.println("paths "+inPath[0].pl+" "+inPath[1].pl+" "+inPath[2].pl+" "+inPath[3].pl);
            maz->soundUpdate(this);
        } 
        else
        {
            this->hasObserver=false;
        }
    }
}

void MazeSquare::getDelays()
{
   for (int dli=0; dli < inDelay.size(); dli++)
    {
        inDelay[dli] = (inPath[dli]->pl-1)*ws;
        strx=0;
        stry=0;
        stoy=0;
        stox=0;
        if (inPath[dli]->pl>1)
        {
            strx=maz->srcLX*abs(inPath[dli]->pts[1][0]-inPath[dli]->pts[0][0]);
            stry=maz->srcLY*abs(inPath[dli]->pts[1][1]-inPath[dli]->pts[0][1]);
            stox=maz->obsLX*abs(inPath[dli]->pts[inPath[dli]->pl-1][0]-inPath[dli]->pts[inPath[dli]->pl-2][0]);
            stoy=maz->obsLY*abs(inPath[dli]->pts[inPath[dli]->pl-1][1]-inPath[dli]->pts[inPath[dli]->pl-2][1]);	
        }
        inDelay[dli]=ws+inDelay[dli]+strx+stry+stox+stoy;
    }
}
   
double MazeSquare::norm(double px, double py)
{
    return sqrt(px*px+py*py);
}

void MazeSquare::dodiffsrc()
{
    diffsrc[0][0]=wx+0.5*ws;
    diffsrc[0][1]=wy+2.*ws;

    diffsrc[1][0]=wx+2.*ws;
    diffsrc[1][1]=wy+0.5*ws;

    diffsrc[2][0]=wx+0.5*ws;
    diffsrc[2][1]=wy-ws;

    diffsrc[3][0]=wx-ws;
    diffsrc[3][1]=wy+0.5*ws;
}

void MazeSquare::checkEntrance(Pose pos, int layer)
{

    if (layer==0)
    {
        dx=pos.x-cx;
        dy=pos.y-cy;
        wallAngles(pos);
        gain=0;
        isEntrance=false;
        if (((pos.x>wx) && (pos.x<wx+ws) && (pos.y>wy+ws) && (pos.y<wy+1.5*ws)) ||
            ((pos.x>wx) && (pos.x<wx+ws) && (pos.y>wy-0.5*ws) && (pos.y<wy)))
        {
            if(maz->debug) std::cout << "North " << x << "/" << y << std::endl;
            observerAngle(pos);
            dx=abs(pos.x-cx);
            dy=abs(pos.y-cy);
            p1y=ws*0.5;
            p1x=p1y*dx/dy;
            p2y=ws;
            p2x=p2y*dx/dy;
            p3x=ws*0.5;
            p3y=p3x*(dy/dx);
            lN=0;
            lW=1;
            lS=1;
            lE=1;
            isEntrance=true;
            gainer();
            if(maz->debug) std::cout << "dis: " << pdis << " " << dis1 << " " << (dis2-dis1) << " " << (dis3-dis1) << std::endl;
        } // North-South entrance check
    

    
    if (((pos.y>wy) && (pos.y<wy+ws) && (pos.x>wx+ws) && (pos.x<wx+1.5*ws)) ||
            ((pos.y>wy) && (pos.y<wy+ws) && (pos.x>wx-0.5*ws) && (pos.x<wx)))
    {
        if(maz->debug) std::cout << "East " << x << "/" << y << std::endl;
        observerAngle(pos);
        dx=abs(pos.x-cx);
        dy=abs(pos.y-cy);
        p1x=ws*0.5;
        p1y=p1x*dy/dx;
        p2x=ws;
        p2y=p2x*dy/dx;
        p3y=ws*0.5;
        p3x=p3y*(dx/dy);
        lN=1;
        lW=1;
        lS=1;
        lE=0;
        isEntrance=true;
        gainer();
        if(maz->debug) std::cout << "dis: " << pdis << " " << dis1 << " " << (dis2-dis1) << " " << (dis3-dis1) << std::endl;
    } // East-West entrance check
    

    if (isEntrance) 
    {
        dodiffsrc();
        maz->soundUpdate(this);
    }
    if (!isEntrance) notEntrance();
    
    if(maz->debug) std::cout << "level: " << gain << " " << lN << " " << lW << " " << lS << " " << lE << std::endl;
    
    }
}

void MazeSquare::notEntrance()
{
    gain=0;
    maz->soundUpdate(this);
}

void MazeSquare::gainer()
{
    pdis=norm(dx,dy)/ws;
    dis1=norm(p1x,p1y)/ws;
    dis2=norm(p2x,p2y)/ws;
    dis3=norm(p3x,p3y)/ws;

    if (dis2>dis3)
    {
        gain=1.0-(pdis-dis1)/(dis3-dis1);
    }
    else
    {
        gain=1.0-(pdis-dis1)/(dis2-dis1);
    }
    if (gain<0) gain+=200;
    getDelays();
}

double MazeSquare::observerAngle(Pose pos)
{
    obsAngl = (atan((pos.y-cy)/(pos.x-cx))) * 180 / M_PI;
    if(maz->debug) std::cout << "obsAngl: " << obsAngl;
    return obsAngl;
}

// TODO
/*void MazeSquare::drawplane(Graphics2D g, CoordMatrix m)
{
    gx=m.xi(wx);
    gy=m.yi(wy);
    gs=m.sxi(ws);
    //determine color: blue if it has observer, green if it has the sound source, shades of grey otherwise
    this.value=0.9;///(min(new Array(this.inPath[0].pts.length,this.inPath[1].pts.length,this.inPath[2].pts.length,this.inPath[3].pts.length))/2.0);
    g.setColor(new Color((int)(255*1-0.5*this.value),(int)(255*1-0.5*this.value),(int)(255*1-0.5*this.value)));
    g.setColor(Color.lightGray);
    if (this.hasObserver) { g.setColor(Color.orange); }
    if (this.hasSource) { g.setColor(Color.blue); }
    //  if(maz.debug){System.out.println("square "+gx+" "+gy+" "+gs+" "+wx+" "+wy+" "+ws);
    g.fillRect(gx,gy-gs,gs,gs);
}


// TODO
void MazeSquare::drawwalls(Graphics2D g, CoordMatrix m)
{
    g.setColor(Color.red);
    gx=m.xi(wx);
    gy=m.yi(wy);
    gs=m.sxi(ws);
    
    g.fillOval(gx-2,gy-gs-2,4,4);
    g.fillOval(gx+gs-2,gy-gs-2,4,4);
    g.fillOval(gx+gs-2,gy-2,4,4);
    g.fillOval(gx-2,gy-2,4,4);
    g.setColor(Color.lightGray);
    //g.drawLine(gx,gy-gs,gx+gs,gy-gs);
    //g.drawLine(gx+gs,gy-gs,gx+gs,gy);
    //g.drawLine(gx+gs,gy,gx,gy);
   // g.drawLine(gx,gy,gx,gy-gs);
    g.setColor(Color.red);
    if (this.walls[0]!="void") g.drawLine(gx,gy-gs,gx+gs,gy-gs);
    if (this.walls[1]!="void") g.drawLine(gx+gs,gy-gs,gx+gs,gy);
    if (this.walls[2]!="void") g.drawLine(gx+gs,gy,gx,gy);
    if (this.walls[3]!="void") g.drawLine(gx,gy,gx,gy-gs);
   // if(maz.debug){System.out.println("walls: N"+walls[0]+" E"+walls[1]+" S"+walls[2]+" W"+walls[3]);
}


// TODO
void MazeSquare::drawpaths(Graphics2D g, CoordMatrix m)
{
    if (this.hasObserver)
    {	
        for (int i=0;i<inPath.length;i++)
        {	
            inPath[i].draw(g,m, this, i);
        }
    }
}
*/
