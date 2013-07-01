
#include "MXJ_AcoustiMaze.h"



namespace AcoustiMaze
{

MXJ_AcoustiMaze::MXJ_AcoustiMaze(Atom[] args)
{			
    Pose.gr=1.0;
    Pose.gn=0.4;
    obs=new Pose(0);
    src=new Pose(1);
    maze=new AcoustiMaze(this);
    declareAttribute("oma");
    declareAttribute("count");
   
    // TODO
    declareInlets(new int[]{DataTypes.ALL,DataTypes.ALL,DataTypes.ALL});
    declareOutlets(new int[]{DataTypes.ALL});
    
    setInletAssist(INLET_ASSIST);
    setOutletAssist(OUTLET_ASSIST);
    
    // TODO 
    graphicsFrame=new DrawFrame("Map",null,2);
    graphicsFrame.show();
    graphicsFrame.toFront();

    graphicsFrame.gUpdate.add(this);
    graphicsFrame.map.set(0,0,10., graphicsFrame.getWidth(), graphicsFrame.getHeight());
}

void MXJ_AcoustiMaze::bang()
{
}


// TODO
void MXJ_AcoustiMaze::redraw(Graphics2D g, CoordMatrix m)
{
    g.setColor(Color.blue);
    maze.paint(g,m);
    g.setColor(Color.red);
    obs.paint(g, m);
    g.setColor(Color.cyan);
    src.paint(g,m);
    //g.drawString(dString, 100,100);
}

void MXJ_AcoustiMaze::inlet(int i)
{
    graphicsFrame.redraw();
}

void MXJ_AcoustiMaze::inlet(float f)
{
    
}

void MXJ_AcoustiMaze::anything(String message, Atom[] args)
{
    String[] OSCmsg=message.split("/");
    if (OSCmsg[1].equals("pose"))
    {
        if (args[0].getInt()==0) 
        {
            obs.set(args); 
            maze.eval(obs,0);
        }
        if (args[0].getInt()==1) 
        {
            src.set(args); 
            maze.eval(src,1);
        }
    //	System.out.println("posein: "+args[0].getInt());
    }
    
    if (OSCmsg[1].equals("floorplan"))
    {
        int[] arg=new int[4];
        for (int i=0;i<arg.length;i++) 
        {
            arg[i]=args[i].getInt();
            //System.out.println("arg"+i+" "+arg[i]);
        }
        maze.FloorPlan(arg);
    }
    if (OSCmsg[1].equals("square"))
    {
        int sqx,sqy,wall=0;
        String par;
        sqx=Integer.valueOf(OSCmsg[2]).intValue();
        sqy=Integer.valueOf(OSCmsg[3]).intValue();
        if(OSCmsg[4].equals("N")) {wall=0;};
        if(OSCmsg[4].equals("E")) {wall=1;};
        if(OSCmsg[4].equals("S")) {wall=2;};
        if(OSCmsg[4].equals("W")) {wall=3;};
        par=OSCmsg[5];
        System.out.println("squaremsg: "+sqx+" "+sqy+" "+wall+" "+args[0].toString());
        if (par.equals("geom"))
        {
            maze.maze[sqx-maze.xo][sqy-maze.yo].walls[wall]=args[0].toString();
        }
    }
}

void MXJ_AcoustiMaze::list(Atom[] list)
{
    if (getInlet() == 0) 
        {dString=list[0].toString();}
    
    graphicsFrame.redraw();
}

};
