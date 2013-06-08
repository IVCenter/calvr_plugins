
#include "Pose.h"

static double gr=20,gn=5;

Pose::Pose(int i)
{
    index=i;
}

// TODO
void Pose::set(/*Atom[] args*/)
{
/*
    x=(float)(args[1].getFloat());
    y=(float)(args[2].getFloat());
    z=(float)(args[3].getFloat());
    yaw=(float)(args[4].getFloat());
    pitch=(float)(args[5].getFloat());
    roll=(float)(args[6].getFloat());
    */
}

void Pose::set(float x,float y,float z,float yaw, float pitch,float roll)
{
    this->x=x;
    this->y=y;
    this->z=z;
    this->yaw=yaw;
    this->pitch=pitch;
    this->roll=roll;
}

// TODO
/*void Pose::paint(Graphics2D g, CoordMatrix m)
{
    px=m.xi(x);
    py=m.yi(y);
    ps=m.sxi(gr);
    pnx=(int)(px-0.5*ps*sin(M_PI * (yaw) / 180));
    pny=(int)(py-0.5*ps*cos(M_PI * (yaw) / 180));
    pns=m.sxi(gn);
    
    g.fillOval(px-(ps/2),py-(ps/2),ps,ps);
    g.setColor(Color.DARK_GRAY);
    g.fillOval(pnx-(pns/2),pny-(pns/2),pns,pns);
    
}*/

