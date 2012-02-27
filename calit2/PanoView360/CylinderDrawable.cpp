#include "CylinderDrawable.h"

#include <kernel/ScreenConfig.h>

using namespace osg;
using namespace std;
using namespace cvr;

CylinderDrawable::CylinderDrawable(float radius_in, float viewanglev_in, float viewangleh_in, float camHeight_in, int segmentsPerTexture_in, int maxTextureSize_in) : PanoDrawable(radius_in,viewanglev_in,viewangleh_in,camHeight_in,segmentsPerTexture_in,maxTextureSize_in)
{
}


CylinderDrawable::~CylinderDrawable()
{
}

void CylinderDrawable::drawShape(PanoDrawable::eye eye,int context) const
{
    float radsPerTile = (2.0 * PI * (viewangleh / 360.0)) / ((float)cols);
    float radsPerSegment = radsPerTile / ((float)segmentsPerTexture);

    //float tlength = (2.0 * PI * radius) /((float)cols);
    //float theight = tlength *(((float)rows) / ((float)cols));
    float theight = tan((viewanglev/360.0)* PI) * radius * 2.0;
    

    glPushAttrib(GL_ALL_ATTRIB_BITS);  

    float AdcamHeight = camHeight + floorOffset + (theight / 2.0);


    glDisable(GL_LIGHTING);
    glEnable(GL_TEXTURE_2D);
    //glDisable(GL_DEPTH_TEST);
    //glDepthFunc(GL_ALWAYS);
    //glEnable(GL_BLEND);
    //glEnable(GL_POLYGON_SMOOTH);
    //glBlendFunc(GL_ONE, GL_ZERO);
    //glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    //glHint(GL_POINT_SMOOTH, GL_NICEST);
    //glHint(GL_LINE_SMOOTH, GL_NICEST);
    //glHint(GL_POLYGON_SMOOTH, GL_NICEST);

    //glEnable(GL_POINT_SMOOTH);
    //glEnable(GL_LINE_SMOOTH);
    //glEnable(GL_POLYGON_SMOOTH);

    //glClearColor(0.0, 0.0, 0.0, 0.0);
    //glClear(GL_COLOR_BUFFER_BIT);

    if(currenteye)
    {
	currenteye = 0;
    }
    else
    {
	currenteye = 1;
    }

    for(int i = 0; i < rows; i++)
    {
        //float rads = 0;
	float rads = _rotation;
        for(int j = 0; j < cols; j++)
        {
	    if(eye == RIGHT || (ScreenConfig::instance()->getEyeSeparationMultiplier() == 0.0))
            {
		if(eye == LEFT && !(_contextinit[context] == BOTH))
		{
		    glPopAttrib();
		    return;
		}
                glBindTexture(GL_TEXTURE_2D, *(rtextures[context][i][j]));
            }
            else
            {
                glBindTexture(GL_TEXTURE_2D, *(ltextures[context][i][j]));
            }
            glBegin(GL_QUADS);
            for(int k = 0; k < segmentsPerTexture; k++)
            {
                //glTexCoord2f(((float)k) / ((float)segmentsPerTexture), ((float)(rows - (i + 1))) / ((float)rows)); 
                glTexCoord2f(((float)k) / ((float)segmentsPerTexture), 0.0);
                glVertex3f(-(cos(rads)*radius), sin(rads)*radius,  ((((float)(i)) / ((float)rows))*theight) - AdcamHeight);	// Bottom Left Of The Texture and Quad
                //glTexCoord2f(((float)k+1.0) / ((float)segmentsPerTexture), ((float)(rows - (i + 1))) / ((float)rows));
                glTexCoord2f(((float)k+1.0) / ((float)segmentsPerTexture), 0.0); 
                glVertex3f(-(cos(rads+radsPerSegment)*radius), sin(rads+radsPerSegment)*radius,  ((((float)(i)) / ((float)rows))*theight) - AdcamHeight);	// Bottom Right Of The Texture and Quad
                //glTexCoord2f(((float)k+1.0) / ((float)segmentsPerTexture), ((float)(rows - (i))) / ((float)rows)); 
                glTexCoord2f(((float)k+1.0) / ((float)segmentsPerTexture), 1.0);
                glVertex3f(-(cos(rads+radsPerSegment)*radius), sin(rads+radsPerSegment)*radius,  ((((float)(i+1.0)) / ((float)rows))*theight) - AdcamHeight);	// Top Right Of The Texture and Quad
                //glTexCoord2f(((float)k) / ((float)segmentsPerTexture), ((float)(rows - (i))) / ((float)rows)); 
                glTexCoord2f(((float)k) / ((float)segmentsPerTexture), 1.0);
                glVertex3f(-(cos(rads)*radius), sin(rads)*radius,  ((((float)(i+1.0)) / ((float)rows))*theight) - AdcamHeight);	// Top Left Of The Texture and Quad
                rads += radsPerSegment;
            }
            glEnd();
        }
    }
    glPopAttrib();
}
