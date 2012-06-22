#include "SphereDrawable.h"

#include <cvrKernel/ScreenConfig.h>

using namespace osg;
using namespace std;
using namespace cvr;

SphereDrawable::SphereDrawable(float radius_in, float viewanglev_in, float viewangleh_in, float camHeight_in, int segmentsPerTexture_in, int maxTextureSize_in) : PanoDrawable(radius_in,viewanglev_in,viewangleh_in,camHeight_in,segmentsPerTexture_in,maxTextureSize_in)
{
}


SphereDrawable::~SphereDrawable()
{
}

SphereDrawable::SphereDrawable(const SphereDrawable& sd,const osg::CopyOp& copyop) : PanoDrawable(sd,copyop)
{
}

void SphereDrawable::drawShape(PanoDrawable::eye eye, int context) const
{
    float radsPerTileH = (viewangleh * (M_PI / 180.0f)) / ((float)cols);
    float radsPerTileV = (viewanglev * (M_PI / 180.0f)) / ((float)rows);
    float radsPerSegmentH = radsPerTileH / ((float)segmentsPerTexture);
    float radsPerSegmentV = radsPerTileV / ((float)segmentsPerTexture);

    //float theight = tan((viewangle/360.0)* PI) * radius * 2.0;

    glPushAttrib(GL_ALL_ATTRIB_BITS);  

    float AdcamHeight = camHeight + floorOffset;


    glDisable(GL_LIGHTING);
    glEnable(GL_TEXTURE_2D);

    float radsv = ((180.0f - viewanglev) / 2.0f) * (M_PI / 180.0f); 
    for(int i = rows - 1; i >= 0; i--)
    {
        float radsh = _rotation;
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
	    float temprads = radsv;
            for(int k = 0; k < segmentsPerTexture; k++)
            {
		radsv = temprads;
		for(int m = 0; m < segmentsPerTexture; m++)
		{
                    float h = radsh;
                    float hpo = radsh + radsPerSegmentH;
                    if(h > (2.0f * M_PI))
                    {
                        h -= (2.0f * M_PI);
                    }
                    if(hpo > (2.0f * M_PI))
                    {
                        hpo -= (2.0f * M_PI);
                    }
		    glTexCoord2f(((float)k) / ((float)segmentsPerTexture), (1.0f - ((float)(m+1)) / ((float)segmentsPerTexture)));
                    //glTexCoord2f((1.0f - ((float)(m+1)) / ((float)segmentsPerTexture)), ((float)k) / ((float)segmentsPerTexture));
		    glVertex3f(radius * cos(-h) * sin(radsv + radsPerSegmentV), radius * sin(-h) * sin(radsv + radsPerSegmentV), (radius * cos(radsv + radsPerSegmentV)) - AdcamHeight);	// Bottom Left Of The Texture and Quad
		    glTexCoord2f(((float)k+1.0) / ((float)segmentsPerTexture), (1.0f - ((float)(m+1)) / ((float)segmentsPerTexture))); 
                    //glTexCoord2f((1.0f - ((float)(m+1)) / ((float)segmentsPerTexture)), ((float)k+1.0) / ((float)segmentsPerTexture));
		    glVertex3f(radius * cos(-hpo) * sin(radsv + radsPerSegmentV), radius * sin(-hpo) * sin(radsv + radsPerSegmentV), (radius * cos(radsv + radsPerSegmentV)) - AdcamHeight);	// Bottom Right Of The Texture and Quad
		    glTexCoord2f(((float)k+1.0) / ((float)segmentsPerTexture), (1.0f - ((float)m) / ((float)segmentsPerTexture)));
                    //glTexCoord2f((1.0f - ((float)m) / ((float)segmentsPerTexture)), ((float)k+1.0) / ((float)segmentsPerTexture));
		    glVertex3f(radius * cos(-hpo) * sin(radsv), radius * sin(-hpo) * sin(radsv), (radius * cos(radsv)) - AdcamHeight);	// Top Right Of The Texture and Quad
		    glTexCoord2f(((float)k) / ((float)segmentsPerTexture), (1.0f - ((float)m) / ((float)segmentsPerTexture)));
                    //glTexCoord2f((1.0f - ((float)m) / ((float)segmentsPerTexture)), ((float)k) / ((float)segmentsPerTexture));
		    glVertex3f(radius * cos(-h) * sin(radsv), radius * sin(-h) * sin(radsv), (radius * cos(radsv)) - AdcamHeight);	// Top Left Of The Texture and Quad
		    radsv += radsPerSegmentV;
		}
		radsh += radsPerSegmentH;
            }
            glEnd();
	    radsv = temprads;
        }
	radsv += radsPerTileV;
    }
    glPopAttrib();
}
