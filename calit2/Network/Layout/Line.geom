#version 150 compatibility
#extension GL_EXT_geometry_shader4: enable
#extension GL_ARB_gpu_shader5 : enable

uniform vec3 colorTable[12];
uniform float global_alpha;
uniform float minWeight;
uniform bool colorEdgesToSample;
uniform int simplify;

void
main(void)
{
    // determine scale to use
    //vec4 scaleTest = gl_ModelViewMatrix * vec4(1.0, 0.0, 0.0, 1.0);
    //float distance = max(length(scaleTest.xyz), 1.0); // distance dont want smaller than 1
    //int depth = int(clamp( 1024.0 / distance, 1.0, 1024.0));

    // disable scaling for now to test how view frustrum culling performs
    //if( ((gl_PrimitiveIDIn % simplify) == 0 ) && (gl_FrontColorIn[1].b > minWeight) )
    if( (gl_FrontColorIn[1].b > minWeight) && (gl_FrontColorIn[1].w != -1.0) )
    {

        // do view frustrum cull check of whether to render line
        vec4 boundingBox[8];
        boundingBox[0] = gl_ProjectionMatrix * vec4(max(gl_PositionIn[0].x, gl_PositionIn[1].x),
                                                    max(gl_PositionIn[0].y, gl_PositionIn[1].y),
                                                    max(gl_PositionIn[0].z, gl_PositionIn[1].z),
                                                    1.0); 
        boundingBox[1] = gl_ProjectionMatrix * vec4(min(gl_PositionIn[0].x, gl_PositionIn[1].x),
                                                    max(gl_PositionIn[0].y, gl_PositionIn[1].y),
                                                    max(gl_PositionIn[0].z, gl_PositionIn[1].z),
                                                    1.0); 
        boundingBox[2] = gl_ProjectionMatrix * vec4(max(gl_PositionIn[0].x, gl_PositionIn[1].x),
                                                    min(gl_PositionIn[0].y, gl_PositionIn[1].y),
                                                    max(gl_PositionIn[0].z, gl_PositionIn[1].z),
                                                    1.0); 
        boundingBox[3] = gl_ProjectionMatrix * vec4(min(gl_PositionIn[0].x, gl_PositionIn[1].x),
                                                    min(gl_PositionIn[0].y, gl_PositionIn[1].y),
                                                    max(gl_PositionIn[0].z, gl_PositionIn[1].z),
                                                    1.0); 
        boundingBox[4] = gl_ProjectionMatrix * vec4(max(gl_PositionIn[0].x, gl_PositionIn[1].x),
                                                    max(gl_PositionIn[0].y, gl_PositionIn[1].y),
                                                    min(gl_PositionIn[0].z, gl_PositionIn[1].z),
                                                    1.0); 
        boundingBox[5] = gl_ProjectionMatrix * vec4(min(gl_PositionIn[0].x, gl_PositionIn[1].x),
                                                    max(gl_PositionIn[0].y, gl_PositionIn[1].y),
                                                    min(gl_PositionIn[0].z, gl_PositionIn[1].z),
                                                    1.0); 
        boundingBox[6] = gl_ProjectionMatrix * vec4(max(gl_PositionIn[0].x, gl_PositionIn[1].x),
                                                    min(gl_PositionIn[0].y, gl_PositionIn[1].y),
                                                    min(gl_PositionIn[0].z, gl_PositionIn[1].z),
                                                    1.0); 
        boundingBox[7] = gl_ProjectionMatrix * vec4(min(gl_PositionIn[0].x, gl_PositionIn[1].x),
                                                    min(gl_PositionIn[0].y, gl_PositionIn[1].y),
                                                    min(gl_PositionIn[0].z, gl_PositionIn[1].z),
                                                    1.0); 


        int outOfBound[6] = int[6](0, 0, 0, 0, 0, 0);

        for (int i = 0; i < 8; i++)
        {
            if( boundingBox[i].x >  boundingBox[i].w ) outOfBound[0] = outOfBound[0] + 1;
            if( boundingBox[i].x < -boundingBox[i].w ) outOfBound[1] = outOfBound[1] + 1;
            if( boundingBox[i].y >  boundingBox[i].w ) outOfBound[2] = outOfBound[2] + 1;
            if( boundingBox[i].y < -boundingBox[i].w ) outOfBound[3] = outOfBound[3] + 1;
            if( boundingBox[i].z >  boundingBox[i].w ) outOfBound[4] = outOfBound[4] + 1;
            if( boundingBox[i].z < -boundingBox[i].w ) outOfBound[5] = outOfBound[5] + 1;
        }

        bool inFrustrum = true;

        for( int i = 0; i < 6; i++ )
        {
            if( outOfBound[i] == 8 )
            {
                // not in frustrum dont render    
                inFrustrum = false;
            }
        }

        // online emit if in frustrum
        if( inFrustrum )
        {
            // compute scale (assume it is an ortho scale)
            //float scale = length(vec3(gl_ModelViewMatrix[0])); (TODO for line thickness)


	    // TODO use colorTable and use last color as default
	    vec3 color = vec3(1.0, 1.0, 1.0);
	    
	    // check if use sample color
	    if( colorEdgesToSample )
		color = colorTable[int(gl_FrontColorIn[1].w)]; 

    	    //gl_FrontColor = gl_FrontColorIn[0];
    	    gl_FrontColor.xyz = color;
    	    gl_FrontColor.a = global_alpha * gl_FrontColorIn[1].r;
    	
            gl_BackColor.xyz = color;
    	    gl_BackColor.a = global_alpha * gl_FrontColorIn[1].r;

            //eye_position = gl_PositionIn[0];

    	    gl_Position = gl_ProjectionMatrix * gl_PositionIn[0];
    	    EmitVertex();
    
    	    gl_Position = gl_ProjectionMatrix * gl_PositionIn[1];
    	    EmitVertex();
    
    	    EndPrimitive();
        }
    }
}
