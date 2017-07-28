#version 150 compatibility
#extension GL_EXT_geometry_shader4: enable
#extension GL_ARB_gpu_shader5 : enable

uniform vec3 colorTable[12];
uniform float global_alpha;
uniform float point_size;
uniform float minWeight;
uniform int minEdges;
flat out vec4 eye_position;
flat out float sphere_radius;

void
main(void)
{
    // filter points
    if( (gl_FrontColorIn[0].r > minEdges) && (gl_FrontColorIn[0].b > minWeight) && (gl_FrontColorIn[0].w != -1.0))
    {
        // compute scale (assume it is an ortho scale)
        float scale = length(vec3(gl_ModelViewMatrix[0]));

        // compute point size
        sphere_radius = scale * point_size;
    	float halfsize = sphere_radius * 0.5;

		eye_position = gl_PositionIn[0];

		// try view frustrum culling
		vec4 boundingBox[8];
        boundingBox[0] = gl_ProjectionMatrix * (gl_PositionIn[0] + vec4( halfsize,  halfsize, halfsize, 0.0));
		boundingBox[1] = gl_ProjectionMatrix * (gl_PositionIn[0] + vec4(-halfsize,  halfsize, halfsize, 0.0));
		boundingBox[2] = gl_ProjectionMatrix * (gl_PositionIn[0] + vec4( halfsize, -halfsize, halfsize, 0.0));	
		boundingBox[3] = gl_ProjectionMatrix * (gl_PositionIn[0] + vec4(-halfsize, -halfsize, halfsize, 0.0));
		boundingBox[4] = gl_ProjectionMatrix * (gl_PositionIn[0] + vec4( halfsize,  halfsize, -halfsize, 0.0));
		boundingBox[5] = gl_ProjectionMatrix * (gl_PositionIn[0] + vec4(-halfsize,  halfsize, -halfsize, 0.0));
		boundingBox[6] = gl_ProjectionMatrix * (gl_PositionIn[0] + vec4( halfsize, -halfsize, -halfsize, 0.0));
		boundingBox[7] = gl_ProjectionMatrix * (gl_PositionIn[0] + vec4(-halfsize, -halfsize, -halfsize, 0.0));

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

		if( inFrustrum )
		{
			//gl_FrontColor = gl_FrontColorIn[0];
			gl_FrontColor.xyz = colorTable[int(gl_FrontColorIn[0].w)];
			gl_FrontColor.a = global_alpha;

		    //eye_position = gl_PositionIn[0];

			gl_TexCoord[0].st = vec2(-1.0,-1.0);
			gl_Position = gl_PositionIn[0];
			gl_Position.xy += vec2(-halfsize, -halfsize);
			gl_Position = gl_ProjectionMatrix * gl_Position;
			EmitVertex();
		
			gl_TexCoord[0].st = vec2(-1.0,1.0);
			gl_Position = gl_PositionIn[0];
			gl_Position.xy += vec2(-halfsize, halfsize);
			gl_Position = gl_ProjectionMatrix * gl_Position;
			EmitVertex();
		
			gl_TexCoord[0].st = vec2(1.0,-1.0);
			gl_Position = gl_PositionIn[0];
			gl_Position.xy += vec2(halfsize, -halfsize);
			gl_Position = gl_ProjectionMatrix * gl_Position;
			EmitVertex();
		
			gl_TexCoord[0].st = vec2(1.0,1.0);
			gl_Position = gl_PositionIn[0];
			gl_Position.xy += vec2(halfsize, halfsize);
			gl_Position = gl_ProjectionMatrix * gl_Position;
			EmitVertex();

			EndPrimitive();
		}
    }
}
