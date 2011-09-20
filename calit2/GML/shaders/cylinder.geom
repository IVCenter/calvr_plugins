#version 120
#extension GL_EXT_geometry_shader4: enable

varying float LightIntensity;
varying in vec3 ls[];
varying in vec3 direction[];

void
ProduceVertex( vec3 v, vec3 ndir)
{
     vec3 n = normalize(ndir);
     vec3 tnorm = normalize( gl_NormalMatrix * n ); // the transformed normal
     vec4 ECposition = gl_ModelViewMatrix * vec4( v, 1. );
     LightIntensity = dot( normalize(ls[0] - ECposition.xyz), tnorm );
     LightIntensity = abs( LightIntensity );
     LightIntensity *= 1.5;
     gl_Position = gl_ProjectionMatrix * ECposition;
     EmitVertex();
}

// produces a 10 sided loop // run out of vertices that can be output
void
ProduceLoop(vec3 point, float radius)
{
	const int size = 10;
	const float y36 = 0.726542528;
	const float y72 = 3.07768537;

	// normals of an 8 sided ring
	vec3 dir[size];
	dir[0] = vec3(-1.0, 0.0, 0.0);
	dir[1] = normalize(vec3(-1.0, y36, 0.0));
	dir[2] = normalize(vec3(-1.0, y72, 0.0));
	dir[3] = normalize(vec3(1.0, y72, 0.0));
	dir[4] = normalize(vec3(1.0, y36, 0.0));
	dir[5] = vec3(1.0, 0.0, 0.0);
	dir[6] = normalize(vec3(1.0, -y36, 0.0));
	dir[7] = normalize(vec3(1.0, -y72, 0.0));
	dir[8] = normalize(vec3(-1.0, -y72, 0.0));
	dir[9] = normalize(vec3(-1.0, -y36, 0.0));

	vec3 dirUp = vec3(0.0, 0.0, 1.0) * gl_FrontColorIn[0].w;

	// the ring generated will be twice the size of the node it is attached too
	for(int i = 0; i < size; i++)
	{
		vec3 dirForward1 = -dir[i] * gl_FrontColorIn[0].w; 
		vec3 dirForward2 = -dir[(i + 1) % size] * gl_FrontColorIn[0].w; 
		vec3 newPos1 = (radius * dir[i]) + (point + vec3(radius, 0.0, 0.0));
		vec3 newPos2 = (radius * dir[(i + 1) % size]) + (point +  vec3(radius, 0.0, 0.0));

		// output a triangle strip cylinder
     		ProduceVertex(newPos1 + dirUp, dirUp); // v4
     		ProduceVertex(newPos2 + dirUp, dirUp); // v0
     		ProduceVertex(newPos1 - dirForward1, -dirForward1); // v7
     		ProduceVertex(newPos2 - dirForward2, -dirForward2); // v3
     		ProduceVertex(newPos1 - dirUp, -dirUp); // v5
     		ProduceVertex(newPos2 - dirUp, -dirUp); // v1
     		ProduceVertex(newPos1 + dirForward1, dirForward1); // v6
     		ProduceVertex(newPos2 + dirForward2, dirForward2); // v2
     		ProduceVertex(newPos1 + dirUp, dirUp); // v4
     		ProduceVertex(newPos2 + dirUp, dirUp); // v0
		EndPrimitive();
	}
}

void
ProduceArrow(float ratio1, float ratio2, vec3 dirLine, vec3 origin, vec3 dirUp, vec3 dirForward)
{

	vec3 dirLine1 = ratio1 * dirLine;
        vec3 dirLine2 = ratio2 * dirLine;

        vec3 newPos1 = dirLine1 + origin;

        // modifies new position position
        vec3 newPos2 = dirLine2 + origin;

        // produces arrow
        ProduceVertex(newPos2 + (4.0 * dirUp), dirUp);
        ProduceVertex(newPos1, dirLine);
        ProduceVertex(newPos2 - (4.0 * dirForward), -dirForward);
        ProduceVertex(newPos1, dirLine);
        ProduceVertex(newPos2 - (4.0 * dirUp), -dirUp);
        ProduceVertex(newPos1, dirLine);
        ProduceVertex(newPos2 + (4.0 * dirForward), dirForward);
        ProduceVertex(newPos1, dirLine);
        ProduceVertex(newPos2 + (4.0 * dirUp), dirUp);
        ProduceVertex(newPos1, dirLine);
        ProduceVertex(newPos2 + (4.0 * dirUp), dirUp);
        EndPrimitive();
}

void
main(void)
{
     // just use first color
     gl_FrontColor = gl_FrontColorIn[0];

	// produce loops (no arrow required on here)
     	if( gl_PositionIn[0].xyz == gl_PositionIn[1].xyz)
     	{
        	ProduceLoop(gl_PositionIn[0].xyz, gl_FrontColorIn[1].r);
     	}
     	else
     	{
        	vec3 dirLine = gl_PositionIn[1].xyz - gl_PositionIn[0].xyz;

		//check if line is shorter than the combined radius of the spheres
		if(length(dirLine) <= gl_FrontColorIn[1].r + gl_FrontColorIn[1].g)
			return;


        	// need to check that line is not in the same direction as the view
        	vec3 parallelCheck = cross(dirLine, direction[0]);
        	if(parallelCheck == vec3(0.0, 0.0, 0.0))
           		return;

        	// cross upvec with direction to eye
        	vec3 dirForward = normalize(cross(direction[0], dirLine)) * gl_FrontColorIn[0].w;

        	// cross direction with resultant (all new vectors are othoginal)
        	vec3 dirUp = normalize(cross(dirForward, dirLine)) * gl_FrontColorIn[0].w;

        	vec3 dirEye = normalize(cross(dirUp, dirLine)) * gl_FrontColorIn[0].w;

        	vec3 newPos1 =  gl_PositionIn[0].xyz;
        	vec3 newPos2 =  gl_PositionIn[1].xyz;

        	//check if source arrow is needed
		if(gl_FrontColorIn[1].b != 0.0)
		{
           		// need to subtract this distance from the end point + 2 times the radius (length of arrow)
           		float dist1 = gl_FrontColorIn[1].r; // diameter of source sphere
           		float dist2 = gl_FrontColorIn[1].r + (8.0 * gl_FrontColorIn[0].w); // diameter of sphere + (radius of line * 8)
           
	   		float lline = length(dirLine);
           		float ratio1 =  (lline - dist1) / lline; // end tip of arrow
           		float ratio2 =  (lline - dist2) / lline; // end of arrow cone
           
	   		// make sure resulting length is not negative (then return and dont draw)
           		if(ratio2 <= 0.0)
                		return;

	   		ProduceArrow(ratio1, ratio2, -dirLine, gl_PositionIn[1].xyz, dirUp, dirForward);

	   		newPos1 = (ratio2 * -dirLine) +  gl_PositionIn[1].xyz;
		}

		//check if target arrow is needed
		if(gl_FrontColorIn[1].a != 0.0)
		{
           		// need to subtract this distance from the end point + 2 times the radius (length of arrow)
           		float dist1 = gl_FrontColorIn[1].g;  // diameter of target sphere
           		float dist2 = gl_FrontColorIn[1].g + (8.0 * gl_FrontColorIn[0].w);  // diameter of sphere + (radius of line * 8)
           
	   		float lline = length(dirLine);
           		float ratio1 =  (lline - dist1) / lline; // end tip of arrow
           		float ratio2 =  (lline - dist2) / lline; // end of arrow cone
           
	   		// make sure resulting length is not negative (then return and dont draw)
           		if(ratio2 <= 0.0)
                		return;

	   		ProduceArrow(ratio1, ratio2, dirLine, gl_PositionIn[0].xyz, dirUp, dirForward);
	   		newPos2 = (ratio2 *  dirLine) +  gl_PositionIn[0].xyz;
		}

        	// produces cylinder
        	ProduceVertex(newPos2 + dirUp, dirUp); // v4
        	ProduceVertex(newPos1 + dirUp, dirUp); // v0
        	ProduceVertex(newPos2 - dirForward, -dirForward); // v7
        	ProduceVertex(newPos1 - dirForward, -dirForward); // v3
        	ProduceVertex(newPos2 - dirUp, -dirUp); // v5
        	ProduceVertex(newPos1 - dirUp, -dirUp); // v1
        	ProduceVertex(newPos2 + dirForward, dirForward); // v6
        	ProduceVertex(newPos1 + dirForward, dirForward); // v2
        	ProduceVertex(newPos2 + dirUp, dirUp); // v4
        	ProduceVertex(newPos1 + dirUp, dirUp); // v0
        	EndPrimitive();
	}
}
