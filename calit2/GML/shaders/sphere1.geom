#version 120
#extension GL_EXT_geometry_shader4: enable

varying float LightIntensity;
varying in vec3 ls[];
varying in vec3 direction[];
varying in vec3 up[];
varying in vec3 right[];

void
ProduceVertex( float s, float t, vec3 vc0, vec3 vc1, vec3 vc2, vec3 position, float radius)
{
     vec3 v = vc0 + s*vc1 + t*vc2;
     v = normalize(v);
     vec3 n = v;
     vec3 tnorm = normalize( gl_NormalMatrix * n ); // the transformed normal
     vec4 ECposition = gl_ModelViewMatrix * vec4( (radius*v) + position, 1. );
     LightIntensity = dot( normalize(ls[0] - ECposition.xyz), tnorm );
     LightIntensity = abs( LightIntensity );
     LightIntensity *= 1.5;
     gl_Position = gl_ProjectionMatrix * ECposition;
     EmitVertex();
}

void 
generateSurfaces(vec3 p0, vec3 p1, vec3 p2, vec3 position, float radius)
{
    int level = 2;
    int numLayers = 1 << level;
    float dt = 1. / float( numLayers );      	
    float t_top = 1.;
    for( int it = 0; it < numLayers; it++ )
    {
  		float t_bot = t_top - dt;
  		float smax_top = 1. - t_top;
 	 	float smax_bot = 1. - t_bot;
  		int nums = it + 1;
  		float ds_top = smax_top / float( nums - 1 );
  		float ds_bot = smax_bot / float( nums );
  		float s_top = 0.;
  		float s_bot = 0.;
  		for( int is = 0; is < nums; is++ )
  		{
       			ProduceVertex( s_bot, t_bot, p0, p1, p2, position, radius);
       			ProduceVertex( s_top, t_top, p0, p1, p2, position, radius);
       			s_top += ds_top;
       			s_bot += ds_bot;
  		}
  		ProduceVertex( s_bot, t_bot, p0, p1, p2, position, radius);
  		EndPrimitive();
  		t_top = t_bot;
  		t_bot -= dt;
	}
}

void
main()
{
     //generate V0 V01 V02 based on location of point and face the user
     for(int i = 0; i < gl_VerticesIn; i++)
     {
	gl_FrontColor  = gl_FrontColorIn[ i ];

     	// triangles (v0, v2, v4) (v0, v4, v3) (v0, v3, v5) (v0, v5, v2)
     	generateSurfaces(direction[i], up[i] - direction[i], right[i] - direction[i],  gl_PositionIn[i].xyz, gl_FrontColorIn[i].w);
     	generateSurfaces(direction[i], -right[i] - direction[i], up[i] - direction[i],  gl_PositionIn[i].xyz, gl_FrontColorIn[i].w);
     	generateSurfaces(direction[i], -up[i] - direction[i], -right[i] - direction[i],  gl_PositionIn[i].xyz, gl_FrontColorIn[i].w);
     	generateSurfaces(direction[i], right[i] - direction[i], -up[i] - direction[i],  gl_PositionIn[i].xyz, gl_FrontColorIn[i].w);
     }
}
