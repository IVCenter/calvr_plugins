#version 150 compatibility
#extension GL_ARB_gpu_shader5 : enable

flat in float sphere_radius;
flat in vec3 vertex_light_position;
flat in vec4 eye_position;

void main (void)
{

    float x = gl_TexCoord[0].x;
    float y = gl_TexCoord[0].y;
    float zz = 1.0 - x*x - y*y;

    if (zz <= 0.0 )
    	discard;

    float z = sqrt(zz);

    vec3 normal = vec3(x, y, z);

    // Lighting
    float diffuse_value = max(dot(normal, vertex_light_position), 0.0);
		
    vec4 pos = eye_position;
    pos.z += z*sphere_radius;
    pos = gl_ProjectionMatrix * pos;
				
    gl_FragDepth = (pos.z / pos.w + 1.0) / 2.0;
    gl_FragColor.rgb = gl_Color.rgb * diffuse_value;
    gl_FragColor.a = gl_Color.a;
}
