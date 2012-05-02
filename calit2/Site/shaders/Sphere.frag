uniform sampler2D baseTexture;
varying in float sphere_radius;
varying in vec3 vertex_light_position;
varying in vec4 eye_position;

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

    // use color alpha to look up corresponding color
    vec4 result = texture2D(baseTexture, vec2(gl_Color.r));
				
    gl_FragDepth = (pos.z / pos.w + 1.0) / 2.0;
    //gl_FragColor = gl_Color * diffuse_value;
    gl_FragColor = result * diffuse_value;
}
