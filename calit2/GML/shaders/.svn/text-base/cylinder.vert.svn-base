varying vec3 ls;
varying vec3 direction;
uniform mat4 mod2obj;
varying float inBounds;

void main()
{
    inBounds = 0.0;
    vec4 npos = gl_ModelViewMatrix * gl_Vertex;
    ls = normalize(gl_LightSource[0].position.xyz - npos.xyz);
    
    direction = -normalize(npos.xyz);

    mat4 modelViewProjectionMatrix = gl_ModelViewProjectionMatrix * mod2obj;

    vec4 result = modelViewProjectionMatrix * gl_Vertex;

    if(result.x > result.w || result.x < -result.w || result.y > result.w
	|| result.y < -result.w || result.z > result.w || result.z < -result.w)
	inBounds = 1.0;
   
    gl_FrontColor = gl_Color; 
    gl_Position = gl_Vertex;

}
