// simple vertex shader
#define ANG2RAD 3.14159265358979323846/180.0

varying vec3 ls;
varying vec3 direction;
varying vec3 up;
varying vec3 right;
uniform mat4 obj2mod;

void main()
{
    vec4 npos = gl_ModelViewMatrix * gl_Vertex;
    //vec4 npos = obj2mod * gl_Vertex;

    // doing frustrum cull check

    ls = normalize(gl_LightSource[0].position.xyz - npos.xyz);

    //mat4 modelViewMatrixInverse = gl_ModelViewMatrixInverse * obj2mod;
    //mat4 modelViewMatrixInverse = gl_ModelViewMatrixInverse;

    //vec3 W = vec3(modelViewMatrixInverse[2][0],modelViewMatrixInverse[2][1],modelViewMatrixInverse[2][2]);

    vec3 W = vec3(0.0, 0.0, 1.0); 

    vec3 U = -vec3(npos.x, npos.z, -npos.y); //rotate back to world co-ordinates
    vec3 V = cross(U, W);
    W = cross(U, V);
 
    up = normalize(W) * gl_Color.w;
    right = normalize(V) * gl_Color.w;
    direction = normalize(U) * gl_Color.w;

    gl_FrontColor = gl_Color;
    gl_Position = gl_Vertex;
}
