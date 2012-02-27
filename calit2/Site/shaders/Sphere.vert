void main(void)
{
    gl_TexCoord[0] = gl_MultiTexCoord0;

    gl_FrontColor = gl_Color;

    // return projection position
    gl_Position = gl_ModelViewMatrix * gl_Vertex;
}
